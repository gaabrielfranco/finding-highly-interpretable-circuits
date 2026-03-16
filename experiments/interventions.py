"""Causal interventions on circuit edges (Appendix E).

Runs local interventions on curated or all edges, measuring the effect on
logit difference. Produces parquet files (curated) or intervention graphml
files (all edges).

This is Step 3 of the Appendix E pipeline.

Usage:
    # Curated edges (Step 3a)
    python experiments/interventions.py -m gpt2-small -t ioi -n 256 -d mps

    # All edges from unified graph (Step 3b)
    python experiments/interventions.py -m gpt2-small -t ioi -n 256 -d mps -ig

"""

import argparse
from copy import deepcopy
import glob
import os
import sys
from typing import Tuple
from einops import einsum
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
from torch import Tensor
import networkx as nx

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer.models import get_model_config, ModelConfig
from accpp_tracer.decomposition import get_omega_decomposition, compute_weight_pseudoinverses
from accpp_tracer.rope import get_rotation_matrix
from accpp_tracer.datasets import IOIDataset, GenderedPronoun, YearDataset, get_valid_years

torch.set_grad_enabled(False)

def load_graphs(model: str, task: str, interventions: bool = False) -> dict:
    if not interventions:
        graph_files = sorted(glob.glob(f"data/combined_graphs/{model}/{task}/*"))
    else:
        graph_files = sorted(glob.glob(f"data/combined_graphs_intervention/{model}/{task}/*"))

    if not graph_files:
        raise FileNotFoundError(f"No combined graph files found for {model}/{task} (interventions={interventions})")

    GRAPHS = {}
    for file in graph_files:
        thresh = file.split("_")[-1].split(".graphml")[0]
        GRAPHS[thresh] = nx.read_graphml(file, force_multigraph=True)

    return GRAPHS

def compute_projections(basis: Float[Tensor, "d_model rank"],
                        svs_used: np.array,
                        random: np.random.RandomState) -> Tuple[Float[Tensor, "d_model d_model"],
                                                                Float[Tensor, "d_model d_model"]]:
    B_s = basis[:, svs_used]
    P = B_s @ B_s.T

    # Random projection
    rank = basis.shape[1]
    svs_not_used = [i for i in range(rank) if i not in svs_used]

    if len(svs_used) > rank - len(svs_used):
        # Sample random SVs from the complement with the same number of SVs
        SVs_rand = random.choice(svs_not_used, rank - len(svs_used), replace=False)
    else:
        # Sample random SVs from the complement with the same number of SVs
        SVs_rand = random.choice(svs_not_used, len(svs_used), replace=False)

    # Compute the random projections
    B_s_rand = basis[:, SVs_rand]

    P_rand = B_s_rand @ B_s_rand.T

    return P, P_rand

def run_local_intervention(model: HookedTransformer,
                       model_name: str,
                       cache: ActivationCache,
                       dataset: IOIDataset | GenderedPronoun | YearDataset,
                       layer: int,
                       ah_idx: int,
                       interv_value: Float[Tensor, "n_prompts n_tokens d_model"],
                       intervention_type: str,
                       is_boosting: bool,
                       projection_type: str) -> Tuple[Float[Tensor, "n_prompts n_tokens d_vocab"],
                                                   ActivationCache,
                                                   Float[Tensor, "n_prompts n_tokens d_model"]]:
    if intervention_type != "local":
        raise ValueError(f"Unknown intervention type {intervention_type}")

    # Run with hooks using the interv_value to intervene
    def intervention_fn_b(x, hook, ah_idx, interv_value, group_value_idx=False):
        print(f"Intervention local in {hook.name}")
        if group_value_idx: # Special case for Gemma value matrix.
            x[:, :, ah_idx // 2, :] = interv_value
        else:
            x[:, :, ah_idx, :] = interv_value
        return x

    print(f"Running intervention with intervention_type={intervention_type} with is_boosting={is_boosting}")

    # Getting the attention input (it's the same for all heads)
    attn_input = cache[f"blocks.{layer}.ln1.hook_normalized"]

    # Inverting the attention input
    attn_input_interv = deepcopy(attn_input)
    if is_boosting:
        attn_input_interv += interv_value
    else:
        attn_input_interv -= interv_value

    # Reset hooks
    model.reset_hooks(including_permanent=True)

    # Computing the Q, K, V after the intervention
    # We use these values to do the actual intervention
    if projection_type == "U":
        q_interv = F.linear(attn_input_interv, model.W_Q[layer, ah_idx, :, :].T, model.b_Q[layer, ah_idx, :])
        hook_fn_q = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=q_interv)

        # Adds a hook into global model state
        model.blocks[layer].attn.hook_q.add_hook(hook_fn_q)
    else:
        k_interv = F.linear(attn_input_interv, model.W_K[layer, ah_idx, :, :].T, model.b_K[layer, ah_idx, :])
        if model_name == "gemma-2-2b":
            hook_fn_k = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=k_interv, group_value_idx=True)
        else:
            hook_fn_k = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=k_interv)

        # Adds a hook into global model state
        model.blocks[layer].attn.hook_k.add_hook(hook_fn_k)

    # Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
    interv_logits, interv_cache = model.run_with_cache(dataset.toks)

    # Remove the hook
    model.reset_hooks(including_permanent=True)

    return interv_logits, interv_cache, attn_input_interv

def run_global_intervention(model: HookedTransformer,
                       dataset: IOIDataset | GenderedPronoun | YearDataset,
                       layer: int,
                       ah_idx: int,
                       interv_value: Float[Tensor, "n_prompts n_tokens d_model"],
                       intervention_type: str,
                       is_boosting: bool) -> Tuple[Float[Tensor, "n_prompts n_tokens d_vocab"],
                                                   ActivationCache,
                                                   Float[Tensor, "n_prompts n_tokens d_model"]]:

    def intervention_fn(x, hook, is_boosting, interv_value):
        assert x.shape == interv_value.shape
        # Intervention is boosting, so we sum the intervention value
        if is_boosting:
            x += interv_value
        else:
            x -= interv_value
        return x

    # Reset hooks
    model.reset_hooks(including_permanent=True)

    hook_fn = partial(intervention_fn, is_boosting=is_boosting, interv_value=interv_value)

    if ah_idx < model.cfg.n_heads or ah_idx == model.cfg.n_heads+1: # AH or attn bias
        # Adds a hook into global model state
        model.blocks[layer].hook_attn_out.add_hook(hook_fn)
    elif ah_idx == model.cfg.n_heads: # MLP case
        model.blocks[layer].hook_mlp_out.add_hook(hook_fn)
    elif ah_idx == model.cfg.n_heads+2:
        model.blocks[0].hook_resid_pre.add_hook(hook_fn)
    else:
        raise Exception(f"Invalid ah_idx={ah_idx}")

    # Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
    interv_logits, interv_cache = model.run_with_cache(dataset.toks)

    # Remove the hook
    model.reset_hooks(including_permanent=True)

    return interv_logits, interv_cache

def compute_logit_diff(dataset: IOIDataset | GenderedPronoun | YearDataset,
                       logits: Float[Tensor, "n_prompts n_tokens d_vocab"],
                       logits_interv: Float[Tensor, "n_prompts n_tokens d_vocab"],
                       device: str = "cpu") -> Tuple[Float[Tensor, "n_prompts"],
                                                     Float[Tensor, "n_prompts"]]:
    n_prompts = logits.size(0)  # Number of prompts

    prompts_array = torch.arange(n_prompts, device=device)

    if isinstance(dataset, IOIDataset):
        # # Getting the logit of the IO token (END position) for each prompt
        last_token_logits_io = deepcopy(logits[prompts_array, dataset.word_idx["end"], dataset.io_tokenIDs])
        last_token_logits_s = deepcopy(logits[prompts_array, dataset.word_idx["end"], dataset.s_tokenIDs])
        logit_diff = last_token_logits_io - last_token_logits_s

        last_token_logits_io_interv = deepcopy(logits_interv[prompts_array, dataset.word_idx["end"], dataset.io_tokenIDs])
        last_token_logits_s_interv = deepcopy(logits_interv[prompts_array, dataset.word_idx["end"], dataset.s_tokenIDs])
        logit_diff_interv = last_token_logits_io_interv - last_token_logits_s_interv
    elif isinstance(dataset, GenderedPronoun):
        last_token_logits_answers = deepcopy(logits[prompts_array, dataset.word_idx["end"], dataset.answers])
        last_token_logits_wrongs = deepcopy(logits[prompts_array, dataset.word_idx["end"], dataset.wrongs])
        logit_diff = last_token_logits_answers - last_token_logits_wrongs

        last_token_logits_answers_interv = deepcopy(logits_interv[prompts_array, dataset.word_idx["end"], dataset.answers])
        last_token_logits_wrongs_interv = deepcopy(logits_interv[prompts_array, dataset.word_idx["end"], dataset.wrongs])
        logit_diff_interv = last_token_logits_answers_interv - last_token_logits_wrongs_interv
    elif isinstance(dataset, YearDataset):
        logit_diff = []
        logit_diff_interv = []
        for prompt_id in range(len(dataset)):
            YY_idx = dataset.possible_targets_toks.index(dataset.YY_toks[prompt_id])
            correct_tokens_idx = list(range(0, YY_idx+1))
            incorrect_tokens_idx = list(range(YY_idx+1, len(dataset.possible_targets_toks)))

            assert len(set(correct_tokens_idx).intersection(incorrect_tokens_idx)) == 0
            assert np.all(np.array(list(set(correct_tokens_idx).union(incorrect_tokens_idx))) == np.array(range(100)))
            assert YY_idx in correct_tokens_idx

            correct_tokens = [dataset.possible_targets_toks[i] for i in correct_tokens_idx]
            incorrect_tokens = [dataset.possible_targets_toks[i] for i in incorrect_tokens_idx]

            logit_diff.append(
                logits[prompt_id, dataset.word_idx["end"][prompt_id], correct_tokens].mean() - \
                    logits[prompt_id, dataset.word_idx["end"][prompt_id], incorrect_tokens].mean()
            )
            logit_diff_interv.append(
                logits_interv[prompt_id, dataset.word_idx["end"][prompt_id], correct_tokens].mean() - \
                    logits_interv[prompt_id, dataset.word_idx["end"][prompt_id], incorrect_tokens].mean()
            )

        logit_diff = torch.tensor(logit_diff, device=device)
        logit_diff_interv = torch.tensor(logit_diff_interv, device=device)
    else:
        raise Exception(f"Error when computing the logit difference. Dataset type is: {type(dataset)}")

    return logit_diff, logit_diff_interv

def run_intervention(model: HookedTransformer,
                     model_name: str,
                     config: ModelConfig,
                     cache: ActivationCache,
                     logits: Float[Tensor, "n_prompts n_tokens d_vocab"],
                     dataset: IOIDataset | GenderedPronoun | YearDataset,
                     U: dict,
                     VT: dict,
                     MODEL_CONSTANTS: dict,
                     edge_ablation: dict,
                     svs_used: dict,
                     intervention_type: str,
                     is_boosting: bool,
                     is_random: bool,
                     random: np.random.RandomState,
                     device: str = "cpu") -> Tuple[Float[Tensor, "n_prompts"],
                                                            Float[Tensor, "n_prompts"],
                                                            Float[Tensor, "n_prompts"],
                                                            Float[Tensor, "n_prompts"],
                                                            list,
                                                            Float[Tensor, "n_prompts"],
                                                            Float[Tensor, "n_prompts"]]:

    edge = (str(edge_ablation["upstream_node"]), str(edge_ablation["downstream_node"]))

    if intervention_type not in ["global", "local"]:
        raise Exception("ablation_type must be either global or local")

    if edge_ablation["type"] not in ["d", "s"]:
        raise Exception("edge_ablation['type'] must be either d or s")

    # Getting the layer and head indexes (src and dest nodes)
    layer_upstream, ah_idx_upstream, dest_token_upstream, src_token_upstream = edge_ablation["upstream_node"]
    layer_downstream, ah_idx_downstream, dest_token_downstream, src_token_downstream = edge_ablation["downstream_node"]

    # Mapping back these upstream components to ids
    if ah_idx_upstream == "MLP":
        ah_idx_upstream = model.cfg.n_heads
    elif ah_idx_upstream == "AH bias":
        ah_idx_upstream = model.cfg.n_heads+1
    elif ah_idx_upstream == "Embedding":
        ah_idx_upstream = model.cfg.n_heads+2
    elif ah_idx_upstream == "AH offset":
        ah_idx_upstream = model.cfg.n_heads+3

    # Getting the src and dest tokens positions from the downstream tokens
    dest_token_pos_list = dataset.word_idx[dest_token_downstream] # Long[Tensor, 'n_prompts']
    src_token_pos_list = dataset.word_idx[src_token_downstream] # Long[Tensor, 'n_prompts']

    assert len(dest_token_pos_list) == len(src_token_pos_list) == len(dataset)

    # We intervene in the dest_token if the edge is of type "d", otherwise we intervene in the src_token
    interv_pos_list = deepcopy(dest_token_pos_list) if edge_ablation["type"] == "d" else deepcopy(src_token_pos_list)

    assert len(interv_pos_list) == len(dataset)

    # Getting the intervention type
    projection_type = "U" if edge_ablation["type"] == "d" else "V"

    # Initializing the delta intervention tensor
    # This tensor will always have the same shape as the attn_out tensor (n_prompts, n_tokens, d_model)
    delta_interv_tensor = torch.zeros(cache[f"blocks.{layer_upstream}.hook_attn_out"].shape, device=device)

    # List of prompts that can be ablated (with SVs)
    prompts_ablation = []
    for prompt_idx in range(len(dataset)):
        # We trace only prompts that this edge appeared
        if not prompt_idx in edge_ablation["prompts_appeared"]:
            continue

        # Getting the src and dest tokens for the prompt in this edge
        dest_token = dest_token_pos_list[prompt_idx].item()
        src_token = src_token_pos_list[prompt_idx].item()
        diff = dest_token - src_token if config.has_rope else -1

        # Getting the position to intervene for the prompt in this edge
        pos_interv = interv_pos_list[prompt_idx].item()

        # Step 1: getting the SVs
        try:
            svs_used_prompt = svs_used[prompt_idx]
        except KeyError:
            svs_used_prompt = [] # No SVs to use

        # Step 2: computing the projections
        if len(svs_used_prompt) == 0:
            # No ablation possible
            continue
        else:
            prompts_ablation.append(prompt_idx)
            if edge_ablation["type"] == "d":
                P_u, P_u_rand = compute_projections(U[layer_downstream, ah_idx_downstream],
                                                    svs_used_prompt,
                                                    random)
            else:
                P_v, P_v_rand = compute_projections(VT[layer_downstream, ah_idx_downstream].T,
                                                    svs_used_prompt,
                                                    random)

        # Step 3: computing the delta used to intervene
        dest_token_upstream_idx = dataset.word_idx[dest_token_upstream][prompt_idx]
        src_token_upstream_idx = dataset.word_idx[src_token_upstream][prompt_idx]
        if ah_idx_upstream < model.cfg.n_heads:
            n_tokens = dataset.word_idx["end"][prompt_idx]+1
            # This only works for AH
            # Breaking down the upstream_output using the OV circuit linearity
            # For AH, both dest_token_upstream and src_token_upstream matter
            A = deepcopy(cache[f"blocks.{layer_upstream}.attn.hook_pattern"][prompt_idx, ah_idx_upstream, :n_tokens, :n_tokens]) # n_tokens x n_tokens
            if model_name == "gemma-2-2b":
                # Gemma-2-2b uses group query attention. Then, ah_idx 0 and 1 have the same hook_v (idx=0).
                V = deepcopy(cache[f"blocks.{layer_upstream}.attn.hook_v"][prompt_idx, :n_tokens, ah_idx_upstream//2, :]) # n_tokens x d_head
            else:
                V = deepcopy(cache[f"blocks.{layer_upstream}.attn.hook_v"][prompt_idx, :n_tokens, ah_idx_upstream, :]) # n_tokens x d_head
            upstream_out = torch.einsum('ti,ij->tij', A, V) @ model.W_O[layer_upstream, ah_idx_upstream, :, :] # n_tokens x n_tokens x d_model
            upstream_out = upstream_out[dest_token_upstream_idx, src_token_upstream_idx, :] # (d_model)

            if model_name == "gemma-2-2b":
                # Post attention layer norm term (not folded)
                upstream_out *= model.blocks[layer_upstream].ln1_post.w.detach()
                ln_post_term = cache[f"blocks.{layer_upstream}.ln1_post.hook_scale"][prompt_idx, dest_token_upstream_idx]
                upstream_out /= ln_post_term

        elif ah_idx_upstream == model.cfg.n_heads: #MLP
            # For MLP, src_token_upstream do not matter
            # (this concept do not exist, so src_token_upstream == dest_token_upstream)
            upstream_out = deepcopy(cache[f"blocks.{layer_upstream}.hook_mlp_out"][prompt_idx, dest_token_upstream_idx]) # (d_model)
        elif ah_idx_upstream == model.cfg.n_heads+1: #Attn bias
            # For attn bias, dest_token_upstream and src_token_upstream DO NOT matter
            upstream_out = deepcopy(model.b_O[layer_upstream]) # (d_model)
        elif ah_idx_upstream == model.cfg.n_heads+2: #Embedding
            # For embedding, src_token_upstream do not matter
            # (this concept do not exist, so src_token_upstream == dest_token_upstream)
            upstream_out = deepcopy(cache["blocks.0.hook_resid_pre"][prompt_idx, dest_token_upstream_idx]) # (d_model)
        elif ah_idx_upstream == model.cfg.n_heads+3: # AH offset
            # In the offset, the data is from the downstream AH
            if config.has_rope:
                R = torch.stack([get_rotation_matrix(model, i, device) for i in range(dest_token+1)])
                M_d = model.W_Q[layer_downstream, ah_idx_downstream] @ R[dest_token].T @ MODEL_CONSTANTS["W_Q_pinv"][layer_downstream, ah_idx_downstream]
                M_s_all = einsum(
                    MODEL_CONSTANTS["W_K_pinv"][layer_downstream, ah_idx_downstream].T,
                    R,
                    model.W_K[layer_downstream, ah_idx_downstream].T,
                    "d_model d_head_a, n_tokens d_head_a d_head_b, d_head_b d_model_out -> n_tokens d_model d_model_out"
                )
                M_s = M_s_all[src_token]
            else:
                # Rotation matrix as the identity for models with no RoPE
                M_d = torch.eye(model.cfg.d_model, device=device)
                M_s = torch.eye(model.cfg.d_model, device=device)

            if edge_ablation["type"] == "d":
                upstream_out = MODEL_CONSTANTS["c_d"][layer_downstream, ah_idx_downstream] @ M_d
            else:
                upstream_out = M_s @ MODEL_CONSTANTS["c_s"][layer_downstream, ah_idx_downstream]

        # Step 3.1) computing the intervention value
        if is_random: # Random projection intervention
            P = P_u_rand if projection_type == "U" else P_v_rand
        else:
            P = P_u if projection_type == "U" else P_v

        # Computing the intervention (signal) by computing the upstream_out @ P
        signal = upstream_out @ P
        assert signal.shape == (model.cfg.d_model,)

        # # Step 3.2) centering the intervention value (only in the position pos_interv)
        # # Gemma doesn't center residuals like GPT-2 and Pythia
        if model_name != "gemma-2-2b":
            signal -= signal.mean()
            assert torch.allclose(signal.mean(), torch.zeros(1, device=device), atol=1e-5) # Due to the precision that we use

        if intervention_type == "local":
            # Step 3.3) scaling the intervention value (only for intervention type B)
            scaling_pos_interv = cache[f"blocks.{layer_downstream}.ln1.hook_scale"][prompt_idx, pos_interv] # Float[Tensor, 1]
            signal /= scaling_pos_interv
            assert pos_interv == dest_token_upstream_idx

        # Step 3.4) putting the intervention value in the delta tensor (only in the position pos_interv)
        delta_interv_tensor[prompt_idx, pos_interv, :] = signal

        # 1) delta_interv_tensor can be non-zero only in the position pos_interv
        assert torch.allclose(delta_interv_tensor[prompt_idx, :pos_interv, :], torch.zeros((pos_interv, delta_interv_tensor.shape[2]), device=device))
        assert torch.allclose(delta_interv_tensor[prompt_idx, pos_interv+1:, :], torch.zeros((delta_interv_tensor.shape[1] - pos_interv - 1, delta_interv_tensor.shape[2]), device=device))
        assert not torch.allclose(delta_interv_tensor[prompt_idx, pos_interv, :], torch.zeros(delta_interv_tensor.shape[2], device=device))

    # Step 4: intervening in the model
    if intervention_type == "global":
        # Upstream node
        interv_logits, interv_cache = run_global_intervention(model, dataset, layer_upstream, ah_idx_upstream, delta_interv_tensor, intervention_type, is_boosting)
    elif intervention_type == "local":
        # Downstream node
        interv_logits, interv_cache, after_interv = run_local_intervention(model, model_name, cache, dataset, layer_downstream, ah_idx_downstream, delta_interv_tensor, intervention_type, is_boosting, projection_type)

    # Step 5: computing the ablation effect
    logit_diff, logit_diff_interv = compute_logit_diff(dataset, logits, interv_logits, device)

    # Computing the values before and after the intervention
    if intervention_type == "global":
        # AH and Attn bias cases: they all change the attn block output
        if ah_idx_upstream < model.cfg.n_heads or ah_idx_upstream == model.cfg.n_heads+1:
            before_interv = cache[f"blocks.{layer_upstream}.hook_attn_out"] # Output of the upstream AH.
            after_interv = interv_cache[f"blocks.{layer_upstream}.hook_attn_out"]
        elif ah_idx_upstream == model.cfg.n_heads: #MLP
            before_interv = cache[f"blocks.{layer_upstream}.hook_mlp_out"] # Output of the upstream MLP.
            after_interv = interv_cache[f"blocks.{layer_upstream}.hook_attn_out"]
        elif ah_idx_upstream == model.cfg.n_heads+2: #Embedding
            before_interv = cache["blocks.0.hook_resid_pre"]
            after_interv = interv_cache["blocks.0.hook_resid_pre"]
    elif intervention_type == "local":
        # We always change the input of a downstream AH
        before_interv = cache[f"blocks.{layer_downstream}.ln1.hook_normalized"] # Input of the downstream AH.

    # Step 6) computing how much we are changing in the intervention
    # Getting only the position that we intervened for each prompt.
    after_interv_interv_pos = after_interv[range(len(dataset)), interv_pos_list, :] # Float[Tensor, 'n_prompts d_model']
    before_interv_interv_pos = before_interv[range(len(dataset)), interv_pos_list, :] # Float[Tensor, 'n_prompts d_model']

    norm_ratio = torch.norm(after_interv_interv_pos, dim=1) / torch.norm(before_interv_interv_pos, dim=1)

    # Step 6.1) compare how much we are changing using cosine similarity
    cos_sim = F.cosine_similarity(after_interv_interv_pos, before_interv_interv_pos, dim=1)

    # Step 7) computing the interv effect in the downstream head probability
    # The intervention effect is measured on the dest_token and src_token
    attn_scores_matrix = cache[f"blocks.{layer_downstream}.attn.hook_pattern"][:, ah_idx_downstream, :, :]
    scores_dest_src_downstream_ah = attn_scores_matrix[range(len(dataset)), dest_token_pos_list, src_token_pos_list]

    attn_scores_matrix_interv = interv_cache[f"blocks.{layer_downstream}.attn.hook_pattern"][:, ah_idx_downstream, :, :]
    scores_dest_src_downstream_ah_interv = attn_scores_matrix_interv[range(len(dataset)), dest_token_pos_list, src_token_pos_list]

    return logit_diff.cpu().numpy(), logit_diff_interv.cpu().numpy(), norm_ratio.cpu().numpy(), cos_sim.cpu().numpy(), prompts_ablation, scores_dest_src_downstream_ah.cpu().numpy(), scores_dest_src_downstream_ah_interv.cpu().numpy()

# ---------------------------------------------------------------------------
# Curated edges experiment (Step 3a)
# ---------------------------------------------------------------------------

def run_interventions_experiments(args):
    model_name = args.model_name
    model_family = args.model_name.split("/")[-1].split("-")[0] # For models of the form COMPANY/MODELNAME-PARAMS
    num_prompts = args.num_prompts
    seed = args.seed
    task = args.task
    device = args.device

    # Loading model
    model = HookedTransformer.from_pretrained(model_name, device=device)

    if task == "ioi":
        dataset = IOIDataset(
            model_family=model_family,
            prompt_type="mixed",
            N=num_prompts,
            tokenizer=model.tokenizer,
            prepend_bos=True if model_family == "gemma" else False,
            seed=seed,
            device=device
        )
        logits, cache = model.run_with_cache(dataset.toks)
    elif task == "gp":
        dataset = GenderedPronoun(model, model_family=model_family, device=device, prepend_bos=True if model_family == "gemma" else False)
        if num_prompts != 100:
            raise Exception("Currently, GenderedPronoun generates only 100 examples. Please set num_prompts to 100.")
        logits, cache = model.run_with_cache(dataset.toks)
    elif task == "gt":
        years_to_sample_from = get_valid_years(model.tokenizer, 1000, 1900)
        dataset = YearDataset(years_to_sample_from,
                              args.num_prompts,
                              model.tokenizer,
                              balanced=True,
                              device=args.device,
                              eos=True, # Following the original paper
                              random_seed=args.seed)
        logits, cache = model.run_with_cache(dataset.toks)

    # Omega SVD
    config = get_model_config(model, use_numpy_svd=("pythia" in model_family))
    U, _, VT = get_omega_decomposition(model, config, device)

    random = np.random.RandomState(seed) # Random state for reproducibility

    # Model constants for AH offset interventions
    W_Q_pinv, W_K_pinv = compute_weight_pseudoinverses(model, config, device)
    c_d = einsum(model.b_Q, W_Q_pinv, "n_layers n_heads d_head, n_layers n_heads d_head d_model -> n_layers n_heads d_model")
    c_s = einsum(model.b_K, W_K_pinv, "n_layers n_heads d_head, n_layers n_heads d_head d_model -> n_layers n_heads d_model")
    MODEL_CONSTANTS = {
        "W_Q_pinv": W_Q_pinv,
        "W_K_pinv": W_K_pinv,
        "c_d": c_d,
        "c_s": c_s
    }

    # Curated edges for local interventions (paper Fig. 19).
    EDGES_ABLATION = {
        "ioi": {
            "gpt2-small": [
                ("(7, 9, 'end', 'S2')", "(9, 9, 'end', 'IO')", 0), # From S-Inhibition Head to Name Mover Head
                ("(5, 5, 'S2', 'S1+1')", "(7, 9, 'end', 'S2')", 0), # From Induction Head to S-Inhibition Head
                ("(2, 2, 'S1+1', 'S1')", "(5, 5, 'S2', 'S1+1')", 0) # From Previous Token Head to Induction Head
            ],
            "pythia-160m":[
                ("(6, 6, 'end', 'S2')", "(8, 10, 'end', 'IO')", 0), # From S-Inhibition Head to Name Mover Head
                ("(4, 11, 'S2', 'S1+1')", "(6, 6, 'end', 'S2')", 0), # From Induction Head to S-Inhibition Head
                ("(3, 2, 'S1+1', 'S1')", "(4, 11, 'S2', 'S1+1')", 0) # From Previous Token Head to Induction Head
            ],
            "gemma-2-2b": [
                # Candidates: From S-Inhibition Head to Name Mover Head
                ("(14, 3, 'end', 'S2')", "(22, 4, 'end', 'IO')", 0),

                # Candidates: From Induction Head to S-Inhibition Head
                ("(1, 4, 'S2', 'S1')", "(23, 5, 'end', 'S2')", 0),

                # Candidates: From Previous Token Head to Induction Head
                ("(3, 0, 'S1+1', 'S1')", "(4, 4, 'S2', 'S1+1')", 0),
            ]
        },
    }

    EDGES_NAMING_MAPPING = {
        "ioi": {
            # GPT-2 Small
            ("(7, 9, 'end', 'S2')", "(9, 9, 'end', 'IO')", 0): "S-Inhibition Head -> Name Mover Head",
            ("(5, 5, 'S2', 'S1+1')", "(7, 9, 'end', 'S2')", 0): "Induction Head -> S-Inhibition Head",
            ("(2, 2, 'S1+1', 'S1')", "(5, 5, 'S2', 'S1+1')", 0): "Previous Token Head -> Induction Head",

            # Pythia-160M
            ("(6, 6, 'end', 'S2')", "(8, 10, 'end', 'IO')", 0): "S-Inhibition Head -> Name Mover Head",
            ("(4, 11, 'S2', 'S1+1')", "(6, 6, 'end', 'S2')", 0): "Induction Head -> S-Inhibition Head",
            ("(3, 2, 'S1+1', 'S1')", "(4, 11, 'S2', 'S1+1')", 0): "Previous Token Head -> Induction Head",

            # Gemma-2 2B
            ("(14, 3, 'end', 'S2')", "(22, 4, 'end', 'IO')", 0): "S-Inhibition Head -> Name Mover Head",
            ("(1, 4, 'S2', 'S1')", "(23, 5, 'end', 'S2')", 0): "Induction Head -> S-Inhibition Head",
            ("(3, 0, 'S1+1', 'S1')", "(4, 4, 'S2', 'S1+1')", 0): "Previous Token Head -> Induction Head",
        }
    }

    # Load the graphs
    GRAPS = {}
    graph_files = glob.glob(f"data/traced_graphs/{model_name.split('/')[-1]}/{task}/*")
    for file in graph_files:
        prompt_id = eval(file.split("/")[-1].split("_")[3])
        GRAPS[prompt_id] = nx.read_graphml(file, force_multigraph=True)

    logit_diff_ablations = pd.DataFrame()

    # Running the intervention
    for idx, edge in enumerate(tqdm(EDGES_ABLATION[task][model_name.split("/")[-1]])):
        edge_ablation = {}
        svs_used_dict = {}
        for prompt_id in GRAPS:
            if edge in GRAPS[prompt_id].edges:
                # Getting the SVs of the edge per prompt
                svs_used = eval(GRAPS[prompt_id].edges[edge]["svs_used"])
                svs_used_dict[prompt_id] = svs_used

                # Update edge dict
                if len(edge_ablation) == 0:
                    edge_ablation = GRAPS[prompt_id].edges[edge]
                    edge_ablation["upstream_node"] = eval(edge[0])
                    edge_ablation["downstream_node"] = eval(edge[1])
                    edge_ablation["prompts_appeared"] = [prompt_id]
                else:
                    edge_ablation["prompts_appeared"].append(prompt_id)
        edge_ablation["prompts_appeared"].sort()

        for is_boosting in [False, True]:
            for is_random in [False, True]:
                logit_diff, logit_diff_interv, norm_ratio, cos_sim, prompts_ablation, scores_dest_src_downstream_ah, scores_dest_src_downstream_ah_interv = run_intervention(model,
                                model_name,
                                config,
                                cache,
                                logits,
                                dataset,
                                U,
                                VT,
                                MODEL_CONSTANTS,
                                edge_ablation,
                                svs_used_dict,
                                "local",
                                is_boosting,
                                is_random,
                                random,
                                device=device)


                assert np.all(np.array(edge_ablation["prompts_appeared"]) == np.array(prompts_ablation))

                prompts_ablation_mask = [True if i in prompts_ablation else False for i in range(len(dataset))]

                intervention_type_name = "Local (Random)" if is_random else "Local (SVs)"

                # Saving the logit difference
                logit_diff_ablations = pd.concat([logit_diff_ablations, pd.DataFrame({
                    "model_name": len(dataset) * [model_name.split("/")[-1]],
                    "prompt_id": range(len(dataset)),
                    "edge": len(dataset) * [str(edge)],
                    "logit_diff": logit_diff,
                    "logit_diff_interv": logit_diff_interv,
                    "norm_ratio": norm_ratio,
                    "cosine_similarity": cos_sim,
                    "intervention_type": len(dataset) * ["local"],
                    "intervention_type_name": len(dataset) * [intervention_type_name],
                    "is_boosting": len(dataset) * [is_boosting],
                    "is_random": len(dataset) * [is_random],
                    "edge_weight": len(dataset) * [edge_ablation["weight"]],
                    "scores_dest_src_downstream_ah": scores_dest_src_downstream_ah,
                    "scores_dest_src_downstream_ah_interv": scores_dest_src_downstream_ah_interv,
                    "is_ablated": prompts_ablation_mask,
                    "scores_dest_src_diff_metric": scores_dest_src_downstream_ah_interv - scores_dest_src_downstream_ah,
                    "logit_diff_metric": (logit_diff_interv - logit_diff) / logit_diff
                })]).reset_index(drop=True)

    logit_diff_ablations["operation_performed"] = logit_diff_ablations.apply(lambda x: "Boosting (Random)" if x["is_boosting"] and x["is_random"] else "Boosting (SVs)" if x["is_boosting"] and not x["is_random"] else "Removing (Random)" if not x["is_boosting"] and x["is_random"] else "Removing (SVs)", axis=1)
    logit_diff_ablations["edge_labeled"] = logit_diff_ablations["edge"].apply(lambda x: f"{eval(x)[0]} -> {eval(x)[1]}")
    logit_diff_ablations["edge_labeled_group"] = logit_diff_ablations["edge"].apply(lambda x: EDGES_NAMING_MAPPING[task][eval(x)])

    folder = "data/intervention_data"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{model_name.split('/')[-1]}_{task}.parquet"
    logit_diff_ablations.to_parquet(filename)
    print(f"Saved: {filename}")

# ---------------------------------------------------------------------------
# All edges experiment (Step 3b) — runs with -ig flag
# ---------------------------------------------------------------------------

def interventions_graph(args):
    model_name = args.model_name
    model_family = args.model_name.split("/")[-1].split("-")[0] # For models of the form COMPANY/MODELNAME-PARAMS
    num_prompts = args.num_prompts
    seed = args.seed
    task = args.task
    device = args.device

    # Loading model
    model = HookedTransformer.from_pretrained(model_name, device=device)

    if task == "ioi":
        dataset = IOIDataset(
            model_family=model_family,
            prompt_type="mixed",
            N=num_prompts,
            tokenizer=model.tokenizer,
            prepend_bos=True if model_family == "gemma" else False,
            seed=seed,
            device=device
        )
        logits, cache = model.run_with_cache(dataset.toks)
        ROOT_NODE = "('IO-S direction', 'end')"
    elif task == "gp":
        dataset = GenderedPronoun(model, model_family=model_family, device=device, prepend_bos=True if model_family == "gemma" else False)
        if num_prompts != 100:
            raise Exception("Currently, GenderedPronoun generates only 100 examples. Please set num_prompts to 100.")
        logits, cache = model.run_with_cache(dataset.toks)
        ROOT_NODE = "('Correct - Incorrect pronoum', 'end')"
    elif task == "gt":
        years_to_sample_from = get_valid_years(model.tokenizer, 1000, 1900)
        dataset = YearDataset(years_to_sample_from,
                              args.num_prompts,
                              model.tokenizer,
                              balanced=True,
                              device=args.device,
                              eos=True, # Following the original paper
                              random_seed=args.seed)
        logits, cache = model.run_with_cache(dataset.toks)
        ROOT_NODE = "('True YY - False YY', 'end')"

    config = get_model_config(model, use_numpy_svd=("pythia" in model_family))
    U, _, VT = get_omega_decomposition(model, config, device)

    random = np.random.RandomState(seed) # Random state for reproducibility

    W_Q_pinv, W_K_pinv = compute_weight_pseudoinverses(model, config, device)

    c_d = einsum(model.b_Q, W_Q_pinv, "n_layers n_heads d_head, n_layers n_heads d_head d_model -> n_layers n_heads d_model")
    c_s = einsum(model.b_K, W_K_pinv, "n_layers n_heads d_head, n_layers n_heads d_head d_model -> n_layers n_heads d_model")

    MODEL_CONSTANTS = {
        "W_Q_pinv": W_Q_pinv,
        "W_K_pinv": W_K_pinv,
        "c_d": c_d,
        "c_s": c_s
    }

    threshold = 0.01
    GRAPHS_AGG = load_graphs(model_name.split("/")[-1], task)
    G: nx.MultiDiGraph = GRAPHS_AGG[f"{threshold}"]
    for edge in G.edges:
        G.edges[edge]["prompts_appeared"] = eval(G.edges[edge]["prompts_appeared"])

    # Load the individual graphs
    GRAPS = {}
    graph_files = glob.glob(f"data/traced_graphs/{model_name.split('/')[-1]}/{task}/*")
    for file in graph_files:
        prompt_id = eval(file.split("/")[-1].split("_")[3])
        GRAPS[prompt_id] = nx.read_graphml(file, force_multigraph=True)


    for idx, edge in enumerate(tqdm(G.edges(keys=True))):
        # No ablations for edges to the root node
        if ROOT_NODE in edge:
            continue

        edge_ablation = G.edges[edge]
        edge_ablation["upstream_node"] = eval(edge[0])
        edge_ablation["downstream_node"] = eval(edge[1])

        # Getting the SVS used
        svs_used_dict = {}
        for prompt_id in edge_ablation["prompts_appeared"]:
            # Getting the SVs of the edge per prompt
            # Finding the correct key for this edge in the prompt graph
            if G.has_edge(edge[0], edge[1]):
                # 3. Iterate over all parallel edges in G
                # G[u][v] returns a dict-like view: {key_G: data_dict, ...}
                for key_G, data in GRAPS[prompt_id][edge[0]][edge[1]].items():
                    if data.get('type') == edge_ablation["type"]:
                        # Found the matching edge in G
                        edge_prompt = (edge[0], edge[1], key_G)
                        break # Stop searching
            else:
                raise Exception(f"Edge {edge} do not exist in prompt {prompt_id} while it should.")

            svs_used = eval(GRAPS[prompt_id].edges[edge_prompt]["svs_used"])
            svs_used_dict[prompt_id] = svs_used

        edge_ablation["prompts_appeared"].sort()

        del edge_ablation["svs_used"]

        logit_diff, logit_diff_interv, norm_ratio, cos_sim, prompts_ablation, scores_dest_src_downstream_ah, scores_dest_src_downstream_ah_interv = run_intervention(model,
                model_name,
                config,
                cache,
                logits,
                dataset,
                U,
                VT,
                MODEL_CONSTANTS,
                edge_ablation,
                svs_used_dict,
                "local",
                False,
                False,
                random,
                device=device)


        assert np.all(np.array(G.edges[edge]["prompts_appeared"]) == np.array(prompts_ablation))

        logit_diff = list(logit_diff_interv[prompts_ablation] - logit_diff[prompts_ablation])

        G.edges[edge]["logit_diff"] = str(logit_diff)

        edge_ablation["upstream_node"] = str(edge[0])
        edge_ablation["downstream_node"] = str(edge[1])

        if idx % 100 == 0:
            print("Saving the partial interventions graph...")

            # Putting prompts_appeared back as a str
            for edge in G.edges(keys=True):
                G.edges[edge]["prompts_appeared"] = str(G.edges[edge]["prompts_appeared"])
            nx.write_graphml(G, f"data/interventions_graph_{model_name.split('/')[-1]}_{task}_n{num_prompts}_{seed}_combined_{threshold}.graphml")

            # Putting prompts_appeared back as a list to continue the interventions
            for edge in G.edges(keys=True):
                G.edges[edge]["prompts_appeared"] = eval(G.edges[edge]["prompts_appeared"])

    # Putting prompts_appeared, upstream_node, and downstream_node back as a str
    for edge in G.edges(keys=True):
        G.edges[edge]["prompts_appeared"] = str(G.edges[edge]["prompts_appeared"])
        if not ROOT_NODE in edge:
            try:
                G.edges[edge]["upstream_node"] = str(G.edges[edge]["upstream_node"])
                G.edges[edge]["downstream_node"] = str(G.edges[edge]["downstream_node"])
            except:
                print(f"Creating upstream_node and downstream_node keys for {edge}...")
                G.edges[edge]["upstream_node"] = edge[0]
                G.edges[edge]["downstream_node"] = edge[1]

    nx.write_graphml(G, f"data/interventions_graph_{model_name.split('/')[-1]}_{task}_n{num_prompts}_{seed}_combined_{threshold}.graphml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal interventions on circuit edges (Step 3)"
    )
    parser.add_argument("-m", "--model_name", choices=["gpt2-small", "EleutherAI/pythia-160m", "gemma-2-2b"], required=True)
    parser.add_argument("-t", "--task", choices=["ioi", "gt", "gp"], required=True)
    parser.add_argument("-n", "--num_prompts", type=int, required=True)
    parser.add_argument("-d", "--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-ig", "--intervention_graph", action="store_true")
    args = parser.parse_args()

    if args.intervention_graph:
        interventions_graph(args)
    else:
        run_interventions_experiments(args)
