"""Stage 5c: Convert an annotated GraphML circuit graph to a Bokeh HTML viewer.

Zero-server (single HTML file) interactive explorer for ACC++ circuit graphs.

What it does:
- Preserves node x/y positions from the Cytoscape-layout GraphML
- Optionally inverts Y (Cytoscape convention)
- Click a node to list **all incoming edges** and their ``interpretation``
  in a side panel; those edges are highlighted in the plot
- (Optional) Shows the token list below the plot for x-position reference
- (Optional) Shows the top-40 text examples per edge in the click panel
  when ``--examples`` is provided (from ``annotate_graphs.py`` output)

Usage:
    python view_circuit.py input.graphml output.html \\\\
        --tokens tokens.json \\\\
        --examples gpt2-small_ioi-balanced_n3000_129_edge_examples.json \\\\
        --xaxis-labels tokens
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional

import networkx as nx
import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    Circle,
    ColumnDataSource,
    CustomJS,
    CustomJSHover,
    Div,
    FixedTicker,
    HoverTool,
    MultiLine,
    Range1d,
    TapTool,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE


# ---------------------------------------------------------------------------
# User-tweakable defaults
# ---------------------------------------------------------------------------
REVERSE_Y = True              # Cytoscape-like convention (more negative = higher)
DEFAULT_NODE_SIZE = 10
SIDE_PANEL_WIDTH = 430


# ---------------------------------------------------------------------------
# Token loading helpers
# ---------------------------------------------------------------------------

def _parse_token_text(text: str) -> Optional[List[str]]:
    """Best-effort token list parsing: JSON list → Python literal → newline-separated.

    Args:
        text: Raw file content.

    Returns:
        Parsed token list, or ``None`` if parsing fails.
    """
    s = text.lstrip("\ufeff").strip()
    if not s:
        return None

    # JSON list
    if s[:1] == "[" and s[-1:] == "]":
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return ["" if x is None else str(x) for x in obj]
        except Exception:
            pass

        # Python list repr
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return ["" if x is None else str(x) for x in obj]
        except Exception:
            pass

    # One token per line (preserve leading spaces)
    lines = text.splitlines()
    toks = [ln.rstrip("\r") for ln in lines if ln.rstrip("\r") != ""]
    return toks if toks else None


def load_tokens(tokens_path: Optional[str]) -> Optional[List[str]]:
    """Load tokens from an optional file.

    Args:
        tokens_path: Path to a tokens file (JSON list, Python list, or one
            token per line). Ignored if ``None``.

    Returns:
        Parsed token list, or ``None`` if not provided or unreadable.
    """
    if not tokens_path:
        return None
    p = Path(tokens_path)
    if not p.exists():
        print(f"[warn] tokens file not found: {tokens_path}", file=sys.stderr)
        return None
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception:
        raw = p.read_text(errors="ignore")
    toks = _parse_token_text(raw)
    if not toks:
        print(f"[warn] could not parse tokens from: {tokens_path}", file=sys.stderr)
    return toks


def load_examples(examples_path: Optional[str]) -> Optional[dict[str, list[str]]]:
    """Load the edge examples JSON file produced by ``annotate_graphs.py``.

    Args:
        examples_path: Path to a JSON file keyed by
            ``"{u}|||{v}|||{multigraph_key}"`` with values being lists of
            formatted text strings. Ignored if ``None``.

    Returns:
        Dict mapping edge key → list of text strings, or ``None`` if not
        provided or unreadable.
    """
    if not examples_path:
        return None
    p = Path(examples_path)
    if not p.exists():
        print(f"[warn] examples file not found: {examples_path}", file=sys.stderr)
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[warn] examples file is not a JSON object: {examples_path}", file=sys.stderr)
            return None
        return data
    except Exception as e:
        print(f"[warn] could not load examples from {examples_path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# X-axis relabeling helpers
# ---------------------------------------------------------------------------

def _visible_token(tok: str) -> str:
    """Make leading spaces visible (e.g. ``' Jack'`` → ``'␠Jack'``).

    Args:
        tok: Token string.

    Returns:
        Token string with leading spaces replaced by ``␠``.
    """
    if tok is None:
        return ""
    s = str(tok)
    n_lead = len(s) - len(s.lstrip(" "))
    if n_lead <= 0:
        return s
    return ("␠" * n_lead) + s.lstrip(" ")


def apply_xaxis_labels(p: figure, nodes_df: pd.DataFrame, tokens: Optional[List[str]], mode: str) -> None:
    """Relabel the x-axis ticks while keeping original x-coordinates.

    Args:
        p: Bokeh figure to modify.
        nodes_df: DataFrame of node positions (must have an ``"x"`` column).
        tokens: Optional token list for ``"tokens"`` mode.
        mode: One of:
            - ``"raw"``: keep the original numeric x-axis (no relabeling)
            - ``"index"``: label each unique x-column as 0..N-1
            - ``"tokens"``: label with token text (falls back to index if no tokens)
    """
    mode = (mode or "index").lower().strip()
    if mode == "raw":
        return

    xs = sorted({float(x) for x in nodes_df["x"].tolist()})
    eps = 1e-6
    x_cols: List[float] = []
    for x in xs:
        if not x_cols or abs(x - x_cols[-1]) > eps:
            x_cols.append(x)

    def _coerce_tick(v: float):
        return int(round(v)) if abs(v - round(v)) < 1e-6 else v

    def _tick_key(v) -> str:
        if isinstance(v, int):
            return str(v)
        return f"{float(v):g}"

    ticks = [_coerce_tick(v) for v in x_cols]
    p.xaxis.ticker = FixedTicker(ticks=ticks)

    if mode == "tokens" and tokens:
        labels: List[str] = []
        for i in range(len(ticks)):
            t = tokens[i] if i < len(tokens) else f"[{i}]"
            labels.append(_visible_token(t))
        p.xaxis.major_label_overrides = {_tick_key(ticks[i]): labels[i] for i in range(len(ticks))}
        p.xaxis.major_label_orientation = 0.95
        p.xaxis.major_label_text_font_size = "8pt"
        p.xaxis.axis_label = "Tokens (x-axis order)"
    else:
        p.xaxis.major_label_overrides = {_tick_key(ticks[i]): str(i) for i in range(len(ticks))}
        p.xaxis.axis_label = "Token index (x-axis order)"


def infer_tokens_from_graph(G: nx.Graph) -> Optional[List[str]]:
    """Best-effort token list inference from GraphML graph-level attributes.

    Args:
        G: NetworkX graph read from GraphML.

    Returns:
        Token list if found, ``None`` otherwise.
    """
    candidate_keys = [
        "tokens", "token_list", "prompt_tokens", "sentence_tokens", "input_tokens", "text_tokens"
    ]
    for k in candidate_keys:
        if k in G.graph:
            val = G.graph.get(k)
            if isinstance(val, list):
                return ["" if x is None else str(x) for x in val]
            if isinstance(val, str):
                toks = _parse_token_text(val)
                if toks:
                    return toks
    return None


def tokens_to_html(tokens: List[str]) -> str:
    """Render tokens as a horizontally-scrollable, index-labeled strip.

    Args:
        tokens: List of token strings.

    Returns:
        HTML string for the token strip.
    """
    items = []
    for i, tok in enumerate(tokens):
        esc = html.escape(tok)
        items.append(
            f'<span style="display:inline-block; margin: 2px 10px 2px 0;">'
            f'<span style="color:#666;"><b>{i:>2}</b>:</span> '
            f'<span style="white-space:pre; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">{esc}</span>'
            f"</span>"
        )
    joined = "".join(items)
    return f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
      <div style="margin: 0 0 6px 0; font-weight:600;">Tokens (x-axis order)</div>
      <div style="
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 10px 12px;
        background: #fafafa;
        overflow-x: auto;
        white-space: nowrap;
      ">{joined}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Edge interpretation helpers
# ---------------------------------------------------------------------------

def _clean_full_interpretation(text: str, short: str = "") -> str:
    """Strip the trailing ``[interpretation]: <short>`` section from full LLM output.

    Args:
        text: Raw LLM output text.
        short: The short interpretation label (to detect and strip the trailing line).

    Returns:
        Cleaned text, or empty string if nothing remains.
    """
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""
    m = re.search(r"\[interpretation\]\s*:\s*", s, flags=re.IGNORECASE)
    if m:
        s = s[: m.start()].strip()
    if short and s.strip() == str(short).strip():
        return ""
    return s


def _guess_full_interpretation(edge_attrs: dict, short: str) -> str:
    """Try to retrieve the long-form LLM interpretation text from edge attributes.

    Args:
        edge_attrs: Dict of edge attributes from the GraphML.
        short: The short interpretation label (for comparison / stripping).

    Returns:
        Full interpretation text, or empty string if not found.
    """
    if not isinstance(edge_attrs, dict):
        return ""
    priority_keys = [
        "full_response",
        "interpretation_full",
        "full_interpretation",
        "interpretation_long",
        "interpretation_detail",
        "interpretation_details",
        "interpretation_text",
        "interpretation_description",
        "interpretation_explanation",
        "llm_interpretation",
        "llm_output",
        "description",
        "explanation",
        "commentary",
        "analysis",
    ]
    for k in priority_keys:
        v = edge_attrs.get(k, None)
        if isinstance(v, str) and v.strip():
            return _clean_full_interpretation(v, short)

    for k, v in edge_attrs.items():
        if not isinstance(v, str) or not v.strip():
            continue
        kl = str(k).lower()
        if "interpret" in kl and kl != "interpretation":
            return _clean_full_interpretation(v, short)

    skip = {
        "interpretation", "type", "color", "svs_used", "weight",
        "line_width", "label", "id", "source", "target", "xs", "ys",
        "examples_json",
    }
    candidates = []
    for k, v in edge_attrs.items():
        if k in skip:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s or (short and s == str(short).strip()):
                continue
            candidates.append((len(s), k, s))
    if candidates:
        candidates.sort(reverse=True)
        best_len, _, best = candidates[0]
        if best_len >= max(80, len(str(short).strip()) + 40):
            return _clean_full_interpretation(best, short)
    return ""


# ---------------------------------------------------------------------------
# GraphML → Bokeh
# ---------------------------------------------------------------------------

def _to_float(v: object, default: float = 0.0) -> float:
    """Safely convert a GraphML attribute value to float.

    Args:
        v: Attribute value (may be None, str, or numeric).
        default: Fallback value on failure.

    Returns:
        Float value.
    """
    try:
        if v is None:
            return float(default)
        s = str(v).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def build_viewer(
    input_graphml: str,
    output_html: str,
    tokens_path: Optional[str] = None,
    xaxis_labels: str = "index",
    examples_path: Optional[str] = None,
) -> None:
    """Build a standalone Bokeh HTML viewer from an annotated GraphML file.

    Reads the GraphML (with ``force_multigraph=True`` to preserve parallel
    edges and their keys), builds Bokeh glyphs, and writes a zero-server
    HTML file.

    Args:
        input_graphml: Path to the Cytoscape-layout GraphML file (output of
            ``annotate_graphs.py``).
        output_html: Output path for the standalone HTML file.
        tokens_path: Optional path to a tokens file (JSON list, Python list,
            or one token per line). If ``None``, attempts to infer from graph
            metadata.
        xaxis_labels: X-axis labeling mode: ``"raw"`` (numeric), ``"index"``
            (0..N-1), or ``"tokens"`` (token text).
        examples_path: Optional path to the ``*_edge_examples.json`` file
            produced by ``annotate_graphs.py``. Keys are
            ``"{u}|||{v}|||{multigraph_key}"``; values are lists of formatted
            text strings. When provided, the click panel shows the top-K
            text examples below the interpretation.
    """
    # Use force_multigraph=True to preserve parallel edges and their integer keys
    G = nx.read_graphml(input_graphml, force_multigraph=True)

    tokens = load_tokens(tokens_path) if tokens_path else None
    if tokens is None:
        tokens = infer_tokens_from_graph(G)

    examples_dict = load_examples(examples_path)

    # --- Nodes ---
    nodes = []
    for n, d in G.nodes(data=True):
        x = _to_float(d.get("x", 0.0), 0.0)
        y = _to_float(d.get("y", 0.0), 0.0)
        if REVERSE_Y:
            y = -y

        nodes.append(
            {
                "id": str(n),
                "label": str(d.get("label", n)),
                "x": x,
                "y": y,
                "color": str(d.get("color", "#1f77b4")),
                "border_color": str(d.get("border_color", "#000000")),
            }
        )

    nodes_df = pd.DataFrame(nodes)
    if nodes_df.empty:
        raise ValueError("No nodes found in the GraphML.")

    node_pos = {row["id"]: (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    # --- Edges ---
    # Iterate with keys=True to get the multigraph key for examples lookup.
    edges = []
    for u, v, mk, d in G.edges(keys=True, data=True):
        su, sv = str(u), str(v)
        x0, y0 = node_pos.get(su, (0.0, 0.0))
        x1, y1 = node_pos.get(sv, (0.0, 0.0))

        w = _to_float(d.get("weight", 0.0), 0.0)
        line_width = 1.0 + min(6.0, math.sqrt(abs(w)) if abs(w) > 0 else 0.0)

        # Look up text examples for this edge using relabeled-node key
        examples_json = "[]"
        if examples_dict is not None:
            ek = f"{su}|||{sv}|||{str(mk)}"
            ex_list = examples_dict.get(ek, [])
            examples_json = json.dumps(ex_list)

        edges.append(
            {
                "source": su,
                "target": sv,
                "xs": [x0, x1],
                "ys": [y0, y1],
                "interpretation": str(d.get("interpretation", "")),
                "interpretation_full": _guess_full_interpretation(d, d.get("interpretation", "")),
                "type": str(d.get("type", "")),
                "weight": w,
                "svs_used": str(d.get("svs_used", "")),
                "color": str(d.get("color", "#999999")),
                "line_width": line_width,
                "examples_json": examples_json,
            }
        )

    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        raise ValueError("No edges found in the GraphML.")

    node_source = ColumnDataSource(nodes_df)
    edge_source = ColumnDataSource(edges_df)

    # --- Figure ---
    x_min, x_max = float(nodes_df["x"].min()), float(nodes_df["x"].max())
    y_min, y_max = float(nodes_df["y"].min()), float(nodes_df["y"].max())
    padx = (x_max - x_min) * 0.05 + 50
    pady = (y_max - y_min) * 0.05 + 50

    p = figure(
        title="GraphML explorer (click a node to read incoming edge interpretations)",
        width=1100,
        height=700,
        x_range=Range1d(x_min - padx, x_max + padx),
        y_range=Range1d(y_min - pady, y_max + pady),
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
    )
    p.grid.visible = False

    apply_xaxis_labels(p, nodes_df, tokens, xaxis_labels)

    # Edges
    edge_renderer = p.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="color",
        line_alpha=0.45,
        line_width="line_width",
    )
    edge_renderer.selection_glyph = MultiLine(line_color="color", line_width="line_width", line_alpha=0.95)
    edge_renderer.nonselection_glyph = MultiLine(line_color="color", line_width="line_width", line_alpha=0.06)

    # Nodes
    node_renderer = p.circle(
        x="x",
        y="y",
        source=node_source,
        size=DEFAULT_NODE_SIZE,
        fill_color="color",
        line_color="border_color",
        line_width=1.0,
        fill_alpha=0.95,
    )
    node_renderer.selection_glyph = Circle(fill_color="color", line_color="#000000", line_width=2.5, fill_alpha=1.0)
    node_renderer.nonselection_glyph = Circle(fill_color="color", line_color="border_color", line_width=1.0, fill_alpha=0.25)

    p.add_tools(TapTool(renderers=[node_renderer]))

    # Hover tools
    node_hover = HoverTool(
        renderers=[node_renderer],
        tooltips=[("id", "@id"), ("label", "@label"), ("x", "@x{0.0}"), ("y", "@y{0.0}")],
    )

    interp_hover = CustomJSHover(code="""
        const v = value == null ? "" : String(value);
        const maxlen = 160;
        if (v.length <= maxlen) return v;
        return v.slice(0, maxlen) + "\\u2026";
    """)
    edge_hover = HoverTool(
        renderers=[edge_renderer],
        line_policy="nearest",
        tooltips=[
            ("src", "@source"),
            ("tgt", "@target"),
            ("type", "@type"),
            ("w", "@weight{0.000}"),
            ("interp", "@interpretation"),
        ],
        formatters={"@interpretation": interp_hover},
    )
    p.add_tools(node_hover, edge_hover)

    # Side panel
    help_text = """
    <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35;">
      <h3 style="margin:0 0 8px 0;">Incoming edges</h3>
      <div style="color:#666; margin-bottom: 10px;">
        Click a node to list all <b>incoming</b> edges and read their <code>interpretation</code>.
      </div>
      <div style="color:#666;">
        Tip: use mousewheel zoom + pan; the selected node's incoming edges will highlight automatically.
      </div>
    </div>
    """
    panel = Div(
        text=f'<div style="height:700px; overflow-y:auto; padding-right:6px;">{help_text}</div>',
        width=SIDE_PANEL_WIDTH,
        height=700,
    )

    tokens_div = Div(text=tokens_to_html(tokens), width=1100 + SIDE_PANEL_WIDTH, height=130) if tokens else None

    # JS callback: node click → incoming interpretations + optional examples
    callback = CustomJS(
        args=dict(node_src=node_source, edge_src=edge_source, panel=panel, help_html=help_text),
        code=r"""
function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const sel = node_src.selected.indices;
const nd = node_src.data;
const ed = edge_src.data;

if (!sel || sel.length === 0) {
  panel.text = `<div style="height:700px; overflow-y:auto; padding-right:6px;">${help_html}</div>`;
  edge_src.selected.indices = [];
  edge_src.change.emit();
  return;
}

const idx = sel[0];
const nodeId = String(nd['id'][idx]);
const nodeLabel = String(nd['label'][idx]);
const x = nd['x'][idx];
const y = nd['y'][idx];

// gather incoming edges by target
let incoming = [];
for (let k = 0; k < ed['target'].length; k++) {
  if (String(ed['target'][k]) === nodeId) incoming.push(k);
}

// sort by weight desc
incoming.sort((a, b) => {
  const wa = Number(ed['weight'][a]);
  const wb = Number(ed['weight'][b]);
  if (Number.isFinite(wa) && Number.isFinite(wb)) return wb - wa;
  return 0;
});

// highlight those edges
edge_src.selected.indices = incoming;
edge_src.change.emit();

let body = `
  <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35;">
    <h3 style="margin:0 0 8px 0;">Incoming edges</h3>
    <div style="margin:0 0 10px 0;">
      <div><b>node:</b> ${escapeHtml(nodeLabel)} <span style="color:#666;">(${escapeHtml(nodeId)})</span></div>
      <div style="color:#666;"><b>x:</b> ${escapeHtml(x)} &nbsp; <b>y:</b> ${escapeHtml(y)}</div>
    </div>
`;

if (incoming.length === 0) {
  body += `<div style="color:#666;">No incoming edges.</div></div>`;
  panel.text = `<div style="height:700px; overflow-y:auto; padding-right:6px;">${body}</div>`;
  return;
}

for (const k of incoming) {
  const src = escapeHtml(ed['source'][k]);
  const typ = escapeHtml(ed['type'][k]);
  const w = escapeHtml(ed['weight'][k]);
  const svs = escapeHtml(ed['svs_used'][k]);
  const interp = escapeHtml(ed['interpretation'][k]);
  const full_interp = escapeHtml(ed['interpretation_full'][k]);

  let full_block = "";
  if (full_interp && full_interp.trim().length > 0) {
    full_block = `
      <details style="margin-top:10px; padding: 8px; border: 1px solid #f0f0f0; border-radius: 10px; background:#fafafa;">
        <summary style="cursor: pointer;"><b>Full interpretation</b></summary>
        <div style="margin-top:8px; white-space: pre-wrap; max-height: 240px; overflow-y: auto;">${full_interp}</div>
      </details>
    `;
  }

  // Text examples block (only if examples_json is non-empty)
  let examples_block = "";
  const examples_json_str = ed['examples_json'][k];
  if (examples_json_str && examples_json_str !== "[]") {
    try {
      const examples = JSON.parse(examples_json_str);
      if (examples.length > 0) {
        const items = examples.map(t =>
          `<div style="margin: 2px 0; white-space: pre-wrap; font-size: 0.9em;">${escapeHtml(t)}</div>`
        ).join("");
        examples_block = `
          <details style="margin-top:10px; padding: 8px; border: 1px solid #e6f3ff; border-radius: 10px; background:#f5f9ff;">
            <summary style="cursor: pointer;"><b>Top-${examples.length} text examples</b></summary>
            <div style="margin-top:8px; max-height: 300px; overflow-y: auto;
                        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                        font-size: 0.85em;">${items}</div>
          </details>
        `;
      }
    } catch(e) {
      // JSON parse failure — skip examples block silently
    }
  }

  body += `
    <details style="margin: 0 0 10px 0; padding: 8px; border: 1px solid #e6e6e6; border-radius: 10px; background:#fff;">
      <summary style="cursor: pointer;">
        <b>${src}</b> → <b>${escapeHtml(nodeId)}</b>
        <span style="color:#666;">&nbsp;&nbsp;type:</span> ${typ}
        <span style="color:#666;">&nbsp;&nbsp;w:</span> ${w}
      </summary>
      <div style="margin-top:8px; white-space: pre-wrap;"><b>interpretation:</b>\n${interp}</div>
      <div style="margin-top:6px; color:#666;"><b>svs_used:</b> ${svs}</div>
      ${full_block}
      ${examples_block}
    </details>
  `;
}

body += `</div>`;
panel.text = `<div style="height:700px; overflow-y:auto; padding-right:6px;">${body}</div>`;
"""
    )

    node_source.selected.js_on_change("indices", callback)

    layout = column(row(p, panel), tokens_div) if tokens_div is not None else row(p, panel)
    html_out = file_html(layout, INLINE, "GraphML Explorer")
    Path(output_html).write_text(html_out, encoding="utf-8")
    print(f"Wrote: {output_html}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the circuit viewer CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Exit code (0 on success).
    """
    ap = argparse.ArgumentParser(
        add_help=True,
        description="Convert an annotated Cytoscape-layout GraphML to a standalone Bokeh HTML viewer.",
    )
    ap.add_argument("input_graphml", help="Path to .graphml file.")
    ap.add_argument("output_html", help="Path to output .html file.")
    ap.add_argument(
        "--tokens",
        dest="tokens_path",
        default=None,
        help=(
            "Optional tokens file (JSON list, Python list, or one token per line). "
            "If omitted, tries to infer from graph metadata."
        ),
    )
    ap.add_argument(
        "--xaxis-labels",
        dest="xaxis_labels",
        choices=["raw", "index", "tokens"],
        default="index",
        help=(
            "How to label the x-axis while keeping original x spacing: "
            "raw (numeric), index (0..N-1), tokens (token text)."
        ),
    )
    ap.add_argument(
        "--examples",
        dest="examples_path",
        default=None,
        help=(
            "Optional path to *_edge_examples.json produced by annotate_graphs.py. "
            "When provided, the click panel shows top-K text examples below each "
            "edge's interpretation."
        ),
    )

    args = ap.parse_args(argv)
    build_viewer(
        args.input_graphml,
        args.output_html,
        tokens_path=args.tokens_path,
        xaxis_labels=args.xaxis_labels,
        examples_path=args.examples_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
