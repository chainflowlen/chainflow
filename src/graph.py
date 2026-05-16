"""
src/graph.py — Transfer network graph visualisation (exchange-merged view).

Key design decisions
────────────────────
• Exchange merging: ALL addresses belonging to the same exchange (e.g., every
  Binance wallet) are collapsed into ONE node labelled with the exchange name.
  This dramatically reduces clutter and surfaces inter-entity flows clearly.

• Node cap: one node per exchange appearing in the data + top-N anonymous whale
  addresses ranked by total transfer volume (default N = 35).

• Edge filter: only transfers where at least one endpoint is a known exchange
  (after merging) are shown. Whale-to-whale flows are included only when both
  whales connect through an exchange.

• Layout: exchange nodes are pinned evenly on an inner circle; whale nodes are
  positioned by a spring layout that clusters them around their exchange(s).

• Edge colour encodes flow direction at a glance:
    teal   = inflow   (whale → exchange)
    red    = outflow  (exchange → whale)
    purple = inter-exchange flow
    gray   = whale-to-whale

• Node size  ∝  log(total transfer volume).
• Edge width ∝  log(aggregated USD amount on that arc).

• A "Top 8 Flows" annotation box lists the largest individual directed flows.

Usage (standalone):
    python visualize_graph.py

Programmatic:
    from src.graph import plot_both_graphs
    usdc_png, usdt_png = plot_both_graphs(
        transfers_csv="data/2026-05-16/transfers.csv",
        labels_csv="data/labels.csv",
        output_dir="output/2026-05-16",
        threshold=1_000_000,
        max_anon_nodes=35,
        date_tag="2026-05-16",
    )
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering – no display required
import matplotlib.pyplot as plt
import networkx as nx


# ── visual constants ──────────────────────────────────────────────────────────
_BG             = "#0D0D0D"
_LABEL_COLOR    = "#FFFFFF"
_NODE_COLORS    = {"USDC": "#4F9BFF", "USDT": "#26C281"}
_EXCHANGE_COLOR = "#F0A500"    # gold: known exchange nodes

# Edge colours keyed by flow direction
_EDGE_EX_TO_EX   = "#CC99FF"   # purple : exchange → exchange
_EDGE_EX_OUTFLOW = "#FF6B6B"   # red    : exchange → anonymous (outflow)
_EDGE_EX_INFLOW  = "#4ECDC4"   # teal   : anonymous → exchange (inflow)
_EDGE_ANON_ANON  = "#777777"   # gray   : anonymous → anonymous

_FONT_SIZE_LBL = 7
_FONT_SIZE_TTL = 12

DEFAULT_THRESHOLD = 1_000_000   # USD — focus on whale-level activity
DEFAULT_MAX_ANON  = 50          # max anonymous (unlabelled) addresses per graph; None = all


# ── helpers ───────────────────────────────────────────────────────────────────

def _short_addr(address):
    """Truncate a raw address to a readable short form."""
    return address[:6] + "…" + address[-4:]


def _log_scale(values, v_min, v_max):
    log_v = np.log1p(values)
    lo, hi = log_v.min(), log_v.max()
    if hi == lo:
        return np.full(len(values), (v_min + v_max) / 2)
    return v_min + (v_max - v_min) * (log_v - lo) / (hi - lo)


def _edge_color(src, dst, exchange_names):
    src_ex = src in exchange_names
    dst_ex = dst in exchange_names
    if src_ex and dst_ex:
        return _EDGE_EX_TO_EX
    if src_ex:
        return _EDGE_EX_OUTFLOW
    if dst_ex:
        return _EDGE_EX_INFLOW
    return _EDGE_ANON_ANON


# ── main visualisation function ───────────────────────────────────────────────

def plot_transfer_graph(
    transfers_df,
    labels_df=None,
    output_dir="output",
    token="USDC",
    threshold=DEFAULT_THRESHOLD,
    max_anon_nodes=DEFAULT_MAX_ANON,
    date_tag="",
):
    """
    Build and save an exchange-merged transfer network graph for one token.

    Each exchange (e.g., Binance) is ONE node regardless of how many wallet
    addresses it has in labels.csv. Only transfers involving at least one
    exchange are shown. Top-N whale addresses form the remaining nodes.

    Args:
        transfers_df:   DataFrame with [from_address, to_address, amount, token].
        labels_df:      DataFrame with [address, exchange, label]. The 'exchange'
                        column is the canonical exchange name used for merging.
        output_dir:     Directory to save the PNG.
        token:          "USDC" or "USDT".
        threshold:      Minimum single-transfer amount (USD) to include.
        max_anon_nodes: Cap on unlabelled whale addresses to display.
        date_tag:       Optional date string appended to the filename.

    Returns:
        Absolute path to the saved PNG, or "" if data is insufficient.
    """
    # ── 1. filter by token + threshold ────────────────────────────────────────
    df = transfers_df[
        (transfers_df["token"] == token) & (transfers_df["amount"] >= threshold)
    ].copy()

    if df.empty:
        print(f"[graph] No {token} transfers >= ${threshold:,.0f}. Skipping.")
        return ""

    # ── 2. build address → exchange name map ──────────────────────────────────
    # Multiple wallet addresses all map to the same exchange canonical name,
    # e.g., "Binance 1" address and "Binance 10" address both → "Binance"
    addr_to_exname = {}         # raw address (lower) → "CEX"  (all exchanges merged)
    if labels_df is not None and not labels_df.empty:
        for _, row in labels_df.iterrows():
            addr = str(row["address"]).lower().strip()
            addr_to_exname[addr] = "CEX"

    exchange_names = {"CEX"} if addr_to_exname else set()

    # ── 3. map transfer endpoints: exchange addresses → canonical name ─────────
    # Unknown addresses keep their raw address string as the node identifier.
    def _resolve(addr):
        return addr_to_exname.get(addr, addr)

    df = df.copy()
    df["from_node"] = df["from_address"].map(_resolve)
    df["to_node"]   = df["to_address"].map(_resolve)

    # ── 4. restrict to exchange-adjacent transfers ────────────────────────────
    from_ex = df["from_node"].isin(exchange_names)
    to_ex   = df["to_node"].isin(exchange_names)
    df_adj  = df[from_ex | to_ex].copy()

    if df_adj.empty:
        print(f"[graph] {token}: no exchange labels matched — showing all transfers.")
        df_adj         = df.copy()
        exchange_names = set()

    # ── 5. compute per-node total volume (exchange names + raw addresses) ─────
    out_vol  = df_adj.groupby("from_node")["amount"].sum()
    in_vol   = df_adj.groupby("to_node")["amount"].sum()
    node_vol = (
        pd.concat([out_vol.rename("vol"), in_vol.rename("vol")])
        .groupby(level=0).sum()
    )

    # ── 6. select nodes: exchanges present in data + top-N whale addresses ────
    ex_in_data  = exchange_names & set(node_vol.index)
    anon_vols   = node_vol[~node_vol.index.isin(exchange_names)]
    top_anon    = (
        anon_vols.index.tolist() if max_anon_nodes is None
        else anon_vols.nlargest(max_anon_nodes).index.tolist()
    )
    keep_set    = ex_in_data | set(top_anon)
    print(
        f"[graph] {token}: {len(ex_in_data)} exchange nodes + "
        f"{len(top_anon)} whale nodes = {len(keep_set)} total"
    )

    # ── 7. aggregate edges between kept nodes ─────────────────────────────────
    df_edges  = df_adj[
        df_adj["from_node"].isin(keep_set) & df_adj["to_node"].isin(keep_set)
    ]
    # Remove self-loops that result from intra-exchange transfers
    df_edges  = df_edges[df_edges["from_node"] != df_edges["to_node"]]
    edges_agg = (
        df_edges.groupby(["from_node", "to_node"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total"})
    )

    # ── 8. build directed graph ───────────────────────────────────────────────
    G = nx.DiGraph()
    for node in keep_set:
        G.add_node(node, vol=float(node_vol.get(node, 0.0)))
    for _, row in edges_agg.iterrows():
        G.add_edge(row["from_node"], row["to_node"], weight=float(row["total"]))

    # Collapse bidirectional edges to net flow:
    # If both (A→B) and (B→A) exist, keep only the dominant direction with
    # the net amount, and remove the reverse edge entirely.
    for u, v in list(G.edges()):
        if not G.has_edge(u, v):   # already removed in a prior iteration
            continue
        if G.has_edge(v, u):
            w_fwd = G[u][v]["weight"]
            w_rev = G[v][u]["weight"]
            net = w_fwd - w_rev
            if net > 0:
                G[u][v]["weight"] = net
                G.remove_edge(v, u)
            elif net < 0:
                G[v][u]["weight"] = -net
                G.remove_edge(u, v)
            else:
                # Exactly equal — remove both (no net flow)
                G.remove_edge(u, v)
                if G.has_edge(v, u):
                    G.remove_edge(v, u)

    G.remove_nodes_from(list(nx.isolates(G)))

    node_list = list(G.nodes())
    n_nodes   = G.number_of_nodes()
    n_edges   = G.number_of_edges()

    if n_nodes == 0:
        print(f"[graph] {token}: no connected nodes after filtering. Skipping.")
        return ""

    print(f"[graph] {token} final graph: {n_nodes} nodes, {n_edges} edges")

    # ── 9. bipartite layout: inflow whales LEFT | CEX CENTER | outflow whales RIGHT
    #
    # After net-flow collapsing, each whale has at most one edge to/from CEX.
    # sent > 0     → net depositor  → left side (teal)
    # received > 0 → net withdrawer → right side (red)
    # ─────────────────────────────────────────────────────────────────────────
    ex_present   = [nd for nd in node_list if nd in exchange_names]
    anon_present = [nd for nd in node_list if nd not in exchange_names]

    cex_node = ex_present[0] if ex_present else None

    inflow_whales  = []   # net depositors  (whale → CEX dominant)
    outflow_whales = []   # net withdrawers (CEX → whale dominant)

    for nd in anon_present:
        sent     = G[nd][cex_node]["weight"] if (cex_node and G.has_edge(nd, cex_node)) else 0.0
        received = G[cex_node][nd]["weight"] if (cex_node and G.has_edge(cex_node, nd)) else 0.0
        if sent >= received:
            inflow_whales.append((nd, sent))
        else:
            outflow_whales.append((nd, received))

    # Sort by volume so largest flows are closest to center (mid-list)
    inflow_whales.sort(key=lambda x: x[1], reverse=True)
    outflow_whales.sort(key=lambda x: x[1], reverse=True)
    inflow_whales  = [nd for nd, _ in inflow_whales]
    outflow_whales = [nd for nd, _ in outflow_whales]

    def _column_positions(nodes, x):
        """Evenly space nodes vertically in a column at horizontal position x."""
        n = len(nodes)
        if n == 0:
            return {}
        # spread across [-1, 1] vertically, slightly compressed for padding
        ys = np.linspace(0.9, -0.9, n) if n > 1 else [0.0]
        return {nd: (x, float(y)) for nd, y in zip(nodes, ys)}

    pos = {}
    if cex_node:
        pos[cex_node] = (0.0, 0.0)

    x_spread = 1.6   # horizontal distance from CEX to whale columns
    pos.update(_column_positions(inflow_whales,  -x_spread))
    pos.update(_column_positions(outflow_whales,  x_spread))

    # ── 10. visual sizing ─────────────────────────────────────────────────────
    vols       = np.array([G.nodes[nd]["vol"] for nd in node_list])
    node_sizes = _log_scale(vols, v_min=200, v_max=2_500)
    node_sizes = np.array([
        s * 2.5 if nd in exchange_names else s
        for nd, s in zip(node_list, node_sizes)
    ])

    edge_list    = list(G.edges())
    edge_weights = np.array([G[u][v]["weight"] for u, v in edge_list])
    edge_widths  = _log_scale(edge_weights, v_min=0.8, v_max=7.0)

    # ── 11. figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, max(14, len(anon_present) * 0.45 + 4)))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")

    token_color = _NODE_COLORS.get(token, "#FFFFFF")

    # Column header labels
    ax.text(-x_spread, 1.05, "DEPOSIT  →  CEX",
            ha="center", va="bottom", color=_EDGE_EX_INFLOW,
            fontsize=11, fontweight="bold", transform=ax.transData)
    ax.text( x_spread, 1.05, "CEX  →  WITHDRAW",
            ha="center", va="bottom", color=_EDGE_EX_OUTFLOW,
            fontsize=11, fontweight="bold", transform=ax.transData)

    # Draw edges — only draw the dominant-direction edge for each whale:
    #   inflow  edges (whale → CEX) only for inflow_whales  (left side)
    #   outflow edges (CEX → whale) only for outflow_whales (right side)
    inflow_set  = set(inflow_whales)
    outflow_set = set(outflow_whales)

    for ecolor, allowed_whales in [
        (_EDGE_EX_INFLOW,  inflow_set),   # teal:  left-side whales only
        (_EDGE_EX_OUTFLOW, outflow_set),  # red:   right-side whales only
    ]:
        idxs = []
        for i, (u, v) in enumerate(edge_list):
            c = _edge_color(u, v, exchange_names)
            if c != ecolor:
                continue
            # whale endpoint must be on the correct side
            whale = u if v in exchange_names else v
            if whale not in allowed_whales:
                continue
            idxs.append(i)
        if not idxs:
            continue
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge_list[i] for i in idxs],
            ax=ax,
            edge_color=ecolor,
            width=[float(edge_widths[i]) for i in idxs],
            alpha=0.70,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.0",
            min_source_margin=10,
            min_target_margin=10,
        )

    # Draw nodes: whales first, CEX on top
    for is_ex in [False, True]:
        group_ids = [i for i, nd in enumerate(node_list) if (nd in exchange_names) == is_ex]
        if not group_ids:
            continue
        # Whale color: inflow side teal-tinted, outflow side red-tinted
        if is_ex:
            colors = _EXCHANGE_COLOR
        else:
            colors = [
                _EDGE_EX_INFLOW if node_list[i] in inflow_whales else _EDGE_EX_OUTFLOW
                for i in group_ids
            ]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node_list[i] for i in group_ids],
            ax=ax,
            node_size=node_sizes[np.array(group_ids)],
            node_color=colors,
            linewidths=2.5 if is_ex else 0.4,
            edgecolors="#FFFFFF" if is_ex else "#222222",
            alpha=0.88,
        )

    # Labels: always show CEX name; show ALL whale addresses (truncated)
    # Left-align inflow whale labels, right-align outflow whale labels
    cex_label = {nd: nd for nd in node_list if nd in exchange_names}
    inflow_labels  = {nd: _short_addr(nd) for nd in inflow_whales  if nd in G}
    outflow_labels = {nd: _short_addr(nd) for nd in outflow_whales if nd in G}

    # Inflow labels: right-aligned (label sits to the right of node, close to CEX)
    nx.draw_networkx_labels(G, pos, labels=inflow_labels, ax=ax,
                            font_size=6.5, font_color=_EDGE_EX_INFLOW,
                            horizontalalignment="right")
    # Outflow labels: left-aligned
    nx.draw_networkx_labels(G, pos, labels=outflow_labels, ax=ax,
                            font_size=6.5, font_color=_EDGE_EX_OUTFLOW,
                            horizontalalignment="left")
    # CEX label: large gold
    nx.draw_networkx_labels(G, pos, labels=cex_label, ax=ax,
                            font_size=13, font_color="#FFD700", font_weight="bold")

    # Volume annotations — only on the dominant-direction edge per whale
    for u, v in edge_list:
        if cex_node not in (u, v):
            continue
        whale = u if v in exchange_names else v
        # teal edge: only annotate if whale is on inflow side
        # red edge:  only annotate if whale is on outflow side
        if _edge_color(u, v, exchange_names) == _EDGE_EX_INFLOW and whale not in inflow_set:
            continue
        if _edge_color(u, v, exchange_names) == _EDGE_EX_OUTFLOW and whale not in outflow_set:
            continue
        w = G[u][v]["weight"]
        if w < threshold:
            continue
        xu, yu = pos[u]
        xv, yv = pos[v]
        xm, ym = (xu + xv) / 2, (yu + yv) / 2
        ecolor  = _edge_color(u, v, exchange_names)
        ax.text(xm, ym, f"${w / 1e6:.0f}M",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=ecolor, alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=_BG, alpha=0.7, edgecolor="none"))

    # ── 12. Top deposits / withdrawals summary boxes ──────────────────────────
    def _top_box(nodes, direction, x_anchor, color):
        """Render a ranked-list annotation box for top-N flows."""
        if not nodes:
            return
        flows = []
        for nd in nodes:
            if direction == "in" and G.has_edge(nd, cex_node):
                flows.append((nd, G[nd][cex_node]["weight"]))
            elif direction == "out" and G.has_edge(cex_node, nd):
                flows.append((nd, G[cex_node][nd]["weight"]))
        flows.sort(key=lambda x: x[1], reverse=True)
        lines = [
            f"  {_short_addr(nd)}   ${v / 1e6:.1f}M"
            for nd, v in flows[:8]
        ]
        title = "Top Deposits:" if direction == "in" else "Top Withdrawals:"
        ax.text(
            x_anchor, 0.01,
            title + "\n" + "\n".join(lines),
            transform=ax.transAxes,
            color="#CCCCCC", fontsize=6.5,
            verticalalignment="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#1A1A1A", alpha=0.75, edgecolor=color),
        )

    if cex_node:
        _top_box(inflow_whales,  "in",  0.01,  _EDGE_EX_INFLOW)
        _top_box(outflow_whales, "out", 0.55,  _EDGE_EX_OUTFLOW)

    # ── 13. title ─────────────────────────────────────────────────────────────
    shown_vol = float(df_edges["amount"].sum())
    ax.set_title(
        f"{token} CEX Flow Network  |  single transfer >= ${threshold / 1e6:.0f}M\n"
        f"{len(inflow_whales)} deposit whales  |  CEX (merged)  |  {len(outflow_whales)} withdraw whales"
        f"  |  shown volume ${shown_vol / 1e9:.2f}B",
        color=_LABEL_COLOR,
        fontsize=_FONT_SIZE_TTL,
        pad=14,
    )

    plt.tight_layout()

    # ── 15. save ──────────────────────────────────────────────────────────────
    suffix   = f"_{date_tag}" if date_tag else ""
    out_path = Path(output_dir) / f"transfer_graph_{token.lower()}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)

    print(f"[graph] Saved -> {out_path}")
    return str(out_path.resolve())


# ── convenience wrapper ───────────────────────────────────────────────────────

def plot_both_graphs(
    transfers_csv,
    labels_csv=None,
    output_dir="output",
    threshold=DEFAULT_THRESHOLD,
    max_anon_nodes=DEFAULT_MAX_ANON,
    date_tag="",
):
    """
    Load CSVs and render USDC + USDT transfer network graphs.

    Returns:
        (usdc_png_path, usdt_png_path)
    """
    print(f"[graph] Loading {transfers_csv} ...")
    transfers_df = pd.read_csv(transfers_csv)
    transfers_df["from_address"] = transfers_df["from_address"].str.lower()
    transfers_df["to_address"]   = transfers_df["to_address"].str.lower()

    labels_df = None
    if labels_csv and Path(labels_csv).exists():
        labels_df = pd.read_csv(labels_csv)
        labels_df["address"] = labels_df["address"].str.lower()

    usdc_path = plot_transfer_graph(
        transfers_df, labels_df, output_dir, "USDC", threshold, max_anon_nodes, date_tag
    )
    usdt_path = plot_transfer_graph(
        transfers_df, labels_df, output_dir, "USDT", threshold, max_anon_nodes, date_tag
    )
    return usdc_path, usdt_path
