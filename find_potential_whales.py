"""
find_potential_whales.py — Identify potential whale addresses from transfer data.

Algorithm
─────────
1. Load all transfer CSV files for the given date from ../../data/<DATE>/.
2. Build a directed graph: nodes = addresses, edges = individual transfers
   (multi-edges are allowed; each transfer becomes one edge with an 'amount').
3. Filter nodes that satisfy ALL three conditions:
     • total transferred volume (sent + received) ≥ VOLUME_THRESHOLD
     • in-degree  < MAX_IN_DEGREE
     • out-degree < MAX_OUT_DEGREE
4. Print and optionally save the results.

The degree constraints capture the "whale" intuition: a genuine whale moves
large amounts but through very few counterparties, as opposed to an exchange
or mixer that fans out to hundreds of addresses.

Usage
─────
# Default thresholds (1 M USD volume, in/out degree < 10)
python find_potential_whales.py

# Custom thresholds
python find_potential_whales.py --date 2026-05-16 --volume 500000 --max-in 5 --max-out 5

# Save results to CSV
python find_potential_whales.py --date 2026-05-16 --output output/whales.csv
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, WHALE_THRESHOLD

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_IN_DEGREE  = 10
DEFAULT_MAX_OUT_DEGREE = 10


# ── data loading ──────────────────────────────────────────────────────────────

def load_transfers(data_date: str) -> pd.DataFrame:
    """
    Load all CSV transfer files for *data_date* from ``../../data/<data_date>/``.

    Each CSV must contain at minimum the columns:
        hash, from_address, to_address, amount, token

    Returns an empty DataFrame if no files are found.
    """
    data_path = Path(DATA_DIR) / data_date
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        print(f"[warn] No CSV files found in {data_path}", file=sys.stderr)
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as exc:
            print(f"[warn] Could not read {f}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    transfers = pd.concat(dfs, ignore_index=True)

    # Normalise addresses to lowercase for consistent matching
    transfers["from_address"] = transfers["from_address"].str.lower().str.strip()
    transfers["to_address"]   = transfers["to_address"].str.lower().str.strip()

    # Deduplicate by tx hash + token to avoid double-counting when multiple
    # CSVs overlap (the same transfer hash can appear in USDC and USDT files).
    transfers = transfers.drop_duplicates(subset=["hash", "from_address", "to_address", "token"])

    return transfers


# ── graph construction ────────────────────────────────────────────────────────

def build_graph(transfers: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build a directed multigraph from the transfer DataFrame.

    Each row becomes a directed edge  from_address → to_address  with
    attributes ``amount`` (USD) and ``token``.
    """
    G = nx.MultiDiGraph()

    for _, row in transfers.iterrows():
        G.add_edge(
            row["from_address"],
            row["to_address"],
            amount=float(row["amount"]),
            token=row.get("token", ""),
        )

    return G


# ── whale detection ───────────────────────────────────────────────────────────

def compute_node_volumes(G: nx.MultiDiGraph) -> dict[str, float]:
    """
    Return a mapping of address → total USD volume (sent + received).
    """
    volumes: dict[str, float] = {}

    for u, v, data in G.edges(data=True):
        amt = data.get("amount", 0.0)
        volumes[u] = volumes.get(u, 0.0) + amt
        volumes[v] = volumes.get(v, 0.0) + amt

    return volumes


def find_potential_whales(
    G: nx.MultiDiGraph,
    volume_threshold: float = WHALE_THRESHOLD,
    max_in_degree: int  = DEFAULT_MAX_IN_DEGREE,
    max_out_degree: int = DEFAULT_MAX_OUT_DEGREE,
) -> pd.DataFrame:
    """
    Identify whale candidates satisfying:
        total_volume >= volume_threshold
        in_degree    <  max_in_degree
        out_degree   <  max_out_degree

    Returns a DataFrame sorted by total_volume descending with columns:
        address, total_volume, in_degree, out_degree,
        total_sent, total_received
    """
    node_volumes = compute_node_volumes(G)

    # Pre-compute sent / received volumes separately
    sent_vol:     dict[str, float] = {}
    received_vol: dict[str, float] = {}
    for u, v, data in G.edges(data=True):
        amt = data.get("amount", 0.0)
        sent_vol[u]     = sent_vol.get(u, 0.0)     + amt
        received_vol[v] = received_vol.get(v, 0.0) + amt

    rows = []
    for node in G.nodes():
        in_deg  = G.in_degree(node)
        out_deg = G.out_degree(node)
        vol     = node_volumes.get(node, 0.0)

        if (
            vol >= volume_threshold
            and in_deg  < max_in_degree
            and out_deg < max_out_degree
        ):
            rows.append(
                {
                    "address":        node,
                    "total_volume":   vol,
                    "total_sent":     sent_vol.get(node, 0.0),
                    "total_received": received_vol.get(node, 0.0),
                    "in_degree":      in_deg,
                    "out_degree":     out_deg,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "address", "total_volume", "total_sent",
                "total_received", "in_degree", "out_degree",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("total_volume", ascending=False)
        .reset_index(drop=True)
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find potential whale addresses in the transfer graph."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date folder to load (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=WHALE_THRESHOLD,
        help=f"Minimum total USD volume. Default: {WHALE_THRESHOLD:,.0f}",
    )
    parser.add_argument(
        "--max-in",
        type=int,
        default=DEFAULT_MAX_IN_DEGREE,
        dest="max_in",
        help=f"Maximum in-degree (exclusive). Default: {DEFAULT_MAX_IN_DEGREE}",
    )
    parser.add_argument(
        "--max-out",
        type=int,
        default=DEFAULT_MAX_OUT_DEGREE,
        dest="max_out",
        help=f"Maximum out-degree (exclusive). Default: {DEFAULT_MAX_OUT_DEGREE}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[*] Loading transfers for {args.date} …")
    transfers = load_transfers(args.date)
    if transfers.empty:
        print("[!] No transfer data found. Exiting.")
        sys.exit(1)
    print(f"    {len(transfers):,} transfers loaded.")

    print("[*] Building transfer graph …")
    G = build_graph(transfers)

    # ── graph summary ─────────────────────────────────────────────────────────
    degrees     = [d for _, d in G.degree()]
    in_degrees  = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    all_amounts = [data["amount"] for _, _, data in G.edges(data=True)]

    print(f"\n  {'─'*40}")
    print(f"  Graph Summary")
    print(f"  {'─'*40}")
    print(f"  Nodes            : {G.number_of_nodes():>12,}")
    print(f"  Edges            : {G.number_of_edges():>12,}")
    if degrees:
        print(f"  Avg degree       : {sum(degrees)/len(degrees):>12.2f}")
        print(f"  Max in-degree    : {max(in_degrees):>12,}")
        print(f"  Max out-degree   : {max(out_degrees):>12,}")
        print(f"  Avg in-degree    : {sum(in_degrees)/len(in_degrees):>12.2f}")
        print(f"  Avg out-degree   : {sum(out_degrees)/len(out_degrees):>12.2f}")
    if all_amounts:
        total_vol = sum(all_amounts)
        print(f"  Total volume ($) : {total_vol:>12,.2f}")
        print(f"  Avg tx amount($) : {total_vol/len(all_amounts):>12,.2f}")
        print(f"  Max tx amount($) : {max(all_amounts):>12,.2f}")
    print(f"  {'─'*40}\n")

    print(
        f"[*] Searching for whales "
        f"(volume ≥ ${args.volume:,.0f}, "
        f"in_degree < {args.max_in}, "
        f"out_degree < {args.max_out}) …"
    )
    whales = find_potential_whales(
        G,
        volume_threshold=args.volume,
        max_in_degree=args.max_in,
        max_out_degree=args.max_out,
    )

    if whales.empty:
        print("[!] No whale candidates found with these thresholds.")
        sys.exit(0)

    print(f"\n    Found {len(whales)} potential whale(s):\n")
    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_colwidth", 44)
    print(whales.to_string(index=True))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        whales.to_csv(out_path, index=False)
        print(f"\n[*] Results saved to {out_path}")


if __name__ == "__main__":
    main()
