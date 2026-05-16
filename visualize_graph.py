#!/usr/bin/env python3
"""
visualize_graph.py — Standalone runner for transfer network graphs.

Produces two PNGs (USDC + USDT) in output/<date>/ using today's transfer data,
and writes a CEX flow summary CSV: output/<date>/cex_flow_summary.csv

Usage:
    python visualize_graph.py [--date YYYY-MM-DD] [--threshold AMOUNT] [--max-anon N]

Examples:
    python visualize_graph.py
    python visualize_graph.py --date 2026-05-16
    python visualize_graph.py --date 2026-05-16 --threshold 500000 --max-anon 50
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.graph import plot_both_graphs

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
LABELS_CSV = DATA_DIR / "labels.csv"


def build_cex_summary(transfers_csv: str, labels_csv: str, output_dir: str, date_tag: str) -> str:
    """
    Compute per-exchange per-token inflow / outflow / netflow across ALL transfers
    (no threshold filter — full picture).

    Writes output_dir/cex_flow_summary_<date>.csv and returns the path.
    """
    df = pd.read_csv(transfers_csv)
    df["from_address"] = df["from_address"].str.lower().str.strip()
    df["to_address"]   = df["to_address"].str.lower().str.strip()

    labels = pd.read_csv(labels_csv)
    labels["address"] = labels["address"].str.lower().str.strip()

    # address → exchange name
    addr_to_ex = dict(zip(labels["address"], labels["exchange"]))
    ex_addrs   = set(addr_to_ex)

    records = []
    for token in ("USDC", "USDT"):
        t = df[df["token"] == token]

        # inflow: transfers INTO any exchange address
        inflow = (
            t[t["to_address"].isin(ex_addrs)]
            .copy()
            .assign(exchange=lambda x: x["to_address"].map(addr_to_ex))
            .groupby("exchange")["amount"].sum()
            .rename("inflow")
        )

        # outflow: transfers OUT OF any exchange address
        outflow = (
            t[t["from_address"].isin(ex_addrs)]
            .copy()
            .assign(exchange=lambda x: x["from_address"].map(addr_to_ex))
            .groupby("exchange")["amount"].sum()
            .rename("outflow")
        )

        summary = (
            pd.concat([inflow, outflow], axis=1)
            .fillna(0)
            .reset_index()
        )
        summary["token"]   = token
        summary["netflow"] = summary["inflow"] - summary["outflow"]
        records.append(summary)

    result = (
        pd.concat(records, ignore_index=True)
        [["exchange", "token", "inflow", "outflow", "netflow"]]
        .sort_values(["token", "inflow"], ascending=[True, False])
        .reset_index(drop=True)
    )

    out_path = Path(output_dir) / f"cex_flow_summary_{date_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\n  CEX flow summary → {out_path}")
    print(result.to_string(index=False))
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render USDC / USDT transfer network graphs.")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date folder to read transfers from (default: today, UTC).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1_000_000,
        help="Minimum single-transfer amount in USD to include (default: 1 000 000).",
    )
    parser.add_argument(
        "--max-anon",
        type=int,
        default=None,
        dest="max_anon",
        help="Max number of anonymous whale addresses to include (default: all).",
    )
    args = parser.parse_args()

    transfers_csv = DATA_DIR / args.date / "transfers.csv"
    output_dir    = OUTPUT_DIR / args.date

    if not transfers_csv.exists():
        print(f"ERROR: {transfers_csv} not found. Run main.py first to fetch data.")
        raise SystemExit(1)

    labels_csv_str = str(LABELS_CSV) if LABELS_CSV.exists() else None

    usdc_png, usdt_png = plot_both_graphs(
        transfers_csv=str(transfers_csv),
        labels_csv=labels_csv_str,
        output_dir=str(output_dir),
        threshold=args.threshold,
        max_anon_nodes=args.max_anon,
        date_tag=args.date,
    )

    if labels_csv_str:
        build_cex_summary(
            transfers_csv=str(transfers_csv),
            labels_csv=labels_csv_str,
            output_dir=str(output_dir),
            date_tag=args.date,
        )

    print("\nDone.")
    if usdc_png:
        print(f"  USDC graph → {usdc_png}")
    if usdt_png:
        print(f"  USDT graph → {usdt_png}")


if __name__ == "__main__":
    main()
