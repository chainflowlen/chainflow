#!/usr/bin/env python3
"""
find_potential_whales.py — Identify potential whale addresses from daily transfer data.

Logic:
  1. Load transfers for a given date from data/YYYY-MM-DD/transfers.csv.
  2. Find all addresses that sent >= $1 M in at least one transfer.
  3. Count how many times each such address appears as a *recipient* (to_address)
     across the full day's transfer dataset — a proxy for "how many inflows did
     this address receive?".
  4. Keep only addresses with < 10 inbound transfers, since exchange deposit
     addresses typically receive hundreds of inflows while true whale wallets
     are quiet on the receiving side.
  5. Print a summary and save results to output/YYYY-MM-DD/potential_whales.csv.

Usage:
    python find_potential_whales.py                  # uses yesterday (UTC)
    python find_potential_whales.py 2026-05-16       # specific date
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

import config

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

WHALE_THRESHOLD: float = 1_000_000   # minimum single-transfer amount (USD)
MAX_INFLOWS: int = 10                # addresses with >= this many inflows are excluded


def find_potential_whales(target_date: date) -> pd.DataFrame:
    """
    Return a DataFrame of potential whale addresses for *target_date*.

    Columns:
        address         — wallet address
        total_sent      — total USD sent in transfers >= WHALE_THRESHOLD that day
        large_tx_count  — number of transfers >= WHALE_THRESHOLD sent
        inflow_count    — number of times the address received *any* transfer that day
        tokens          — comma-separated list of tokens used in large transfers
        large_transfers — number of individual large-send transactions
    """
    date_str = target_date.strftime("%Y-%m-%d")
    transfers_path = DATA_DIR / date_str / "transfers.csv"

    if not transfers_path.exists():
        raise FileNotFoundError(f"Transfer data not found: {transfers_path}")

    df = pd.read_csv(transfers_path)
    df["from_address"] = df["from_address"].str.lower().fillna("")
    df["to_address"] = df["to_address"].str.lower().fillna("")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # --- Step 1: addresses that sent at least one transfer >= $1 M ---
    large_sends = df[df["amount"] >= WHALE_THRESHOLD].copy()
    if large_sends.empty:
        print(f"No transfers >= ${WHALE_THRESHOLD:,.0f} found on {date_str}.")
        return pd.DataFrame(columns=[
            "address", "total_sent", "large_tx_count", "inflow_count", "tokens"
        ])

    # Aggregate per sender
    agg = (
        large_sends
        .groupby("from_address")
        .agg(
            total_sent=("amount", "sum"),
            large_tx_count=("amount", "count"),
            tokens=("token", lambda s: ",".join(sorted(s.unique()))),
        )
        .reset_index()
        .rename(columns={"from_address": "address"})
    )

    # --- Step 2: count inflows (to_address appearances) for each candidate ---
    inflow_counts = (
        df[df["to_address"].isin(agg["address"])]
        .groupby("to_address")
        .size()
        .reset_index(name="inflow_count")
        .rename(columns={"to_address": "address"})
    )

    result = agg.merge(inflow_counts, on="address", how="left")
    result["inflow_count"] = result["inflow_count"].fillna(0).astype(int)

    # --- Step 3: exclude addresses with too many inflows (likely exchange deposits) ---
    whales = (
        result[result["inflow_count"] < MAX_INFLOWS]
        .sort_values("total_sent", ascending=False)
        .reset_index(drop=True)
    )

    return whales


def main() -> None:
    # Determine target date
    if len(sys.argv) >= 2:
        try:
            target_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"ERROR: Invalid date '{sys.argv[1]}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        target_date = date.today() - timedelta(days=1)  # yesterday UTC

    date_str = target_date.strftime("%Y-%m-%d")
    print(f"Scanning potential whales for {date_str} …")
    print(f"  Threshold : >= ${WHALE_THRESHOLD:,.0f} sent in a single transfer")
    print(f"  Max inflows: < {MAX_INFLOWS} inbound transfers to the address")

    try:
        whales = find_potential_whales(target_date)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if whales.empty:
        print("No potential whales found after filtering.")
        return

    # Print summary
    print(f"\nFound {len(whales)} potential whale address(es):\n")
    print(
        whales[["address", "total_sent", "large_tx_count", "inflow_count", "tokens"]]
        .to_string(index=False)
    )

    # Save output
    out_dir = OUTPUT_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "potential_whales.csv"
    whales.to_csv(out_path, index=False)
    print(f"\nSaved {len(whales)} rows → {out_path}")


if __name__ == "__main__":
    main()
