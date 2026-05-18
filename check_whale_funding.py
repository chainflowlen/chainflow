#!/usr/bin/env python3
"""
check_whale_funding.py — Check whether potential whale addresses were funded by a CEX.

For each address in potential_whales.csv, this script uses the Etherscan API to
find the first ETH transaction received by that address (i.e. who "funded" it for
gas), then checks whether the funder is a known CEX in data/labels.csv.

Adds a `funded_by` column:
  - Exchange name (e.g. "Binance") if the first funder is a known CEX
  - "Unknown" otherwise

Results are printed and saved back to potential_whales.csv (in-place).

Usage:
    python check_whale_funding.py                  # uses yesterday (UTC)
    python check_whale_funding.py 2026-05-16       # specific date

Prerequisites:
    Set ETHERSCAN_API_KEY in .env  (free tier: https://etherscan.io/myapikey)
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

import config

DATA_DIR   = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

ETHERSCAN_API = "https://api.etherscan.io/api"
RATE_LIMIT_DELAY = 0.25   # 4 req/s — stays under free-tier 5 req/s limit


def _get_first_funder(address: str, api_key: str) -> str | None:
    """
    Return the address that sent the first inbound ETH transaction to `address`,
    or None if no such transaction is found.
    """
    params = {
        "module":     "account",
        "action":     "txlist",
        "address":    address,
        "startblock": 0,
        "endblock":   99999999,
        "sort":       "asc",
        "page":       1,
        "offset":     5,          # grab first 5 txs, look for an inbound one
        "apikey":     api_key,
    }
    try:
        resp = requests.get(ETHERSCAN_API, params=params, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") != "1":
            return None
        txs = result.get("result", [])
        for tx in txs:
            if tx.get("to", "").lower() == address.lower():
                return tx.get("from", "").lower()
    except Exception:
        pass
    return None


def check_funding(target_date: date, api_key: str) -> pd.DataFrame:
    date_str = target_date.strftime("%Y-%m-%d")
    whales_path = OUTPUT_DIR / date_str / "potential_whales.csv"

    if not whales_path.exists():
        raise FileNotFoundError(f"potential_whales.csv not found: {whales_path}")

    labels_path = DATA_DIR / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    whales = pd.read_csv(whales_path)
    labels = pd.read_csv(labels_path)
    labels["address"] = labels["address"].str.lower()

    # Build lookup: address → exchange name
    cex_lookup: dict[str, str] = dict(zip(labels["address"], labels["exchange"]))

    funded_by_list: list[str] = []
    total = len(whales)

    for i, row in whales.iterrows():
        addr = row["address"].lower()
        funder = _get_first_funder(addr, api_key)
        time.sleep(RATE_LIMIT_DELAY)

        if funder and funder in cex_lookup:
            label = cex_lookup[funder]
        else:
            label = "Unknown"

        funded_by_list.append(label)
        print(f"  [{i+1:>4}/{total}] {addr[:12]}…  funded by: {label}")

    whales["funded_by"] = funded_by_list
    return whales


def main() -> None:
    if len(sys.argv) >= 2:
        try:
            target_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"ERROR: Invalid date '{sys.argv[1]}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        target_date = date.today() - timedelta(days=1)

    if not config.ETHERSCAN_API_KEY:
        print(
            "ERROR: ETHERSCAN_API_KEY is not set.\n"
            "Add ETHERSCAN_API_KEY=<your_key> to .env  "
            "(free key at https://etherscan.io/myapikey)"
        )
        sys.exit(1)

    date_str = target_date.strftime("%Y-%m-%d")
    print(f"Checking funding sources for {date_str} …\n")

    try:
        result = check_funding(target_date, config.ETHERSCAN_API_KEY)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Summary
    cex_funded = result[result["funded_by"] != "Unknown"]
    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Total checked  : {len(result)}")
    print(f"  Funded by CEX  : {len(cex_funded)}  ({len(cex_funded)/len(result)*100:.1f}%)")
    if not cex_funded.empty:
        print(cex_funded.groupby("funded_by").size().sort_values(ascending=False).to_string())

    # Save back (overwrite with new funded_by column)
    out_path = OUTPUT_DIR / date_str / "potential_whales.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
