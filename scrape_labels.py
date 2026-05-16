#!/usr/bin/env python3
"""
scrape_labels.py — Fetch CEX wallet address labels from GitHub-hosted data.

Source
──────
  brianleect/etherscan-labels  (github.com/brianleect/etherscan-labels)

  This public repo mirrors Etherscan's label database as raw JSON files with
  zero bot-protection — each exchange is one lightweight JSON request:
    https://raw.githubusercontent.com/brianleect/etherscan-labels/main/
        data/etherscan/accounts/{slug}.json

  Format: {"0xaddress": "Label Name", ...}

  The repo is updated periodically and currently tracks 400+ labelled entities
  across all categories, including 30+ CEX exchanges.

Output
──────
Appends new rows to data/labels.csv (deduplicated by address).
Columns: address, exchange, label

Usage
─────
    python scrape_labels.py                            # all configured CEX
    python scrape_labels.py --exchanges binance okx    # selected only
    python scrape_labels.py --dry-run                  # preview, no file write
    python scrape_labels.py --output path/to/out.csv   # custom output path
    python scrape_labels.py --github-token ghp_xxx     # raise API rate limit
"""

import sys
import time
import argparse
from pathlib import Path

import requests
import pandas as pd


# ── configuration ─────────────────────────────────────────────────────────────

DEFAULT_OUTPUT  = Path("data/labels.csv")
REQUEST_TIMEOUT = 20    # seconds per HTTP call
RATE_DELAY      = 0.3   # seconds between requests (raw.githubusercontent.com
                        # has no hard rate-limit, but be a good citizen)

_RAW_BASE = (
    "https://raw.githubusercontent.com/brianleect/etherscan-labels"
    "/main/data/etherscan/accounts"
)

# exchange display name → GitHub slug(s) in priority order.
# Multiple slugs let us merge wallets labelled under different filenames
# (e.g.  OKX has both "okex" and "okx" files).
EXCHANGE_SLUGS = {
    "Binance":    ["binance", "binance-us"],
    "Coinbase":   ["coinbase"],
    "OKX":        ["okx", "okex"],
    "Kraken":     ["kraken"],
    "Huobi":      ["huobi"],
    "KuCoin":     ["kucoin"],
    "Bitfinex":   ["bitfinex"],
    "Bybit":      ["bybit"],
    "Gate.io":    ["gate-io"],
    "MEXC":       ["mexc"],
    "Bitget":     ["bitget"],
    "Crypto.com": ["crypto-com"],
    "Gemini":     ["gemini"],
    "Bitstamp":   ["bitstamp"],
    "Poloniex":   ["poloniex"],
    "Bittrex":    ["bittrex"],
    "Bithumb":    ["bithumb"],
    "FTX":        ["ftx"],      # defunct but useful for historical data
}


# ── per-exchange fetcher ───────────────────────────────────────────────────────

def _fetch_exchange(
    exchange_name,
    slugs,
    session,
    github_token=None,
):
    """
    Download address-label mappings for one exchange from all its slugs.

    Returns a list of {address, exchange, label} dicts.
    """
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    records = []
    seen = set()

    for slug in slugs:
        url = f"{_RAW_BASE}/{slug}.json"
        print(f"  [{exchange_name}]  GET {url}")

        try:
            resp = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as err:
            print(f"  [error] {exchange_name}/{slug}: {err}")
            time.sleep(RATE_DELAY)
            continue

        if resp.status_code == 404:
            # Slug doesn't exist in this repo — silently skip
            time.sleep(RATE_DELAY)
            continue

        if resp.status_code != 200:
            print(f"  [warn] {exchange_name}/{slug}: HTTP {resp.status_code}")
            time.sleep(RATE_DELAY)
            continue

        try:
            data: dict[str, str] = resp.json()
        except ValueError as err:
            print(f"  [warn] {exchange_name}/{slug}: JSON parse error — {err}")
            time.sleep(RATE_DELAY)
            continue

        for addr_raw, label in data.items():
            addr = addr_raw.strip().lower()
            if addr and addr not in seen:
                seen.add(addr)
                records.append({
                    "address":  addr,
                    "exchange": exchange_name,
                    "label":    str(label).strip(),
                })

        print(f"  [{exchange_name}/{slug}] +{len(data)} addresses")
        time.sleep(RATE_DELAY)

    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CEX wallet address labels from brianleect/etherscan-labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available exchanges:  " + ", ".join(EXCHANGE_SLUGS),
    )
    parser.add_argument(
        "--exchanges", nargs="+", metavar="NAME",
        help="Exchange names to fetch (case-insensitive, partial match). Default: all.",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print collected results without writing to disk.",
    )
    parser.add_argument(
        "--github-token", default=None, dest="github_token",
        metavar="TOKEN",
        help=(
            "Optional GitHub personal-access token to raise raw.githubusercontent.com "
            "rate limits. Not required for normal use."
        ),
    )
    args = parser.parse_args()

    # ── resolve target exchanges ───────────────────────────────────────────────
    targets = dict(EXCHANGE_SLUGS)
    if args.exchanges:
        requested = {e.lower() for e in args.exchanges}
        targets = {
            name: slugs
            for name, slugs in EXCHANGE_SLUGS.items()
            if any(r in name.lower() for r in requested)
        }
        if not targets:
            print(f"No matching exchanges. Available: {list(EXCHANGE_SLUGS)}")
            sys.exit(1)

    # ── load existing labels ───────────────────────────────────────────────────
    output_path = Path(args.output)
    existing = pd.DataFrame(columns=["address", "exchange", "label"])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing["address"] = existing["address"].str.lower().str.strip()
        print(f"Loaded {len(existing)} existing labels from {output_path}\n")

    existing_addresses = set(existing["address"].dropna())

    # ── fetch ─────────────────────────────────────────────────────────────────
    all_new = []
    session = requests.Session()

    for name, slugs in targets.items():
        print(f"\n{'─' * 60}")
        print(f"  {name}  (slugs: {', '.join(slugs)})")
        print(f"{'─' * 60}")

        records = _fetch_exchange(name, slugs, session, args.github_token)

        seen_new = {r["address"] for r in all_new}
        new_rows = [
            r for r in records
            if r["address"] not in existing_addresses
            and r["address"] not in seen_new
        ]
        all_new.extend(new_rows)

        skipped = len(records) - len(new_rows)
        print(
            f"  → {len(new_rows)} new  |  {skipped} duplicates skipped  "
            f"|  {len(records)} total fetched"
        )

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Total new addresses collected: {len(all_new)}")

    if not all_new:
        print("  Nothing new to add — done.")
        return

    new_df = pd.DataFrame(all_new)[["address", "exchange", "label"]]

    if args.dry_run:
        print("\n[dry-run] Would add:")
        pd.set_option("display.max_rows", 50)
        print(new_df.to_string(index=False))
        return

    # ── write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        .drop_duplicates("address")
        .sort_values(["exchange", "label"])
        .reset_index(drop=True)
    )
    combined.to_csv(output_path, index=False)

    added = len(combined) - len(existing)
    print(f"  Wrote {len(combined)} total addresses to {output_path}")
    print(f"  (was {len(existing)}, added {added} new)\n")


if __name__ == "__main__":
    main()
