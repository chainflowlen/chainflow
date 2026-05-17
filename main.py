#!/usr/bin/env python3
"""
ChainFlow Lens — main pipeline.

Executes the full pipeline in sequence:
  Step 1  Fetch yesterday's USDC/USDT transfers via Alchemy (exact 00:00–24:00 UTC)
  Step 2  Save raw transfers to data/YYYY-MM-DD/transfers.csv
  Step 3  Calculate CEX netflow (per exchange, per token)
  Step 4  Compute flow signal + whale alerts
  Step 5  Build 7-day chart from the last 7 saved daily CSVs (no extra API calls)
  Step 6  Write three post templates to output/posts/

Usage:
    python main.py

Prerequisites:
    Copy .env.example to .env and set ALCHEMY_API_KEY.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import config
from src.fetch import fetch_transfers_for_date
from src.metrics import calculate_netflow, daily_netflow
from src.signals import flow_signal, whale_signal
from src.chart import plot_netflow_chart
from src.content import stablecoin_flow_post, whale_alert_post, netflow_chart_post

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)


def _banner(step: int, total: int, msg: str) -> None:
    print(f"\n[{step}/{total}] {msg}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Directories are named after the data date (yesterday in UTC)
    yesterday_date = date.today() - timedelta(days=1)
    today = yesterday_date.strftime("%Y-%m-%d")  # used as folder name throughout
    run_data_dir = DATA_DIR / today
    run_output_dir = OUTPUT_DIR / today
    run_data_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data date: {today} (yesterday UTC)  →  data/{today}/  |  output/{today}/")

    if not config.ALCHEMY_API_KEY:
        print(
            "ERROR: ALCHEMY_API_KEY is not set.\n"
            "Copy .env.example → .env and add your key, then re-run."
        )
        sys.exit(1)

    labels_path = DATA_DIR / "labels.csv"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        sys.exit(1)

    labels = pd.read_csv(labels_path)
    labels["address"] = labels["address"].str.lower()
    cex_addresses: list[str] = labels["address"].tolist()

    # ------------------------------------------------------------------
    # STEP 1 & 2 — Fetch yesterday's transfers and save to CSV
    # ------------------------------------------------------------------
    _banner(1, 6, f"Fetching USDC/USDT transfers for {today} (00:00–24:00 UTC)…")
    transfers_24h = fetch_transfers_for_date(config.ALCHEMY_API_KEY, yesterday_date)

    transfers_path = run_data_dir / "transfers.csv"
    transfers_24h.to_csv(transfers_path, index=False)
    print(f"  Saved {len(transfers_24h):,} rows → {transfers_path}")

    # ------------------------------------------------------------------
    # STEP 3 — CEX Netflow (per exchange, per token)
    # ------------------------------------------------------------------
    _banner(2, 6, "Calculating 24h CEX netflow…")
    netflow_df = calculate_netflow(transfers_24h, labels)

    if netflow_df.empty:
        print("  No CEX activity found in the 24h window.")
    else:
        print(netflow_df.to_string(index=False))

    netflow_path = run_output_dir / "netflow_24h.csv"
    netflow_df.to_csv(netflow_path, index=False)
    print(f"  Saved → {netflow_path}")

    # ------------------------------------------------------------------
    # STEP 4 — Flow signal + Whale signal
    # ------------------------------------------------------------------
    _banner(3, 6, "Computing signals…")

    signal = flow_signal(transfers_24h, labels)
    print(
        f"  Flow  — inflow: ${signal['total_inflow']:>14,.0f}"
        f"  outflow: ${signal['total_outflow']:>14,.0f}"
        f"  net: ${signal['net_flow']:>+14,.0f}"
    )

    whales = whale_signal(transfers_24h, threshold=config.WHALE_THRESHOLD)
    print(
        f"  Whale — {len(whales)} transfer(s) above "
        f"${config.WHALE_THRESHOLD:,.0f}"
    )

    whale_path = run_output_dir / "whale_alerts.csv"
    whales.to_csv(whale_path, index=False)
    print(f"  Saved whale alerts → {whale_path}")

    # ------------------------------------------------------------------
    # STEP 5 — 7-day chart (assembled from saved daily CSVs, no API call)
    # ------------------------------------------------------------------
    _banner(4, 6, "Building 7-day netflow chart from saved daily CSVs…")
    csv_frames: list[pd.DataFrame] = []
    for i in range(7):
        d = yesterday_date - timedelta(days=i)
        p = DATA_DIR / d.strftime("%Y-%m-%d") / "transfers.csv"
        if p.exists():
            csv_frames.append(pd.read_csv(p))
            print(f"  Loaded {p}")
        else:
            print(f"  (missing: {p})")

    if csv_frames:
        cex_flows_7d = pd.concat(csv_frames, ignore_index=True).drop_duplicates(
            subset=["hash", "token", "from_address", "to_address"]
        )
    else:
        cex_flows_7d = pd.DataFrame(
            columns=["hash", "time", "from_address", "to_address", "amount", "token"]
        )
    print(f"  7-day dataset: {len(cex_flows_7d):,} transfers across {len(csv_frames)} day(s)")

    daily_df = daily_netflow(cex_flows_7d, labels)
    chart_path = plot_netflow_chart(
        daily_df,
        output_path=str(run_output_dir / "netflow_7d.png"),
    )
    print(f"  Chart saved → {chart_path}")

    # ------------------------------------------------------------------
    # STEP 6 — Post templates
    # ------------------------------------------------------------------
    _banner(5, 6, "Generating post templates…")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    posts = {
        "stablecoin_flow": stablecoin_flow_post(signal, ts),
        "whale_alert": whale_alert_post(whales),
        "netflow_chart": netflow_chart_post(chart_path, signal, ts[:10]),
    }

    posts_dir = run_output_dir / "posts"
    posts_dir.mkdir(exist_ok=True)

    for name, content in posts.items():
        post_file = posts_dir / f"{name}.txt"
        post_file.write_text(content, encoding="utf-8")
        print(f"\n  ── {post_file} ──")
        print(content)

    _banner(6, 6, f"Done. All outputs saved to output/{today}/")
    print(
        f"\n  data/{today}/transfers.csv\n"
        f"  output/{today}/netflow_24h.csv\n"
        f"  output/{today}/whale_alerts.csv\n"
        f"  output/{today}/netflow_7d.png\n"
        f"  output/{today}/posts/stablecoin_flow.txt\n"
        f"  output/{today}/posts/whale_alert.txt\n"
        f"  output/{today}/posts/netflow_chart.txt\n"
    )


if __name__ == "__main__":
    main()
