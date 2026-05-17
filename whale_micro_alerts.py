#!/usr/bin/env python3
"""
whale_micro_alerts.py — Micro-level whale deposit narrative generator.

Reads 24h transfer data, finds large deposits to a CEX that happened
*before* that exchange's hourly inflow spike, and formats them as
human-readable social-media alerts.

Algorithm
─────────
1. Load transfers.csv + labels.csv for the given date.
2. Tag every transfer whose `to_address` is a known CEX wallet.
3. Bin tagged transfers into 1-hour buckets per (exchange, token).
4. Flag spike hours: hourly inflow > mean + SPIKE_SIGMA * std
   (with a minimum absolute inflow of MIN_SPIKE_USD to skip noise).
5. For each spike, retrieve whale deposits (amount ≥ WHALE_THRESHOLD)
   that arrived from a *non-CEX* address within the LEAD_HOURS window
   before the spike hour.
6. Render one alert post per unique whale tx (deduplicated).

Usage
─────
    python whale_micro_alerts.py [YYYY-MM-DD]   # explicit date
    python whale_micro_alerts.py                # auto-picks latest data dir
    python whale_micro_alerts.py --debug        # also print hourly breakdown

Output
──────
  • Printed to stdout.
  • Saved to output/<date>/posts/micro_alerts.txt
"""

import sys
import pandas as pd
from pathlib import Path

import config

# ── Tunable parameters ────────────────────────────────────────────────────────
WHALE_THRESHOLD: float = config.WHALE_THRESHOLD   # min tx size to call a "whale"
SPIKE_SIGMA: float = 1.5          # std-dev above hourly mean → spike
MIN_SPIKE_USD: float = 5_000_000  # ignore spikes smaller than $5M (noise filter)
LEAD_HOURS: int = 6               # look-back window before spike start (hours)
MAX_ALERTS: int = 5               # max posts to generate / save

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_usd(amount: float) -> str:
    """Compact USD string: $1.2B / $345M / $12.3M / $1,234,567."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.0f}M"
    return f"${amount:,.0f}"


def _short_addr(addr: str) -> str:
    return f"{addr[:6]}…{addr[-4:]}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_path = DATA_DIR / "labels.csv"
    transfers_path = DATA_DIR / date_str / "transfers.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not transfers_path.exists():
        raise FileNotFoundError(f"Transfers file not found: {transfers_path}")

    labels = pd.read_csv(labels_path)
    labels["address"] = labels["address"].str.lower()

    transfers = pd.read_csv(transfers_path, parse_dates=["time"])
    transfers["from_address"] = transfers["from_address"].str.lower()
    transfers["to_address"] = transfers["to_address"].str.lower()

    return transfers, labels


# ── CEX deposit detection ─────────────────────────────────────────────────────

def tag_cex_deposits(
    transfers: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Return transfers where `to_address` is a known CEX wallet.
    Adds columns: exchange, cex_label, from_is_cex.
    """
    addr_map: dict[str, dict] = (
        labels.set_index("address")[["exchange", "label"]].to_dict("index")
    )
    cex_addr_set: set[str] = set(addr_map.keys())

    mask = transfers["to_address"].isin(cex_addr_set)
    deposits = transfers[mask].copy()

    deposits["exchange"] = deposits["to_address"].map(
        lambda a: addr_map[a]["exchange"]
    )
    deposits["cex_label"] = deposits["to_address"].map(
        lambda a: addr_map[a]["label"]
    )
    # Flag whether the sender is also a CEX address (internal CEX moves)
    deposits["from_is_cex"] = deposits["from_address"].isin(cex_addr_set)

    return deposits.sort_values("time").reset_index(drop=True)


# ── Inflow spike detection ────────────────────────────────────────────────────

def detect_spike_hours(
    deposits: pd.DataFrame,
    sigma: float = SPIKE_SIGMA,
    min_inflow: float = MIN_SPIKE_USD,
) -> pd.DataFrame:
    """
    Bin CEX deposits into 1-hour windows per (exchange, token).
    Flag hours whose inflow exceeds  mean + sigma * std  AND  > min_inflow.

    Returns DataFrame: [hour, exchange, token, inflow, mean_inflow, z_score]
    sorted by z_score descending.
    """
    df = deposits.copy()
    df["hour"] = df["time"].dt.floor("h")

    hourly = (
        df.groupby(["hour", "exchange", "token"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "inflow"})
    )

    # Per-(exchange, token) statistics
    stats = (
        hourly.groupby(["exchange", "token"])["inflow"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    # std is NaN when only 1 data point — treat as 0 (any positive inflow = spike)
    stats["std"] = stats["std"].fillna(0).clip(lower=1.0)

    hourly = hourly.merge(stats, on=["exchange", "token"])
    hourly["z_score"] = (hourly["inflow"] - hourly["mean"]) / hourly["std"]

    spikes = hourly[
        (hourly["z_score"] >= sigma) & (hourly["inflow"] >= min_inflow)
    ].sort_values("z_score", ascending=False).reset_index(drop=True)

    return spikes[["hour", "exchange", "token", "inflow", "mean", "z_score"]]


# ── Whale-before-spike matching ───────────────────────────────────────────────

def match_whales_to_spikes(
    deposits: pd.DataFrame,
    spikes: pd.DataFrame,
    threshold: float = WHALE_THRESHOLD,
    lead_hours: int = LEAD_HOURS,
) -> list[dict]:
    """
    For each spike hour, find external whale deposits (non-CEX sender,
    amount >= threshold) that arrived in [spike_hour - lead_hours, spike_hour).

    Returns alert dicts sorted by spike inflow descending.
    Deduplicates on (from_address, to_address, time, amount).
    """
    alerts: list[dict] = []
    seen: set[tuple] = set()

    for _, spike in spikes.iterrows():
        window_end = spike["hour"]
        window_start = window_end - pd.Timedelta(hours=lead_hours)

        candidates = deposits[
            (deposits["time"] >= window_start)
            & (deposits["time"] < window_end)
            & (deposits["exchange"] == spike["exchange"])
            & (deposits["token"] == spike["token"])
            & (deposits["amount"] >= threshold)
            & (~deposits["from_is_cex"])   # exclude internal CEX moves
        ].sort_values("amount", ascending=False)

        for _, w in candidates.iterrows():
            key = (w["from_address"], w["to_address"], str(w["time"]), w["amount"])
            if key in seen:
                continue
            seen.add(key)

            lead_secs = (window_end - w["time"]).total_seconds()
            alerts.append(
                {
                    "from_address": w["from_address"],
                    "to_address": w["to_address"],
                    "amount": float(w["amount"]),
                    "token": w["token"],
                    "exchange": spike["exchange"],
                    "tx_time": w["time"],
                    "spike_hour": spike["hour"],
                    "spike_inflow": float(spike["inflow"]),
                    "lead_minutes": int(lead_secs / 60),
                    "z_score": float(spike["z_score"]),
                }
            )

    # Sort by spike magnitude so the most dramatic events come first
    alerts.sort(key=lambda x: x["spike_inflow"], reverse=True)
    return alerts


# ── Debug helper ─────────────────────────────────────────────────────────────

def print_hourly_debug(deposits: pd.DataFrame, spikes: pd.DataFrame) -> None:
    """Print full hourly inflow table per (exchange, token) with z-scores."""
    df = deposits.copy()
    df["hour"] = df["time"].dt.floor("h")
    hourly = (
        df.groupby(["hour", "exchange", "token"])["amount"]
        .sum().reset_index().rename(columns={"amount": "inflow"})
    )
    stats = (
        hourly.groupby(["exchange", "token"])["inflow"]
        .agg(mean="mean", std="std").reset_index()
    )
    stats["std"] = stats["std"].fillna(0).clip(lower=1.0)
    hourly = hourly.merge(stats, on=["exchange", "token"])
    hourly["z_score"] = (hourly["inflow"] - hourly["mean"]) / hourly["std"]

    spike_keys = set()
    for _, s in spikes.iterrows():
        spike_keys.add((s["hour"], s["exchange"], s["token"]))

    sep = "─" * 64
    for (exchange, token), grp in hourly.groupby(["exchange", "token"]):
        g_mean = grp["mean"].iloc[0]
        g_std = grp["std"].iloc[0]
        print(f"\n{sep}")
        print(f"  {exchange} / {token}   mean={_fmt_usd(g_mean)}/h   std={_fmt_usd(g_std)}")
        print(f"  {'Hour (UTC)':<16} {'Inflow':>14}  {'z-score':>8}")
        print(f"  {'─' * 14:<16} {'─' * 14:>14}  {'─' * 8:>8}")
        for _, row in grp.sort_values("hour").iterrows():
            flag = "  ◀ SPIKE" if (row["hour"], exchange, token) in spike_keys else ""
            print(
                f"  {row['hour'].strftime('%Y-%m-%d %H:%M'):<16} "
                f"{_fmt_usd(row['inflow']):>14}  "
                f"{row['z_score']:>+8.2f}"
                f"{flag}"
            )
    print(f"\n{sep}")


# ── Post formatting ───────────────────────────────────────────────────────────

def format_alert_post(alert: dict) -> str:
    """Render one micro-level whale alert in social post format."""
    amount_str = _fmt_usd(alert["amount"])
    spike_str = _fmt_usd(alert["spike_inflow"])
    token = alert["token"]
    exchange = alert["exchange"]
    addr = alert["from_address"]

    lead = alert["lead_minutes"]
    if lead < 60:
        timing = f"{lead}m"
    elif lead % 60 == 0:
        timing = f"{lead // 60}h"
    else:
        timing = f"{lead // 60}h {lead % 60:02d}m"

    return (
        f"A whale deposited {amount_str} ${token} to {exchange} "
        f"~{timing} before a {spike_str} inflow spike.\n"
        f"Wallet: {addr}\n"
        f"\n"
        f"Is smart money positioning early? 👀\n"
        f"\n"
        f"#WhaleAlert #{token} #{exchange} #OnChain #ChainFlowLens"
    )


# ── Fallback: top whale deposits (no spike correlation) ───────────────────────

def format_fallback_post(whale_row: pd.Series, exchange: str, label: str) -> str:
    """Used when no spike correlation is found — shows top external whale deposit."""
    amount_str = _fmt_usd(whale_row["amount"])
    token = whale_row["token"]
    addr = whale_row["from_address"]
    ts = whale_row["time"].strftime("%H:%M UTC")

    return (
        f"A whale deposited {amount_str} ${token} to {exchange} ({label}) at {ts}.\n"
        f"Wallet: {addr}\n"
        f"\n"
        f"Is smart money accumulating or de-risking? 👀\n"
        f"\n"
        f"#WhaleAlert #{token} #{exchange} #OnChain #ChainFlowLens"
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(date_str: str, debug: bool = False) -> None:
    sep = "─" * 64

    print(f"\n{sep}")
    print(f"  ChainFlow Lens · Micro Whale Alerts · {date_str}")
    print(sep)

    # 1. Load data
    print("\n[1/4] Loading data…")
    transfers, labels = load_data(date_str)
    print(f"      {len(transfers):,} transfers  |  {len(labels):,} CEX label entries")

    # 2. Tag CEX deposits
    print("[2/4] Tagging CEX deposits…")
    deposits = tag_cex_deposits(transfers, labels)
    external = deposits[~deposits["from_is_cex"]]
    print(
        f"      {len(transfers):,} total transfers  |  "
        f"{len(deposits):,} CEX inflows  |  "
        f"{len(external):,} from external wallets"
    )

    if deposits.empty:
        print("\n  No CEX deposits found. Nothing to analyse.")
        return

    # 3. Detect spike hours
    print("[3/4] Detecting inflow spike hours…")
    spikes = detect_spike_hours(deposits)

    if debug:
        print("\n  ── DEBUG: Hourly inflow breakdown ──")
        print_hourly_debug(deposits, spikes)

    if not spikes.empty:
        top = spikes.iloc[0]
        print(
            f"      {len(spikes)} spike window(s) found  "
            f"(largest: {_fmt_usd(top['inflow'])} {top['token']} "
            f"→ {top['exchange']} @ {top['hour'].strftime('%H:%M UTC')}, "
            f"z={top['z_score']:.1f})"
        )
    else:
        print(
            f"      No spikes above σ={SPIKE_SIGMA} / ${MIN_SPIKE_USD/1e6:.0f}M threshold."
        )

    # 4. Match whales to spikes
    print("[4/4] Matching whale deposits to spike windows…")
    alerts = match_whales_to_spikes(deposits, spikes)
    print(f"      {len(alerts)} whale-before-spike event(s) found\n")

    posts: list[str] = []

    if alerts:
        for i, alert in enumerate(alerts[:MAX_ALERTS], 1):
            post = format_alert_post(alert)
            posts.append(post)
            print(sep)
            print(
                f"  Alert {i}/{min(len(alerts), MAX_ALERTS)}  "
                f"z={alert['z_score']:.1f}  "
                f"spike @ {alert['spike_hour'].strftime('%H:%M UTC')}  "
                f"lead={alert['lead_minutes']}m"
            )
            print()
            print(post)
    else:
        # Fallback: show top external whale deposits even without a spike match
        print(
            "  No spike correlations found — showing top external whale deposits.\n"
        )
        top_whales = (
            external[external["amount"] >= WHALE_THRESHOLD]
            .sort_values("amount", ascending=False)
            .head(MAX_ALERTS)
        )
        if top_whales.empty:
            print(
                f"  No external whale deposits above "
                f"${WHALE_THRESHOLD/1e6:.1f}M found."
            )
            return
        addr_map = labels.set_index("address")[["exchange", "label"]].to_dict("index")
        for i, (_, row) in enumerate(top_whales.iterrows(), 1):
            info = addr_map.get(row["to_address"], {"exchange": "CEX", "label": "?"})
            post = format_fallback_post(row, info["exchange"], info["label"])
            posts.append(post)
            print(sep)
            print(f"  Top whale {i}/{len(top_whales)}")
            print()
            print(post)

    # Save to file
    print(f"\n{sep}")
    out_dir = OUTPUT_DIR / date_str / "posts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "micro_alerts.txt"

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, post in enumerate(posts, 1):
            fh.write(f"=== MICRO ALERT {i} ===\n")
            fh.write(post)
            fh.write("\n\n")

    print(f"  Saved {len(posts)} post(s)  →  {out_path}\n")


def _latest_date() -> str:
    """Return the most recent YYYY-MM-DD folder under DATA_DIR."""
    candidates = sorted(
        [
            d.name
            for d in DATA_DIR.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name != "__pycache__"
        ],
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No data folders found under {DATA_DIR}/")
    return candidates[0]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    debug_flag = "--debug" in sys.argv or "-d" in sys.argv
    date_arg = args[0] if args else _latest_date()
    run(date_arg, debug=debug_flag)
