"""
src/content.py — Post template generators.

Three templates:
  stablecoin_flow_post  → Template 1: 24h stablecoin flow summary.
  whale_alert_post      → Template 2: large single-transfer alert.
  netflow_chart_post    → Template 3: 7-day chart + caption.
"""

import pandas as pd
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def stablecoin_flow_post(signal: dict, timestamp: str | None = None) -> str:
    """
    Template 1 — Stablecoin Flow summary.

    Args:
        signal:    Output dict from signals.flow_signal().
        timestamp: Optional override; defaults to current UTC time.
    """
    ts = timestamp or _utc_now()

    usdc = signal["by_token"].get("USDC", {"inflow": 0, "outflow": 0, "netflow": 0})
    usdt = signal["by_token"].get("USDT", {"inflow": 0, "outflow": 0, "netflow": 0})

    direction = "flowing INTO exchanges 🟢" if signal["net_flow"] >= 0 else "flowing OUT OF exchanges 🔴"

    return (
        f"📊 Stablecoin CEX Flow  |  {ts}\n"
        f"\n"
        f"Total inflow   →  ${signal['total_inflow']:>15,.0f}\n"
        f"Total outflow  ←  ${signal['total_outflow']:>15,.0f}\n"
        f"Net flow          ${signal['net_flow']:>+15,.0f}\n"
        f"\n"
        f"  USDC  net: ${usdc['netflow']:>+12,.0f}\n"
        f"  USDT  net: ${usdt['netflow']:>+12,.0f}\n"
        f"\n"
        f"Stablecoins are {direction}.\n"
        f"\n"
        f"#Crypto #Stablecoins #OnChain #ChainFlowLens"
    )


def whale_alert_post(whale_df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Template 2 — Whale Alert.

    Args:
        whale_df: Output DataFrame from signals.whale_signal().
        max_rows: Maximum number of transactions to list in the post.
    """
    if whale_df.empty:
        return "🐳 No whale transfers (>$1M) detected in the past 24 h.\n\n#Crypto #OnChain #ChainFlowLens"

    lines = [f"🐳 Whale Alert  |  Single transfers >$1,000,000\n"]

    for _, row in whale_df.head(max_rows).iterrows():
        frm = f"{row['from_address'][:6]}…{row['from_address'][-4:]}"
        to_ = f"{row['to_address'][:6]}…{row['to_address'][-4:]}"
        lines.append(f"  {frm} → {to_}   ${row['amount']:>14,.0f}  {row['token']}")

    if len(whale_df) > max_rows:
        lines.append(f"  … and {len(whale_df) - max_rows} more transfer(s)")

    lines.append("\n#WhaleAlert #Crypto #OnChain #ChainFlowLens")
    return "\n".join(lines)


def netflow_chart_post(
    chart_path: str, signal: dict, timestamp: str | None = None
) -> str:
    """
    Template 3 — 7-day Netflow Chart + caption.

    Args:
        chart_path: Filesystem path (or URL) to the saved chart PNG.
        signal:     Output dict from signals.flow_signal() (used for the caption).
        timestamp:  Optional override; defaults to current UTC date string.
    """
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    net = signal["net_flow"]
    direction = "net inflow 🟢" if net >= 0 else "net outflow 🔴"

    return (
        f"📈 7-Day Stablecoin CEX Netflow  |  {ts}\n"
        f"\n"
        f"Chart → {chart_path}\n"
        f"\n"
        f"Over the past 7 days, stablecoins recorded a {direction}\n"
        f"of ${abs(net):,.0f} across tracked exchanges.\n"
        f"\n"
        f"Exchanges: Binance · Coinbase · OKX\n"
        f"Tokens:    USDC · USDT\n"
        f"\n"
        f"#Crypto #Stablecoins #OnChain #ChainFlowLens"
    )
