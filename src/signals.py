"""
src/signals.py — On-chain signal extraction.

flow_signal   → total stablecoin CEX inflow / outflow / net for a transfer window.
whale_signal  → individual transfers above a USD threshold.
"""

import pandas as pd


def flow_signal(transfers: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """
    Compute total stablecoin inflow to / outflow from all tracked CEX addresses.

    Args:
        transfers: DataFrame with columns [from_address, to_address, amount, token].
        labels:    DataFrame with columns [address, exchange].

    Returns:
        {
            "total_inflow":  float,
            "total_outflow": float,
            "net_flow":      float,
            "by_token": {
                "USDC": {"inflow": float, "outflow": float, "netflow": float},
                "USDT": {...},
            },
        }
    """
    cex_addrs: set[str] = set(labels["address"].str.lower())

    inflows = transfers[transfers["to_address"].isin(cex_addrs)]
    outflows = transfers[transfers["from_address"].isin(cex_addrs)]

    total_inflow = float(inflows["amount"].sum())
    total_outflow = float(outflows["amount"].sum())

    by_token: dict[str, dict] = {}
    for token in sorted(transfers["token"].unique()):
        ti = float(inflows[inflows["token"] == token]["amount"].sum())
        to_ = float(outflows[outflows["token"] == token]["amount"].sum())
        by_token[token] = {
            "inflow": round(ti, 2),
            "outflow": round(to_, 2),
            "netflow": round(ti - to_, 2),
        }

    return {
        "total_inflow": round(total_inflow, 2),
        "total_outflow": round(total_outflow, 2),
        "net_flow": round(total_inflow - total_outflow, 2),
        "by_token": by_token,
    }


def whale_signal(
    transfers: pd.DataFrame, threshold: float = 1_000_000
) -> pd.DataFrame:
    """
    Filter transfers with a single-transaction amount >= `threshold` USD.

    Args:
        transfers: DataFrame with columns [time, from_address, to_address, amount, token].
        threshold: Minimum USD amount to qualify as a whale transfer (default $1 M).

    Returns:
        DataFrame with columns [time, from_address, to_address, amount, token],
        sorted by amount descending.
    """
    whales = (
        transfers[transfers["amount"] >= threshold]
        .copy()
        .sort_values("amount", ascending=False)
        .reset_index(drop=True)
    )
    return whales[["time", "from_address", "to_address", "amount", "token"]]
