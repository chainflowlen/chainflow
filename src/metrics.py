"""
src/metrics.py — CEX netflow calculations.

calculate_netflow  → per-exchange, per-token inflow / outflow / netflow summary.
daily_netflow      → aggregate CEX netflow grouped by day (used for 7-day chart).
"""

import pandas as pd


def calculate_netflow(transfers: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CEX inflow, outflow, and netflow per exchange per token.

    Inflow  = transfers whose to_address is a labelled CEX address.
    Outflow = transfers whose from_address is a labelled CEX address.
    Intra-exchange transfers (e.g. Binance hot-wallet to hot-wallet) cancel out
    naturally because they contribute equally to inflow and outflow.

    Args:
        transfers: DataFrame with columns [from_address, to_address, amount, token].
        labels:    DataFrame with columns [address, exchange].

    Returns:
        DataFrame with columns [exchange, token, inflow, outflow, netflow].
    """
    if transfers.empty:
        return pd.DataFrame(columns=["exchange", "token", "inflow", "outflow", "netflow"])

    addr_map: dict[str, str] = dict(
        zip(labels["address"].str.lower(), labels["exchange"])
    )

    inflows = transfers[transfers["to_address"].isin(addr_map)].copy()
    inflows["exchange"] = inflows["to_address"].map(addr_map)
    inflow_agg = (
        inflows.groupby(["exchange", "token"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "inflow"})
    )

    outflows = transfers[transfers["from_address"].isin(addr_map)].copy()
    outflows["exchange"] = outflows["from_address"].map(addr_map)
    outflow_agg = (
        outflows.groupby(["exchange", "token"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "outflow"})
    )

    result = pd.merge(inflow_agg, outflow_agg, on=["exchange", "token"], how="outer").fillna(0)
    result["netflow"] = result["inflow"] - result["outflow"]
    return result.sort_values(["exchange", "token"]).reset_index(drop=True)


def daily_netflow(transfers: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total CEX netflow by calendar day and token.

    Args:
        transfers: DataFrame with columns [time, from_address, to_address, amount, token].
        labels:    DataFrame with columns [address, exchange].

    Returns:
        DataFrame with columns [date, token, inflow, outflow, netflow], sorted by date.
    """
    if transfers.empty:
        return pd.DataFrame(columns=["date", "token", "inflow", "outflow", "netflow"])

    cex_addrs: set[str] = set(labels["address"].str.lower())

    df = transfers.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["date"] = df["time"].dt.date

    inflows = df[df["to_address"].isin(cex_addrs)]
    outflows = df[df["from_address"].isin(cex_addrs)]

    in_agg = (
        inflows.groupby(["date", "token"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "inflow"})
    )
    out_agg = (
        outflows.groupby(["date", "token"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "outflow"})
    )

    result = pd.merge(in_agg, out_agg, on=["date", "token"], how="outer").fillna(0)
    result["netflow"] = result["inflow"] - result["outflow"]
    return result.sort_values(["date", "token"]).reset_index(drop=True)
