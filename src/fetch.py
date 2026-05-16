"""
src/fetch.py — Alchemy API data acquisition.

Provides two entry points:
  fetch_transfers(api_key, hours=24)
      → All USDC/USDT transfers over the past N hours (for 24h signals).

  fetch_cex_flows(api_key, cex_addresses, days=7)
      → USDC/USDT transfers to/from specific CEX addresses (for 7-day chart).
        Makes one request per (token, address, direction), then deduplicates.
"""

import time

import requests
import pandas as pd

USDC_CONTRACT = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_CONTRACT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
BLOCKS_PER_HOUR = 300   # Ethereum ~12 s/block → ~300 blocks/hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_current_block(url: str) -> int:
    resp = requests.post(
        url,
        json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
        timeout=30,
    )
    resp.raise_for_status()
    return int(resp.json()["result"], 16)


_RATE_LIMIT_DELAY = 0.05   # 20 req/s, well under the 25 req/s free-tier cap


def _paginate_transfers(url: str, params: dict) -> list[dict]:
    """Call alchemy_getAssetTransfers with automatic pagination."""
    records: list[dict] = []
    page_key: str | None = None

    while True:
        if page_key:
            params = {**params, "pageKey": page_key}

        resp = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getAssetTransfers",
                "params": [params],
            },
            timeout=30,
        )
        resp.raise_for_status()
        time.sleep(_RATE_LIMIT_DELAY)  # respect 25 req/s rate limit

        result = resp.json().get("result", {})
        records.extend(result.get("transfers", []))
        page_key = result.get("pageKey")

        if not page_key:
            break

    return records


def _to_df(raw: list[dict], token: str) -> pd.DataFrame:
    rows = [
        {
            "hash": t.get("hash", ""),
            "time": t.get("metadata", {}).get("blockTimestamp", ""),
            "from_address": (t.get("from") or "").lower(),
            "to_address": (t.get("to") or "").lower(),
            "amount": float(t.get("value") or 0),
            "token": token,
        }
        for t in raw
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["hash", "time", "from_address", "to_address", "amount", "token"]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_transfers(api_key: str, hours: int = 24) -> pd.DataFrame:
    """
    Fetch all USDC and USDT ERC-20 transfers for the past `hours` hours.

    Saved columns: hash, time, from_address, to_address, amount, token
    """
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    current_block = _get_current_block(url)
    from_block = hex(max(0, current_block - BLOCKS_PER_HOUR * hours))

    frames: list[pd.DataFrame] = []

    for token, contract in [("USDC", USDC_CONTRACT), ("USDT", USDT_CONTRACT)]:
        params = {
            "fromBlock": from_block,
            "toBlock": "latest",
            "contractAddresses": [contract],
            "category": ["erc20"],
            "withMetadata": True,
            "excludeZeroValue": True,
            "maxCount": "0x3e8",  # 1 000 per page
        }
        raw = _paginate_transfers(url, params)
        frames.append(_to_df(raw, token))
        print(f"  {token}: fetched {len(raw):,} transfers")

    if not frames:
        return pd.DataFrame(
            columns=["hash", "time", "from_address", "to_address", "amount", "token"]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["hash", "token", "from_address", "to_address"])
    return df


def fetch_cex_flows(
    api_key: str, cex_addresses: list[str], days: int = 7
) -> pd.DataFrame:
    """
    Fetch USDC/USDT transfers to and from known CEX addresses for the past `days` days.

    Uses targeted address filtering (one request per address + direction) to keep
    data volume manageable. Records are deduplicated by (hash, token) before return.
    """
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    current_block = _get_current_block(url)
    from_block = hex(max(0, current_block - BLOCKS_PER_HOUR * 24 * days))

    frames: list[pd.DataFrame] = []
    total_calls = len(cex_addresses) * 2 * 2   # addresses × tokens × directions
    call_n = 0

    for token, contract in [("USDC", USDC_CONTRACT), ("USDT", USDT_CONTRACT)]:
        for addr in cex_addresses:
            for direction_key in ("toAddress", "fromAddress"):
                call_n += 1
                print(
                    f"  [{call_n}/{total_calls}] {token} {'→' if direction_key == 'toAddress' else '←'} {addr[:10]}…"
                )
                params = {
                    "fromBlock": from_block,
                    "toBlock": "latest",
                    "contractAddresses": [contract],
                    direction_key: addr.lower(),
                    "category": ["erc20"],
                    "withMetadata": True,
                    "excludeZeroValue": True,
                    "maxCount": "0x3e8",
                }
                raw = _paginate_transfers(url, params)
                frames.append(_to_df(raw, token))

    if not frames:
        return pd.DataFrame(
            columns=["hash", "time", "from_address", "to_address", "amount", "token"]
        )

    df = pd.concat(frames, ignore_index=True)
    # Deduplicate: same on-chain transfer fetched from multiple directions
    df = df.drop_duplicates(subset=["hash", "token", "from_address", "to_address"])
    return df
