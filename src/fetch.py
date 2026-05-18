"""
src/fetch.py — Alchemy API data acquisition.

Provides three entry points:
  fetch_transfers_for_date(api_key, target_date)
      → All USDC/USDT transfers for a specific calendar day (UTC), using
        binary-search block lookup for precise boundaries.

  fetch_transfers(api_key, hours=24)
      → All USDC/USDT transfers over the past N hours (rolling window).

  fetch_cex_flows(api_key, cex_addresses, days=7)
      → USDC/USDT transfers to/from specific CEX addresses (for 7-day chart).
        Makes one request per (token, address, direction), then deduplicates.
"""

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import pandas as pd

USDC_CONTRACT = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_CONTRACT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
BLOCKS_PER_HOUR = 300   # Ethereum ~12 s/block → ~300 blocks/hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NO_PROXY = {"http": None, "https": None}  # bypass any system proxy


def _get_current_block(url: str) -> int:
    resp = requests.post(
        url,
        json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
        timeout=30,
        proxies=_NO_PROXY,
    )
    resp.raise_for_status()
    return int(resp.json()["result"], 16)


_RATE_LIMIT_DELAY = 0.05   # 20 req/s, well under the 25 req/s free-tier cap


def _get_block_at_timestamp(url: str, target_ts: int) -> int:
    """Binary-search for the first block whose timestamp >= target_ts (Unix seconds)."""
    lo = 1
    hi = _get_current_block(url)
    while lo < hi:
        mid = (lo + hi) // 2
        resp = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByNumber",
                "params": [hex(mid), False],  # False = header only, no full tx list
            },
            timeout=30,
            proxies=_NO_PROXY,
        )
        resp.raise_for_status()
        block_ts = int(resp.json()["result"]["timestamp"], 16)
        if block_ts < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


_MAX_RETRIES = 5          # max attempts per page on network error
_RETRY_BACKOFF = 2.0      # seconds; doubled on each failure


def _post_with_retry(url: str, payload: dict) -> dict:
    """POST with exponential-backoff retry; returns parsed JSON result dict."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=60, proxies=_NO_PROXY)
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as exc:
            last_exc = exc
            wait = _RETRY_BACKOFF * (2 ** attempt)
            print(f"  [retry {attempt+1}/{_MAX_RETRIES}] network error, retrying in {wait:.0f}s… ({exc})")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _paginate_transfers(
    url: str,
    params: dict,
    token: str,
    csv_path: Path | None = None,
    checkpoint: dict | None = None,
) -> list[dict]:
    """
    Paginate alchemy_getAssetTransfers.

    If csv_path is given, each page is appended to the CSV immediately
    (streaming mode) and checkpoint is updated so runs can be resumed.
    Returns all records only in non-streaming mode; in streaming mode returns [].
    """
    records: list[dict] = []
    streaming = csv_path is not None

    # Resume: pick up from last saved page_key for this token
    page_key: str | None = None
    if streaming and checkpoint and checkpoint.get(token, {}).get("page_key"):
        page_key = checkpoint[token]["page_key"]
        print(f"  Resuming {token} from saved page_key…")

    page_num = 0
    while True:
        req_params = {**params}
        if page_key:
            req_params["pageKey"] = page_key

        result = _post_with_retry(url, {
            "jsonrpc": "2.0", "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [req_params],
        })
        time.sleep(_RATE_LIMIT_DELAY)

        transfers = result.get("transfers", [])
        page_key = result.get("pageKey")
        page_num += 1

        if streaming:
            # Append this page to CSV immediately
            page_df = _to_df(transfers, token)
            write_header = not csv_path.exists()
            page_df.to_csv(csv_path, mode="a", header=write_header, index=False)
            # Update checkpoint
            if checkpoint is not None:
                checkpoint.setdefault(token, {})["page_key"] = page_key or ""
                checkpoint[token]["pages"] = checkpoint[token].get("pages", 0) + 1
                ckpt_path = csv_path.with_suffix(".checkpoint.json")
                ckpt_path.write_text(json.dumps(checkpoint))
            if page_num % 50 == 0:
                print(f"  {token}: {page_num} pages written so far…")
        else:
            records.extend(transfers)

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

def fetch_transfers_for_date(
    api_key: str,
    target_date: date,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Fetch all USDC and USDT ERC-20 transfers for an exact calendar day (UTC).

    If csv_path is provided, data is streamed page-by-page into the CSV file
    and a checkpoint file (<csv_path>.checkpoint.json) is maintained so an
    interrupted run can be resumed automatically on next call.
    """
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"

    csv_path = Path(csv_path) if csv_path else None
    ckpt_path = csv_path.with_suffix(".checkpoint.json") if csv_path else None

    # Load or initialise checkpoint
    checkpoint: dict = {}
    if ckpt_path and ckpt_path.exists():
        checkpoint = json.loads(ckpt_path.read_text())

    start_ts = int(datetime(target_date.year, target_date.month, target_date.day,
                            tzinfo=timezone.utc).timestamp())
    end_ts = start_ts + 86400

    # Reuse block range from checkpoint if available (avoids ~48 extra RPC calls)
    if checkpoint.get("from_block") and checkpoint.get("to_block"):
        from_block = checkpoint["from_block"]
        to_block   = checkpoint["to_block"]
        print(f"  Block range from checkpoint: {from_block:,} → {to_block:,}")
    else:
        print(f"  Binary-searching start block (target: {target_date} 00:00 UTC)…")
        from_block = _get_block_at_timestamp(url, start_ts)
        print(f"  Binary-searching end block   (target: {target_date} 24:00 UTC)…")
        to_block = _get_block_at_timestamp(url, end_ts) - 1
        print(f"  Block range: {from_block:,} → {to_block:,}  ({to_block - from_block + 1:,} blocks)")
        checkpoint["from_block"] = from_block
        checkpoint["to_block"]   = to_block
        if ckpt_path:
            ckpt_path.write_text(json.dumps(checkpoint))

    for token, contract in [("USDC", USDC_CONTRACT), ("USDT", USDT_CONTRACT)]:
        # Skip tokens already fully fetched in a previous run
        if checkpoint.get(token, {}).get("done"):
            print(f"  {token}: already complete (from checkpoint), skipping.")
            continue

        params = {
            "fromBlock": hex(from_block),
            "toBlock":   hex(to_block),
            "contractAddresses": [contract],
            "category": ["erc20"],
            "withMetadata": True,
            "excludeZeroValue": True,
            "maxCount": "0x3e8",
        }
        _paginate_transfers(url, params, token=token,
                            csv_path=csv_path, checkpoint=checkpoint)

        # Mark token as fully done in checkpoint
        checkpoint.setdefault(token, {})["done"] = True
        checkpoint[token]["page_key"] = ""
        if ckpt_path:
            ckpt_path.write_text(json.dumps(checkpoint))
        total = checkpoint.get(token, {}).get("pages", "?")
        print(f"  {token}: done ({total} pages written)")

    # Read final CSV (streaming) or concat in-memory frames
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset=["hash", "token", "from_address", "to_address"])
        df.to_csv(csv_path, index=False)          # overwrite deduped
        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink()                    # clean up checkpoint
        return df

    return pd.DataFrame(
        columns=["hash", "time", "from_address", "to_address", "amount", "token"]
    )


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
