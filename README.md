# ChainFlow Lens

On-chain stablecoin flow monitor for USDC and USDT on Ethereum mainnet.
Tracks CEX inflows/outflows, whale alerts, and 7-day netflow charts.

---

## Project structure

```
chainflow/
├── main.py               ← Full pipeline entry point
├── config.py             ← API keys, contract addresses, constants
├── requirements.txt
├── .env.example
├── data/
│   ├── labels.csv        ← CEX address labels (Binance, Coinbase, OKX)
│   └── transfers.csv     ← Generated: raw 24h transfer data
├── src/
│   ├── fetch.py          ← Alchemy API data acquisition
│   ├── metrics.py        ← CEX netflow calculations
│   ├── signals.py        ← Flow signal & whale filter
│   ├── chart.py          ← 7-day netflow line chart
│   └── content.py        ← Post template generators
└── output/
    ├── netflow_24h.csv
    ├── whale_alerts.csv
    ├── netflow_7d.png
    └── posts/
        ├── stablecoin_flow.txt
        ├── whale_alert.txt
        └── netflow_chart.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Alchemy API key
cp .env.example .env
# edit .env and fill in ALCHEMY_API_KEY

# 3. Run the full pipeline
python main.py
```

Get a free Alchemy API key at https://www.alchemy.com.

---

## What it produces

| Output | Description |
|---|---|
| `data/transfers.csv` | Raw 24h USDC/USDT transfers (time, from, to, amount, token) |
| `output/netflow_24h.csv` | CEX inflow/outflow/netflow per exchange per token |
| `output/whale_alerts.csv` | Transfers > $1 M in the 24h window |
| `output/netflow_7d.png` | Line chart: 7-day daily CEX netflow (black bg, blue/green) |
| `output/posts/*.txt` | Three ready-to-publish post templates |

---

## Signals

### Flow signal
Total stablecoin inflow to tracked exchanges in the past 24 h, broken down by USDC and USDT.

### Whale signal
Any single transfer ≥ $1 M — outputs `from / to / amount / token` sorted by size.

---

## CEX labels (`data/labels.csv`)

Pre-seeded with 5 addresses each for **Binance**, **Coinbase**, and **OKX**.
Add or update rows freely — the pipeline re-reads the file on every run.

```
address,exchange,label
0x3f5ce5…,Binance,Binance Hot Wallet 7
…
```

---

## Chart spec

- 7-day daily netflow, one line per token (USDC = blue, USDT = green)
- Black background (`#0D0D0D`), no grid lines
- Dashed zero reference line
- Y-axis labelled in millions (e.g. `$12.3M`)

---

## Scope

1 metric (CEX netflow) · 1 whale filter ($1M threshold) · 1 chart type (line). No additions.
