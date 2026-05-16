import os
from dotenv import load_dotenv

load_dotenv()

ALCHEMY_API_KEY: str = os.getenv("ALCHEMY_API_KEY", "")
ALCHEMY_URL: str = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

# ERC-20 contract addresses (Ethereum mainnet)
USDC_CONTRACT: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_CONTRACT: str = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

# Signal thresholds
WHALE_THRESHOLD: float = 1_000_000  # USD

# Filesystem
DATA_DIR: str = "data"
OUTPUT_DIR: str = "output"
