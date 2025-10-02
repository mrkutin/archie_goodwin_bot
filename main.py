import os
import asyncio
from app.config import get_telegram_token
from app.telegram_bot import run_bot

# Disable HuggingFace tokenizers parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    token = get_telegram_token()
    asyncio.run(run_bot(token))


if __name__ == "__main__":
    main()
