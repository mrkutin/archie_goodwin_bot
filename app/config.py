import os
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def load_environment() -> None:
    if load_dotenv is not None:
        # Load .env if present
        load_dotenv()  # type: ignore


def get_telegram_token() -> str:
    load_environment()
    token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Provide it in environment or .env file."
        )
    return token
