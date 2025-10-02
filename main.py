import asyncio
from app.config import get_telegram_token
from app.telegram_bot import run_bot


def main() -> None:
    token = get_telegram_token()
    asyncio.run(run_bot(token))


if __name__ == "__main__":
    main()
