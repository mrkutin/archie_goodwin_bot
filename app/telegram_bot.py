from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram import Router
from aiogram.client.default import DefaultBotProperties

from app.agent import answer_question


router = Router()


_MAX_TG_MESSAGE_LEN = 4000


def _split_for_telegram(text: str, max_len: int = _MAX_TG_MESSAGE_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        # Try to break on a paragraph or line boundary for nicer splits
        chunk = text[start:end]
        last_break = max(chunk.rfind("\n\n"), chunk.rfind("\n"))
        if last_break != -1 and (start + last_break) > start + max_len // 2:
            end = start + last_break
            chunk = text[start:end]
        parts.append(chunk)
        start = end
    return parts


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer("Hello! I'm your legal research assistant. Ask a question to begin.")


@router.message(F.text)
async def handle_user_message(message: Message) -> None:
    text = message.text or ""
    if not text.strip():
        return
    await message.chat.do("typing")
    try:
        response_text = answer_question(text, thread_id=str(message.chat.id))
    except Exception as exc:
        response_text = (
            "Sorry, I couldn't process that right now. "
            f"Error: {exc}"
        )
    for chunk in _split_for_telegram(response_text):
        await message.answer(chunk)


async def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def run_bot(token: str) -> None:
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = await create_dispatcher()

    # Start polling
    await dp.start_polling(bot)
