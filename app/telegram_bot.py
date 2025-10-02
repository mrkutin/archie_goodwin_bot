from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram import Router
from aiogram.client.default import DefaultBotProperties


router = Router()


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer("Hello! I'm alive. Send me any message to echo it back.")


@router.message(F.text)
async def handle_echo(message: Message) -> None:
    text = message.text or ""
    if not text.strip():
        return
    await message.answer(text)


async def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def run_bot(token: str) -> None:
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = await create_dispatcher()

    # Start polling
    await dp.start_polling(bot)
