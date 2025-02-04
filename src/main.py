import os
from pathlib import Path
from typing import List
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram import F
from torchvision import transforms
from torch import Tensor
from PIL import Image, ImageSequence
from models.nts import NeuralStyleTransfer
from models.utils import save_image, image_loader
from misc.message import MESSAGE
from typing import Dict
# from dotenv import load_dotenv

# Инициализация бота и модели
TOKEN = os.getenv('BOT_TOKEN')
bot = Bot(token=TOKEN)
dp = Dispatcher()
style_transfer_model = NeuralStyleTransfer()

user_images: Dict[int, Dict[str, str]] = {}

# Функция сохранения изображения от пользователя
async def save_photo(message: Message) -> str:
    """Сохраняет изображение и возвращает путь к нему."""
    photo = message.photo[-1]
    path = f"images/source/{photo.file_id}.jpg"
    await bot.download(photo, destination=path)
    return path

# Обработчик команд
@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer(MESSAGE.WELCOME_MESSAGE.value)

@dp.message(Command("help"))
async def help_message(message: Message):
    await message.answer(MESSAGE.HELP_MESSAGE.value)


@dp.message(Command("transfer"))
async def transfer_style(message: Message):
    chat_id = message.chat.id

    if chat_id not in user_images or "content" not in user_images[chat_id] or "style" not in user_images[chat_id]:
        await message.answer("Сначала отправьте два изображения: одно для контента, второе для стиля.")
        return

    await message.answer("Обрабатываю изображение...")

    content_path = user_images[chat_id]["content"]
    style_path = user_images[chat_id]["style"]
    output_path = f"images/{chat_id}_output.jpg"
    gif_output_path = f"images/{chat_id}_output.gif"

    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    # Запуск переноса стиля
    output_img, intermediate_images = style_transfer_model.run_style_transfer(
        content_img=content_img,
        style_img=style_img,
        input_img=input_img,
        save_steps=True
    )

    # Сохранение результата
    save_image(output_img, output_path)

    # Генерация GIF
    if intermediate_images:
        intermediate_images[0].save(
            gif_output_path,
            save_all=True,
            append_images=intermediate_images[1:],
            duration=100,
            loop=0
        )

    # Отправка результатов
    await message.answer_photo(photo=FSInputFile(output_path))
    await message.answer_document(document=FSInputFile(gif_output_path))

    await message.answer("Обработка завершена! Вы можете снова ввести /transfer для повторной обработки или /clear для очистки изображений.")


# Обработчик изображений
@dp.message(F.photo)
async def handle_image(message: Message):
    chat_id = message.chat.id

    if chat_id not in user_images:
        user_images[chat_id] = {}

    if "content" not in user_images[chat_id]:
        # Сохраняем первое изображение (контент)
        user_images[chat_id]["content"] = await save_photo(message)
        await message.answer(MESSAGE.FIRST_PICTURE_UPLOADED.value)
    elif "style" not in user_images[chat_id]:
        # Сохраняем второе изображение (стиль)
        user_images[chat_id]["style"] = await save_photo(message)
        await message.answer("Изображения сохранены! Введите команду /transfer для переноса стиля.")

    else:
        await message.answer("У вас уже загружены изображения. Введите /transfer для обработки или /clear для очистки.")


@dp.message(Command("clear"))
async def clear_images(message: Message):
    chat_id = message.chat.id

    if chat_id in user_images:
        content_path = user_images[chat_id].get("content")
        style_path = user_images[chat_id].get("style")

        if content_path and os.path.exists(content_path):
            os.remove(content_path)
        if style_path and os.path.exists(style_path):
            os.remove(style_path)

        del user_images[chat_id]

        await message.answer("Ваши изображения удалены. Отправьте новые для обработки.")
    else:
        await message.answer("У вас нет сохраненных изображений.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
