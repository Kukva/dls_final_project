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
TOKEN = "7898819934:AAFmy8Cp2vpXL00A5xqr3y0FhN1GKwIY3Tw"
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

# # Функция сохранения GIF от пользователя
# async def get_gif_frames_from_message(message: Message) -> List[Image.Image]:
#     path = f"images/{message.document.file_id}.gif"
#     await bot.download(message.document, destination=path)
    
#     gif = Image.open(path)
#     frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]
#     return frames

# Функция обработки и отправки результата
async def send_img_and_clear_output(message: Message, path_input: str, path_output: str):
    await message.answer_photo(photo=FSInputFile(path_output))
    os.remove(path_input)
    os.remove(path_output)

# # Функция обработки GIF
# def process_gif(frames: List[Image.Image]) -> str:
#     processed_frames = []
#     for frame in frames:
#         style_transfer_model.set_content_img(frame)
#         output_img: Tensor = style_transfer_model.fit()
#         processed_frames.append(transforms.ToPILImage()(output_img))
    
#     output_gif_path = "images/output.gif"
#     processed_frames[0].save(output_gif_path, save_all=True, append_images=processed_frames[1:], duration=100, loop=0)
#     return output_gif_path

# Обработчик команд
@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer(MESSAGE.WELCOME_MESSAGE.value)

@dp.message(Command("help"))
async def help_message(message: Message):
    await message.answer(MESSAGE.HELP_MESSAGE.value)

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

        # Запускаем перенос стиля
        await message.answer(MESSAGE.PLEASE_WAIT.value)

        content_path = user_images[chat_id]["content"]
        style_path = user_images[chat_id]["style"]
        output_path = f"images/result/{chat_id}_output.jpg"

        content_img = image_loader(content_path)
        style_img = image_loader(style_path)
        input_img = content_img.clone()

        # style_transfer_model.set_content_img(content_path)
        # style_transfer_model.set_style_img(style_path)
        output_img = style_transfer_model.run_style_transfer(
            content_img=content_img,  # Открываем контентное изображение
            style_img=style_img,      # Открываем стиль
            input_img=input_img     # Начальное изображение
        )
        save_image(output_img, output_path)
        print("Завершена обработка")
        # Отправляем результат
        await message.answer_photo(photo=FSInputFile(output_path))

        # Удаляем файлы
        os.remove(content_path)
        os.remove(style_path)
        os.remove(output_path)

        # Очищаем данные пользователя
        del user_images[chat_id]

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
