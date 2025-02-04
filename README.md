# dls_final_project

@dls_project_style_transfer_bot

Доступные команды:

    /start - начать работу
    /help - краткая сводка о боте
    /transfer - перейти к обработке после сохранения картинок
    /clear - очистить историю картинок


## Использование

1. Отправьте команду `/start`, чтобы начать работу с ботом.
2. Бот попросит вас отправить два изображения:
    - Первое изображение — это "контент" (например, ваше изображение, к которому вы хотите применить стиль).
    - Второе изображение — это "стиль" (например, изображение, чей стиль вы хотите перенести).
3. Бот применит стиль к контенту и отправит вам результат.
4. Бот также может отправить промежуточные изображения, показывающие процесс обучения модели.

## Примечание
- Бот использует модель **Neural Style Transfer**, которая применяет стиль одного изображения (например, картины художника) к другому изображению (например, фотографии).
- Для работы модели требуется наличие **GPU**. Вы можете протестировать модель в **Google Colab** с использованием GPU.
- Изображения сохраняются в локальной папке images

