from enum import Enum

# Сообщения от бота
class MESSAGE(Enum):
    WELCOME_MESSAGE = "Привет! Я — бот, который поможет перенести стиль с одного изображения на другое.\n" \
                  "Отправьте мне два изображения: сначала то, на которое вы хотите наложить стиль, " \
                  "а затем — изображение, с которого будет взят стиль."

    HELP_MESSAGE = "Чтобы получить результат, отправьте мне два изображения. " \
                "Сначала отправьте фото, на которое будет наложен стиль, а затем — фото, с которого " \
                "будет взят стиль. Я все сделаю!"

    FIRST_PICTURE_UPLOADED = "Отлично! Вы загрузили изображение, на которое я перенесу стиль. " \
                            "Теперь отправьте изображение, с которого будет взят стиль."

    PLEASE_WAIT = "Спасибо за терпение! Преобразование может занять несколько минут. Пожалуйста, подождите..."

    PHOTO_TOO_LARGE = "Размер изображения слишком большой. Попробуйте отправить его в другом формате или уменьшите размер."

    RESULT = "Ваше изображение готово! Посмотрите результат."