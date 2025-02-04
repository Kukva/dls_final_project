# Используем официальный Python образ
FROM python:3.9-slim

# Установим рабочую директорию
WORKDIR /app

# Копируем все файлы из текущей директории в рабочую директорию контейнера
COPY . .

# Установим зависимости
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

# Указываем команду для запуска бота
CMD ["python", "main.py"]