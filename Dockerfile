FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y git

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Устанавливаем huggingface_hub для скачивания моделей
RUN pip install --no-cache-dir huggingface_hub

# Копируем скрипт для скачивания моделей
COPY download_models.py .

# Скачиваем модели с Hugging Face Hub (это скачает настоящие файлы, а не Git LFS pointer'ы)
RUN python download_models.py

# Копируем весь код проекта
COPY . .

# Открываем порт 8000
EXPOSE 8000

# Запускаем FastAPI приложение
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

