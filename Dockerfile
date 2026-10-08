FROM python:3.10-slim

WORKDIR /app

# Устанавливаем git для клонирования моделей с Hugging Face
RUN apt-get update && apt-get install -y git

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Клонируем репозиторий с моделями из Hugging Face
RUN git clone https://huggingface.co/butuzik/sms-spam-models /app/models

# Копируем весь код проекта
COPY . .

# Открываем порт 8000
EXPOSE 8000

# Запускаем FastAPI приложение
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

