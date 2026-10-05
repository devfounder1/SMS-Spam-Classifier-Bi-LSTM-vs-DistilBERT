# SMS Spam Classifier: Bi-LSTM vs DistilBERT

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

API для классификации SMS-сообщений на спам/не спам с использованием двух различных архитектур. Проект демонстрирует эволюцию подходов в NLP: от классических рекуррентных сетей, написанных с нуля, до современных трансформеров.


## Описание проекта

API для классификации SMS-сообщений на спам/не спам с использованием двух архитектур:
- **Bi-LSTM** (написана с нуля на PyTorch)
- **DistilBERT** (fine-tuning предобученной модели через Hugging Face)

Проект демонстрирует эволюцию подходов в NLP: от классических рекуррентных сетей к современным трансформерам.

## Сравнение моделей

Обе модели обучались на [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (5572 сообщения).

| Метрика | Bi-LSTM (Custom) | DistilBERT (Hugging Face) |
|---------|------------------|---------------------------|
| **Accuracy** | 0.9587 | **0.9919** |
| **Precision** | 0.8503 | **0.9730** |
| **Recall** | 0.8389 | **0.9664** |
| **F1-Score** | 0.8446 | **0.9697** |
| **Параметры** | ~500K | ~67M |

**Вывод:** DistilBERT показывает на 12.5% лучший F1-Score благодаря предобученным языковым представлениям.

## Быстрый старт

### Локальный запуск

```bash
# 1. Клонируем репозиторий
git clone https://github.com/devfounder1/SMS-Spam-Classifier-Bi-LSTM-vs-DistilBERT.git
cd SMS-Spam-Classifier-Bi-LSTM-vs-DistilBERT

# 2. Создаем виртуальное окружение и устанавливаем зависимости
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Запускаем API сервер
uvicorn api:app --reload
```
*Примечание: Для работы API предварительно необходимо обучить и сохранить модели, запустив скрипты `LSTM_classifier.py` и `distilBERT_classifier.py`, чтобы в папке `models/` появились веса.*

### Docker Запуск

```bash
# 1. Собираем образ
docker build -t spam-classifier .

# 2. Запускаем контейнер
docker run -p 8000:8000 spam-classifier
```
API будет доступен по адресу: http://localhost:8000

##  API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Предсказание через Bi-LSTM
```bash
curl -X POST "http://localhost:8000/predict/lstm" \
  -H "Content-Type: application/json" \
  -d '{"text": "Win a free iPhone now!"}'
```

### Предсказание через DistilBERT
```bash
curl -X POST "http://localhost:8000/predict/distilbert" \
  -H "Content-Type: application/json" \
  -d '{"text": "Win a free iPhone now!"}'
```

### Пример ответа API:
```json
{
  "text": "Win a free iPhone now!",
  "model_used": "DistilBERT (Hugging Face)",
  "is_spam": true,
  "confidence_spam": 0.9875,
  "confidence_ham": 0.0125
}
```

### Swagger UI
Интерактивная документация доступна по адресу: http://localhost:8000/docs

## Модели
Модели не включены в репозиторий из-за размера. Для локального запуска:
1. Запустите LSTM_classifier.py для обучения и сохранения LSTM модели
2. Запустите destilBERT_classifier.py для fine-tuning DistilBERT
3. Модели сохранятся в папку models/

## ️Технологии

**Python 3.10**
**PyTorch 2.0+** — фреймворк для глубокого обучения
**Hugging Face Transformers** — библиотека для работы с предобученными моделями
**FastAPI** — современный веб-фреймворк для API
**Docker** — контейнеризация приложения
**Scikit-learn** — метрики и разделение данных

## Структура проекта
```text
.
├── api.py                          # FastAPI сервер с эндпоинтами для обеих моделей
├── LSTM_classifier.py              # Скрипт для обучения и сохранения Bi-LSTM с нуля
├── distilBERT_classifier.py        # Скрипт для fine-tuning DistilBERT
├── requirements.txt                # Зависимости проекта
├── Dockerfile                      # Инструкции для сборки Docker-образа
├── .gitignore                      # Исключенные файлы (модели, кэш, venv)
└── README.md                       # Документация проекта
```

## Планы на будущее

- [ ] Экспорт моделей в формат **ONNX** для ускорения инференса на CPU и снижения потребления памяти.
- [ ] Добавление мониторинга метрик API в реальном времени (**Prometheus + Grafana**).
- [ ] Настройка **CI/CD** пайплайна (GitHub Actions) с автоматическим тестированием эндпоинтов.
- [ ] Деплой приложения на облачную платформу (AWS EC2 / Google Cloud Run).

## Лицензия
Этот проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).