import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

print("Загружаем локальную модель из папки ./models/distilbert...")
tokenizer = AutoTokenizer.from_pretrained("./models/distilbert")
model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert")
model.eval()

text = "You won a free prize!"
print(f"Тестируем текст: '{text}'")

# Токенизация (как в api.py)
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=64,
    padding=True
)

# Предсказание
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

# Чтобы корректно обработать случай, если probs - это одно число
if isinstance(probs, float):
    probs = [1.0 - probs, probs]

is_spam = bool(probs[1] > probs[0])
print("\nРЕЗУЛЬТАТ:")
print(f"is_spam: {is_spam}")
print(f"confidence_spam: {probs[1]:.4f}")
print(f"confidence_ham: {probs[0]:.4f}")

if is_spam and probs[1] > 0.9:
    print("\nОТЛИЧНО! Модель работает корректно.")
else:
    print("\nПЛОХО. Модель выдает неверное предсказание локально. Проблема в файлах сохранения.")