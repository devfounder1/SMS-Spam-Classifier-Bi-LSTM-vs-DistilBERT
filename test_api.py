import pytest
from fastapi.testclient import TestClient
from api import app, lstm_model, hf_model

client = TestClient(app)

def test_health_check():
    """Тест проверки здоровья API - должен работать всегда"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["Models_loaded"] is True
    assert "/predict/lstm" in data["Endpoints"]
    assert "/predict/distilbert" in data["Endpoints"]

@pytest.mark.skipif(lstm_model is None, reason="LSTM модель не загружена (нет папки models/)")
def test_predict_lstm():
    """Тест предсказания через LSTM - работает только если модель загружена"""
    response = client.post(
        "/predict/lstm",
        json={"text": "You won a free prize!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_spam" in data
    assert "confidence_spam" in data
    assert "confidence_ham" in data
    assert "model_used" in data
    assert data["model_used"] == "Bi-LSTM (Custom)"

@pytest.mark.skipif(hf_model is None, reason="DistilBERT модель не загружена (нет папки models/)")
def test_predict_distilbert():
    """Тест предсказания через DistilBERT - работает только если модель загружена"""
    response = client.post(
        "/predict/distilbert",
        json={"text": "Hey, are we still meeting at 5?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_spam" in data
    assert "confidence_spam" in data
    assert "confidence_ham" in data
    assert "model_used" in data
    assert data["model_used"] == "DistilBERT (Hugging Face)"

def test_empty_text_validation():
    """Тест валидации пустого текста"""
    # Проверяем LSTM
    response = client.post("/predict/lstm", json={"text": ""})
    assert response.status_code == 400
    
    # Проверяем DistilBERT
    response = client.post("/predict/distilbert", json={"text": "   "})
    assert response.status_code == 400