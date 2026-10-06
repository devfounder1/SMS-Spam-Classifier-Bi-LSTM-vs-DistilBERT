import pytest
from fastapi.testclient import TestClient
from api import app, lstm_model, hf_model

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["Models_loaded"] is True

@pytest.mark.skipif(lstm_model is None, reason="LSTM модель не загружена")
def test_predict_lstm():
    response = client.post("/predict/lstm", json={"text": "You won a free prize!"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_spam"] is True
    assert "confidence_spam" in data

@pytest.mark.skipif(hf_model is None, reason="DistilBERT модель не загружена")
def test_predict_distilbert():
    response = client.post("/predict/distilbert", json={"text": "Hey, meeting at 5?"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_spam"] is False
    assert "confidence_ham" in data

def test_empty_text_validation():
    assert client.post("/predict/lstm", json={"text": ""}).status_code == 400
    assert client.post("/predict/distilbert", json={"text": "   "}).status_code == 400

@pytest.mark.skipif(lstm_model is None or hf_model is None, reason="Модели не загружены")
def test_compare_endpoint():
    response = client.post("/predict/compare", json={"text": "You won a free prize!"})
    assert response.status_code == 200
    data = response.json()
    assert "lstm_prediction" in data
    assert "distilbert_prediction" in data
    assert "model_agree" in data
    assert "final_verdict" in data
    assert "recomendation" in data

def test_stats_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime" in data