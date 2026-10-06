import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_predict_lstm():
    response = client.post(
        "/predict/lstm",
        json={"text": "You won a free prize!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_spam" in data
    assert "confidence_spam" in data

def test_predict_distilbert():
    response = client.post(
        "/predict/distilbert",
        json={"text": "Hey, are we still meeting at 5?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_spam" in data
    assert "confidence_ham" in data