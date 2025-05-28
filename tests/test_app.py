# test_app.py
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def generate_dummy_landmarks():
    # Return 21 points with dummy (x, y) values
    return [[i * 0.01, i * 0.01] for i in range(21)]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_input():
    landmarks = generate_dummy_landmarks()
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 200
    assert "hand_sign" in response.json()

def test_predict_invalid_count():
    # Only 10 points instead of 21
    landmarks = [[0.1, 0.2] for _ in range(10)]
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 400
    assert "Expected exactly 21 landmarks" in response.json()["detail"]

def test_predict_invalid_format():
    # A landmark with 3 values instead of 2
    landmarks = [[0.1, 0.2] for _ in range(20)] + [[0.3, 0.4, 0.5]]
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 400
    assert "Each landmark must have exactly 2 values" in response.json()["detail"]
