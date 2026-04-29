from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_success():
    response = client.post(
        "/predict",
        json={"features": [1, 298.4, 308.2, 1282, 60.7, 216]}
    )

    assert response.status_code == 200
    assert "failure" in response.json()


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"features": "not_a_list"}
    )

    assert response.status_code in [400, 422]

def test_predict_response_structure():
    response = client.post(
        "/predict",
        json={"features": [1, 298.4, 308.2, 1282, 60.7, 216]}
    )

    data = response.json()

    assert response.status_code == 200
    assert "failure" in data
    assert isinstance(data["failure"], bool)


def test_predict_multiple_requests():
    for _ in range(5):
        response = client.post(
            "/predict",
            json={"features": [1, 298.4, 308.2, 1282, 60.7, 216]}
        )
        assert response.status_code == 200