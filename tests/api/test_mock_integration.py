from fastapi.testclient import TestClient

from src.api.main import app


def test_mock_is_explicitly_labeled():
    response = TestClient(app).get("/api/egramswaraj/mock")
    assert response.status_code == 200
    assert response.headers["X-Integration-Status"] == "Mock-Prototype"
