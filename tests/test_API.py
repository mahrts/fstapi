from fastapi.testclient import TestClient
from API.endpoints import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "app_type" in data
        assert "model_features" in data
        assert "data_columns" in data
        assert "target" in data
