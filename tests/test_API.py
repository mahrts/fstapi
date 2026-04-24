from fastapi.testclient import TestClient
from API.endpoints import app

with TestClient(app) as client:
    def test_root_endpoint():
        """Testing root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "about" in data

    def test_scores_endpoint():
        """Testing the get endpoint."""
        response = client.get("/scores")

        assert response.status_code == 200
        assert response.json() is not None

    def test_slice_success():
        payload = {
                   "data": {
                            "native-country": "United-States",
                            "race": "White",
                            "education": "Masters",
                            },
                  }

        response = client.post("/slice", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "scores" in data
