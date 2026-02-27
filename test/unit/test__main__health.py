import pytest
from fastapi.testclient import TestClient

from src.main import app, reset_job


@pytest.fixture
def client():
    reset_job()
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint returns expected response."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service_name"] == "evtexture"
    assert "timestamp" in data


def test_get_job_no_job(client):
    """Test get job endpoint when no job exists."""
    response = client.get("/job")
    assert response.status_code == 200
    assert response.json() is None


def test_start_job(client):
    """Test that starting a job returns expected response."""
    response = client.post(
        "/job",
        json={
            "job_id": "job1",
            "input_params": {"input": "/path/to/input.mp4"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "job1"
    assert data["status"] == "running"


def test_cancel_job_no_job(client):
    """Test cancel endpoint when no job exists."""
    response = client.post("/job/cancel")
    assert response.status_code == 404


def test_cancel_job_not_running(client):
    """Test cancel endpoint when job is not running."""
    client.post(
        "/job",
        json={
            "job_id": "job1",
            "input_params": {"input": "/path/to/input.mp4"},
        },
    )

    response = client.post("/job/cancel")
    assert response.status_code == 400
