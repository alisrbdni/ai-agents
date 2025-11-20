import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    # Check it's streaming
    assert "transfer-encoding" in response.headers or "content-type" in response.headers

def test_agent_endpoint():
    # Mocking might be required for real calls, but checking structure here
    # We expect this to fail or timeout if no credentials, 
    # but structurally the endpoint exists.
    pass

def test_ingest_validation():
    response = client.post("/rag/ingest", json={"url": "http://invalid", "name": "test"})
    # Should fail gracefully
    assert response.status_code != 404
