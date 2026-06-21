from fastapi.testclient import TestClient

from app.main import create_app


def test_index_renders_chat_ui() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Ask questions about SEC filings" in response.text
    assert 'name="message"' in response.text
    assert "New conversation" in response.text
