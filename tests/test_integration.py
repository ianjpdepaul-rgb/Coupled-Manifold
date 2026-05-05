"""
Integration tests — require running server on localhost:7860.
Run with: pytest -m integration
"""

import pytest
import httpx

pytestmark = pytest.mark.integration


class TestServerHealth:
    def test_root_returns_200(self, server_up, server_url):
        r = httpx.get(server_url, timeout=5)
        assert r.status_code == 200

    def test_health_endpoint(self, server_up, server_url):
        r = httpx.get(f"{server_url}/health", timeout=5)
        assert r.status_code == 200


class TestChatAPI:
    def test_chat_endpoint_exists(self, server_up, server_url):
        r = httpx.post(
            f"{server_url}/api/chat",
            json={"message": "hello", "history": []},
            timeout=30,
        )
        assert r.status_code in (200, 422)  # 422 if schema differs
