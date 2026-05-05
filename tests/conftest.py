"""
Shared fixtures for Graceful test suite.
"""

import sys, os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def server_url():
    """Base URL for integration tests. Requires a running server."""
    return "http://localhost:7860"


@pytest.fixture(scope="session")
def server_up(server_url):
    """Skip integration tests if server isn't running."""
    import httpx
    try:
        r = httpx.get(server_url, timeout=3)
        if r.status_code != 200:
            pytest.skip("Server not reachable")
    except Exception:
        pytest.skip("Server not running on localhost:7860")


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests that need filesystem."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "sessions").mkdir()
    return tmp_path
