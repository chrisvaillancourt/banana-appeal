"""Pytest configuration and fixtures."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests."""
    import banana_appeal.server as server_module

    # Clear cached config and client
    server_module._config = None
    server_module._client = None

    yield

    # Cleanup after test
    server_module._config = None
    server_module._client = None


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Set mock API key in environment and use default Flash model."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-12345")
    # Ensure tests use Flash model by default (can be overridden in specific tests)
    monkeypatch.setenv("BANANA_MODEL", "gemini-2.5-flash-image")
    return "test-api-key-12345"


@pytest.fixture
def mock_genai_client(mock_env_api_key):
    """Mock the Gemini API client with new response structure."""
    with patch("banana_appeal.server.genai.Client") as mock_client_class:
        mock_client = MagicMock()

        # Create mock response matching API structure:
        # response.candidates[0].content.parts[].inline_data
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake_image_data"
        mock_part.inline_data.mime_type = "image/png"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        # Mock the async API
        mock_aio = MagicMock()
        mock_aio.models = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_empty_response(mock_env_api_key):
    """Mock a response with no image data."""
    with patch("banana_appeal.server.genai.Client") as mock_client_class:
        mock_client = MagicMock()

        # Empty content parts
        mock_content = MagicMock()
        mock_content.parts = []

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_aio = MagicMock()
        mock_aio.models = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_no_candidates(mock_env_api_key):
    """Mock a response with no candidates."""
    with patch("banana_appeal.server.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = []

        mock_aio = MagicMock()
        mock_aio.models = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    from PIL import Image

    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path
