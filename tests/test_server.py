"""Tests for the Banana Appeal server."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from banana_appeal.models import (
    BlendImagesRequest,
    ConfigurationError,
    EditImageRequest,
    GenerateImageRequest,
    ImageFormat,
    ImageResolution,
    ImageResult,
    ServerConfig,
)
from banana_appeal.server import (
    _blend_images_impl,
    _edit_image_impl,
    _extract_image_data,
    _generate_image_impl,
    generate_image,
)


class TestServerConfig:
    """Tests for server configuration."""

    def test_config_from_env(self, mock_env_api_key):
        """Test loading config from environment."""
        config = ServerConfig.from_env()
        assert config.api_key.get_secret_value() == "test-api-key-12345"
        assert config.retry_attempts == 3
        assert config.retry_timeout_seconds == 60

    def test_config_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ConfigurationError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
            ServerConfig.from_env()

    def test_config_custom_values(self, monkeypatch):
        """Test custom configuration values."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("BANANA_RETRY_ATTEMPTS", "5")
        monkeypatch.setenv("BANANA_RETRY_TIMEOUT", "120")
        monkeypatch.setenv("BANANA_MODEL", "custom-model")

        config = ServerConfig.from_env()
        assert config.retry_attempts == 5
        assert config.retry_timeout_seconds == 120
        assert config.model_name == "custom-model"


class TestExtractImageData:
    """Tests for the _extract_image_data helper function."""

    def test_extract_with_valid_data(self):
        """Test extraction with valid image data."""
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"test_data"
        mock_part.inline_data.mime_type = "image/png"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = _extract_image_data(mock_response)
        assert result is not None
        assert result[0] == b"test_data"
        assert result[1] == ImageFormat.PNG

    def test_extract_with_jpeg(self):
        """Test extraction with JPEG format."""
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"jpeg_data"
        mock_part.inline_data.mime_type = "image/jpeg"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = _extract_image_data(mock_response)
        assert result is not None
        assert result[1] == ImageFormat.JPEG

    def test_extract_with_no_candidates(self):
        """Test extraction with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        result = _extract_image_data(mock_response)
        assert result is None

    def test_extract_with_empty_parts(self):
        """Test extraction with empty parts."""
        mock_content = MagicMock()
        mock_content.parts = []

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = _extract_image_data(mock_response)
        assert result is None


class TestImageResult:
    """Tests for ImageResult model."""

    def test_from_saved(self, tmp_path):
        """Test creating result from saved path."""
        path = tmp_path / "test.png"
        result = ImageResult.from_saved(path, ImageFormat.PNG)
        assert result.success is True
        assert result.saved_path == path
        assert result.format == ImageFormat.PNG
        assert result.data is None
        assert result.error is None

    def test_from_data(self):
        """Test creating result from data."""
        result = ImageResult.from_data(b"image_bytes", ImageFormat.JPEG)
        assert result.success is True
        assert result.data == b"image_bytes"
        assert result.format == ImageFormat.JPEG
        assert result.saved_path is None

    def test_from_error(self):
        """Test creating error result."""
        result = ImageResult.from_error("Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None
        assert result.saved_path is None

    def test_invalid_success_without_data_or_path(self):
        """Test that success=True requires data or path."""
        with pytest.raises(ValueError, match="must have either data or saved_path"):
            ImageResult(success=True)

    def test_invalid_failure_without_error(self):
        """Test that success=False requires error message."""
        with pytest.raises(ValueError, match="must have an error message"):
            ImageResult(success=False)


class TestPydanticModels:
    """Tests for Pydantic validation models."""

    def test_generate_image_request_valid(self):
        """Test valid generation request."""
        request = GenerateImageRequest(prompt="a cat", save_path="/tmp/cat.png")
        assert request.prompt == "a cat"
        assert request.save_path == Path("/tmp/cat.png")

    def test_generate_image_request_no_save_path(self):
        """Test generation request without save path."""
        request = GenerateImageRequest(prompt="a dog")
        assert request.prompt == "a dog"
        assert request.save_path is None

    def test_generate_image_request_empty_prompt(self):
        """Test that empty prompt raises validation error."""
        with pytest.raises(ValueError):
            GenerateImageRequest(prompt="")

    def test_generate_image_request_whitespace_prompt(self):
        """Test that whitespace-only prompt raises validation error."""
        # After stripping, becomes empty string which fails min_length validation
        with pytest.raises(ValueError):
            GenerateImageRequest(prompt="   ")

    def test_generate_image_request_strips_whitespace(self):
        """Test that prompt whitespace is stripped."""
        request = GenerateImageRequest(prompt="  hello world  ")
        assert request.prompt == "hello world"

    def test_edit_image_request_valid(self, temp_image):
        """Test valid edit request."""
        request = EditImageRequest(
            image_path=str(temp_image),
            edit_prompt="make it blue",
        )
        assert request.edit_prompt == "make it blue"
        assert request.image_path == temp_image

    def test_edit_image_request_missing_file(self, tmp_path):
        """Test that non-existent file raises validation error."""
        with pytest.raises(ValueError, match="not found"):
            EditImageRequest(
                image_path=tmp_path / "nonexistent.png",
                edit_prompt="make it blue",
            )

    def test_blend_images_request_valid(self, temp_image):
        """Test valid blend request."""
        paths = [str(temp_image), str(temp_image)]
        request = BlendImagesRequest(image_paths=paths, prompt="blend them")
        assert len(request.image_paths) == 2
        assert request.prompt == "blend them"

    def test_blend_images_request_too_few(self, temp_image):
        """Test that <2 images raises validation error."""
        with pytest.raises(ValueError):
            BlendImagesRequest(image_paths=[str(temp_image)], prompt="blend")

    def test_blend_images_request_too_many(self, temp_image):
        """Test that >14 images raises validation error."""
        paths = [str(temp_image)] * 15
        with pytest.raises(ValueError):
            BlendImagesRequest(image_paths=paths, prompt="blend them")


class TestGenerateImage:
    """Tests for generate_image tool."""

    async def test_generate_image_success(self, mock_genai_client):
        """Test successful image generation."""
        result = await _generate_image_impl(prompt="a beautiful sunset")

        assert result.success is True
        assert result.data == b"fake_image_data"
        assert result.format == ImageFormat.PNG
        mock_genai_client.aio.models.generate_content.assert_called_once()

    async def test_generate_image_with_save(self, mock_genai_client, tmp_path):
        """Test image generation with save path."""
        save_path = tmp_path / "output.png"
        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            save_path=str(save_path),
        )

        assert result.success is True
        assert result.saved_path == save_path
        assert result.data is None
        assert save_path.exists()

    async def test_generate_image_no_result(self, mock_empty_response):
        """Test handling of empty response."""
        result = await _generate_image_impl(prompt="a cat")
        assert result.success is False
        assert "No image was generated" in result.error

    async def test_generate_image_no_candidates(self, mock_no_candidates):
        """Test handling of response with no candidates."""
        result = await _generate_image_impl(prompt="a cat")
        assert result.success is False
        assert "No image was generated" in result.error

    async def test_generate_image_invalid_prompt(self, mock_genai_client):
        """Test handling of invalid prompt."""
        result = await _generate_image_impl(prompt="   ")
        assert result.success is False
        assert "Invalid request" in result.error


class TestEditImage:
    """Tests for edit_image tool."""

    async def test_edit_image_success(self, mock_genai_client, temp_image):
        """Test successful image editing."""
        result = await _edit_image_impl(
            image_path=str(temp_image),
            edit_prompt="make it blue",
        )

        assert result.success is True
        assert result.saved_path == temp_image  # Overwrites original

    async def test_edit_image_custom_output(self, mock_genai_client, temp_image, tmp_path):
        """Test image editing with custom output path."""
        output_path = tmp_path / "edited.png"
        result = await _edit_image_impl(
            image_path=str(temp_image),
            edit_prompt="add a hat",
            output_path=str(output_path),
        )

        assert result.success is True
        assert result.saved_path == output_path
        assert output_path.exists()

    async def test_edit_image_missing_file(self, mock_genai_client, tmp_path):
        """Test editing non-existent file."""
        result = await _edit_image_impl(
            image_path=str(tmp_path / "nonexistent.png"),
            edit_prompt="change color",
        )

        assert result.success is False
        assert "Invalid request" in result.error


class TestBlendImages:
    """Tests for blend_images tool."""

    async def test_blend_images_success(self, mock_genai_client, temp_image, tmp_path):
        """Test successful image blending."""
        from PIL import Image

        img2_path = tmp_path / "test2.png"
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        result = await _blend_images_impl(
            image_paths=[str(temp_image), str(img2_path)],
            prompt="combine these images",
        )

        assert result.success is True
        assert result.data == b"fake_image_data"

    async def test_blend_images_with_output(self, mock_genai_client, temp_image, tmp_path):
        """Test blending with output path."""
        from PIL import Image

        img2_path = tmp_path / "test2.png"
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        output_path = tmp_path / "blended.png"
        result = await _blend_images_impl(
            image_paths=[str(temp_image), str(img2_path)],
            prompt="combine",
            output_path=str(output_path),
        )

        assert result.success is True
        assert result.saved_path == output_path
        assert output_path.exists()

    async def test_blend_images_too_few(self, mock_genai_client, temp_image):
        """Test blending with too few images."""
        result = await _blend_images_impl(
            image_paths=[str(temp_image)],
            prompt="blend",
        )

        assert result.success is False
        assert "Invalid request" in result.error

    async def test_blend_images_many_inputs(self, monkeypatch, mock_genai_client, tmp_path):
        """Test blending with multiple images (requires Pro model for >3 images)."""
        from PIL import Image

        # Use Pro model for >3 images
        monkeypatch.setenv("BANANA_MODEL", "gemini-3-pro-image-preview")
        import banana_appeal.server as server_module

        server_module._config = None

        paths = []
        for i in range(5):
            path = tmp_path / f"img{i}.png"
            Image.new("RGB", (50, 50), color=f"#{i * 20:02x}{i * 30:02x}{i * 40:02x}").save(path)
            paths.append(str(path))

        result = await _blend_images_impl(
            image_paths=paths,
            prompt="blend all images together",
        )

        assert result.success is True
        assert result.data == b"fake_image_data"


class TestModelValidation:
    """Tests for model-specific parameter validation."""

    async def test_resolution_2k_rejected_on_flash_model(self, mock_genai_client):
        """Test that 2K resolution is rejected on Flash models."""
        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            resolution=ImageResolution.MEDIUM,
        )

        assert result.success is False
        assert "2K resolution requires a Pro model" in result.error

    async def test_resolution_4k_rejected_on_flash_model(self, mock_genai_client):
        """Test that 4K resolution is rejected on Flash models."""
        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            resolution=ImageResolution.HIGH,
        )

        assert result.success is False
        assert "4K resolution requires a Pro model" in result.error

    async def test_resolution_1k_allowed_on_flash_model(self, mock_genai_client):
        """Test that 1K resolution is allowed on Flash models."""
        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            resolution=ImageResolution.LOW,
        )

        # Should succeed, not error about resolution
        assert result.success is True

    async def test_resolution_2k_allowed_on_pro_model(self, monkeypatch, mock_genai_client):
        """Test that 2K resolution is allowed on Pro models."""
        monkeypatch.setenv("BANANA_MODEL", "gemini-3-pro-image-preview")

        # Reset config to pick up new model
        import banana_appeal.server as server_module

        server_module._config = None

        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            resolution=ImageResolution.MEDIUM,
        )

        # Should succeed, not error about resolution
        assert result.success is True

    async def test_blend_limit_flash_model(self, mock_genai_client, tmp_path):
        """Test that Flash models reject >3 images for blending."""
        from PIL import Image

        paths = []
        for i in range(4):
            path = tmp_path / f"img{i}.png"
            Image.new("RGB", (50, 50), color="red").save(path)
            paths.append(str(path))

        result = await _blend_images_impl(
            image_paths=paths,
            prompt="blend all",
        )

        assert result.success is False
        assert "max 3 images" in result.error

    async def test_blend_limit_pro_model(self, monkeypatch, mock_genai_client, tmp_path):
        """Test that Pro models allow >3 images for blending."""
        from PIL import Image

        monkeypatch.setenv("BANANA_MODEL", "gemini-3-pro-image-preview")

        # Reset config to pick up new model
        import banana_appeal.server as server_module

        server_module._config = None

        paths = []
        for i in range(4):
            path = tmp_path / f"img{i}.png"
            Image.new("RGB", (50, 50), color="blue").save(path)
            paths.append(str(path))

        result = await _blend_images_impl(
            image_paths=paths,
            prompt="blend all",
        )

        # Should succeed, not error about image count
        assert result.success is True

    async def test_none_parameters_allowed(self, mock_genai_client):
        """Test that None/default parameters don't trigger validation errors."""
        result = await _generate_image_impl(
            prompt="a sunset",
            aspect_ratio=None,
            resolution=None,
            seed=None,
        )
        assert result.success is True

    async def test_invalid_aspect_ratio_returns_helpful_error(self, mock_genai_client):
        """Test that invalid aspect_ratio returns a helpful error message."""
        # Access the underlying function via the tool's fn attribute
        result = await generate_image.fn(
            prompt="a sunset",
            aspect_ratio="invalid",
        )
        assert isinstance(result, str)
        assert "Error: Invalid aspect_ratio" in result
        assert "1:1" in result  # Should list valid options

    async def test_invalid_resolution_returns_helpful_error(self, mock_genai_client):
        """Test that invalid resolution returns a helpful error message."""
        result = await generate_image.fn(
            prompt="a sunset",
            resolution="8K",
        )
        assert isinstance(result, str)
        assert "Error: Invalid resolution" in result
        assert "1K" in result  # Should list valid options


class TestServerConfigProperties:
    """Tests for ServerConfig cached properties."""

    def test_is_pro_model_with_flash(self, monkeypatch):
        """Test is_pro_model returns False for Flash models."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("BANANA_MODEL", "gemini-2.5-flash-image")
        config = ServerConfig.from_env()
        assert config.is_pro_model is False
        assert config.max_blend_images == 3

    def test_is_pro_model_with_pro(self, monkeypatch):
        """Test is_pro_model returns True for Pro models."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("BANANA_MODEL", "gemini-3-pro-image-preview")
        config = ServerConfig.from_env()
        assert config.is_pro_model is True
        assert config.max_blend_images == 14

    def test_is_pro_model_pattern_matching(self, monkeypatch):
        """Test Pro model detection with various patterns."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Test gemini-pro pattern
        monkeypatch.setenv("BANANA_MODEL", "gemini-pro-vision")
        config = ServerConfig.from_env()
        assert config.is_pro_model is True

        # Test that random "pro" in name doesn't match (must have pattern)
        monkeypatch.setenv("BANANA_MODEL", "some-production-model")

        # Need to create new config since cached_property

        # Clear the cached_property by creating fresh config
        config2 = ServerConfig(
            api_key="test",
            model_name="some-production-model",
        )
        assert config2.is_pro_model is False
