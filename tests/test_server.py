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
    blend_images,
    edit_image,
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
        from banana_appeal.models import ImageOperationResponse

        save_path = tmp_path / "output.png"
        result = await _generate_image_impl(
            prompt="a beautiful sunset",
            save_path=str(save_path),
        )

        # Now returns ImageOperationResponse when saving
        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(save_path)
        assert result.format == "png"
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
        from banana_appeal.models import ImageOperationResponse

        result = await _edit_image_impl(
            image_path=str(temp_image),
            edit_prompt="make it blue",
        )

        # Now returns ImageOperationResponse
        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(temp_image)  # Overwrites original

    async def test_edit_image_custom_output(self, mock_genai_client, temp_image, tmp_path):
        """Test image editing with custom output path."""
        from banana_appeal.models import ImageOperationResponse

        output_path = tmp_path / "edited.png"
        result = await _edit_image_impl(
            image_path=str(temp_image),
            edit_prompt="add a hat",
            output_path=str(output_path),
        )

        # Now returns ImageOperationResponse
        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(output_path)
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

        from banana_appeal.models import ImageOperationResponse

        img2_path = tmp_path / "test2.png"
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        output_path = tmp_path / "blended.png"
        result = await _blend_images_impl(
            image_paths=[str(temp_image), str(img2_path)],
            prompt="combine",
            output_path=str(output_path),
        )

        # Now returns ImageOperationResponse when saving
        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(output_path)
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


class TestCorrectExtension:
    """Tests for _correct_extension function."""

    def test_matching_jpg_extension(self):
        """Test that .jpg extension is not corrected for JPEG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.jpg"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpg")
        assert warning is None

    def test_matching_jpeg_extension(self):
        """Test that .jpeg extension is not corrected for JPEG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.jpeg"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpeg")
        assert warning is None

    def test_matching_png_extension(self):
        """Test that .png extension is not corrected for PNG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.png"), ImageFormat.PNG)
        assert path == Path("/tmp/image.png")
        assert warning is None

    def test_json_to_jpg_correction(self):
        """Test that .json is corrected to .jpg for JPEG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.json"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpg")
        assert warning is not None
        assert ".json" in warning
        assert ".jpg" in warning
        assert "JPEG" in warning

    def test_png_to_jpg_correction(self):
        """Test that .png is corrected to .jpg when Gemini returns JPEG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.png"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpg")
        assert warning is not None
        assert ".png" in warning
        assert ".jpg" in warning

    def test_uppercase_extension_normalized(self):
        """Test that uppercase extensions are normalized and corrected."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.JSON"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpg")
        assert warning is not None

    def test_no_extension_appends_jpg(self):
        """Test that missing extension gets .jpg appended for JPEG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image"), ImageFormat.JPEG)
        assert path == Path("/tmp/image.jpg")
        assert warning is not None
        assert "No extension provided" in warning

    def test_no_extension_appends_png(self):
        """Test that missing extension gets .png appended for PNG."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image"), ImageFormat.PNG)
        assert path == Path("/tmp/image.png")
        assert warning is not None
        assert "No extension provided" in warning

    def test_webp_correction(self):
        """Test correction to .webp format."""
        from pathlib import Path

        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _correct_extension

        path, warning = _correct_extension(Path("/tmp/image.jpg"), ImageFormat.WEBP)
        assert path == Path("/tmp/image.webp")
        assert warning is not None


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

        # Test that random "pro" in name doesn't match (must have "gemini" too)
        config2 = ServerConfig(api_key="test", model_name="some-production-model")
        assert config2.is_pro_model is False

        # Test that "gemini" alone doesn't match (must have "pro" too)
        config3 = ServerConfig(api_key="test", model_name="gemini-flash-image")
        assert config3.is_pro_model is False

    def test_is_pro_model_future_versions(self):
        """Test Pro model detection works for future model versions."""
        # Should match future Pro versions
        pro_models = [
            "gemini-4-pro-image",
            "gemini-5-pro",
            "gemini-10-pro-vision",
            "gemini-pro-ultra",
            "GEMINI-PRO-IMAGE",  # Case insensitive
        ]
        for model in pro_models:
            config = ServerConfig(api_key="test", model_name=model)
            assert config.is_pro_model is True, f"{model} should be detected as Pro"
            assert config.max_blend_images == 14, f"{model} should have 14 max blend"

        # Should NOT match Flash models
        flash_models = [
            "gemini-4-flash",
            "gemini-5.5-flash-image",
            "gemini-flash-ultra",
            "other-pro-model",  # Has "pro" but not "gemini"
        ]
        for model in flash_models:
            config = ServerConfig(api_key="test", model_name=model)
            assert config.is_pro_model is False, f"{model} should NOT be Pro"
            assert config.max_blend_images == 3, f"{model} should have 3 max blend"


class TestAddVerboseFields:
    """Tests for _add_verbose_fields helper."""

    @pytest.fixture
    def sample_png_bytes(self) -> bytes:
        """Create a minimal valid PNG image."""
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def base_response(self):
        """Create a base response to enhance."""
        from banana_appeal.models import ImageOperationResponse

        return ImageOperationResponse(
            path="/tmp/test.png",
            format="png",
            warnings=["test warning"],
        )

    @pytest.mark.asyncio
    async def test_adds_dimensions(self, sample_png_bytes, base_response):
        """Test that dimensions are extracted from image data."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "test-model", None
        )
        assert result.dimensions is not None
        assert result.dimensions.width == 100
        assert result.dimensions.height == 50

    @pytest.mark.asyncio
    async def test_adds_size_bytes(self, sample_png_bytes, base_response):
        """Test that size_bytes is set to image data length."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "test-model", None
        )
        assert result.size_bytes == len(sample_png_bytes)

    @pytest.mark.asyncio
    async def test_adds_generation_time(self, sample_png_bytes, base_response):
        """Test that generation_time_ms is passed through."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1234.5, "test-model", None
        )
        assert result.generation_time_ms == 1234.5

    @pytest.mark.asyncio
    async def test_adds_model_name(self, sample_png_bytes, base_response):
        """Test that model name is included."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "gemini-2.5-flash", None
        )
        assert result.model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_adds_seed_when_provided(self, sample_png_bytes, base_response):
        """Test that seed is included when provided."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "test-model", 42
        )
        assert result.seed == 42

    @pytest.mark.asyncio
    async def test_preserves_base_fields(self, sample_png_bytes, base_response):
        """Test that base response fields are preserved."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "test-model", None
        )
        assert result.path == "/tmp/test.png"
        assert result.format == "png"
        assert result.warnings == ["test warning"]

    @pytest.mark.asyncio
    async def test_preserves_original_path(self, sample_png_bytes):
        """Test that original_path is preserved when present."""
        from banana_appeal.models import ImageOperationResponse
        from banana_appeal.server import _add_verbose_fields

        base_response = ImageOperationResponse(
            path="/tmp/test.png",
            format="png",
            warnings=["extension corrected"],
            original_path="/tmp/test.jpg",
        )
        result = await _add_verbose_fields(
            base_response, sample_png_bytes, 1000.0, "test-model", None
        )
        assert result.original_path == "/tmp/test.jpg"

    @pytest.mark.asyncio
    async def test_handles_pillow_decode_error(self, base_response):
        """Test that Pillow decode errors return base response with warning."""
        from banana_appeal.server import _add_verbose_fields

        # Invalid image data that Pillow can't decode
        corrupt_data = b"not a valid image format at all"

        result = await _add_verbose_fields(base_response, corrupt_data, 1000.0, "test-model", 42)

        # Should return response with base fields preserved
        assert result.path == "/tmp/test.png"
        assert result.format == "png"
        # Should have warning about decode failure
        assert any("dimension" in w.lower() for w in result.warnings)
        # Dimensions should be None (couldn't extract)
        assert result.dimensions is None
        # Other verbose fields should still be populated
        assert result.size_bytes == len(corrupt_data)
        assert result.generation_time_ms == 1000.0
        assert result.model == "test-model"
        assert result.seed == 42


class TestGenerateImageStructuredResponse:
    """Tests for generate_image structured response."""

    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini response with image data."""
        # Create a minimal PNG
        import io

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 50), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        # Mock response structure matching Gemini API
        mock_part = type(
            "Part",
            (),
            {
                "inline_data": type(
                    "InlineData", (), {"data": png_bytes, "mime_type": "image/png"}
                )()
            },
        )()
        mock_candidate = type(
            "Candidate", (), {"content": type("Content", (), {"parts": [mock_part]})()}
        )()
        return type("Response", (), {"candidates": [mock_candidate]})()

    @pytest.mark.asyncio
    async def test_returns_dict_when_saving(
        self, mock_gemini_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that saving returns a dict with path, format, warnings."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_gemini_response)
        )

        save_path = str(tmp_path / "test.png")
        result = await generate_image.fn(prompt="a blue square", save_path=save_path)

        assert isinstance(result, dict)
        assert "path" in result
        assert "format" in result
        assert "warnings" in result
        assert result["format"] == "png"

    @pytest.mark.asyncio
    async def test_extension_correction_in_response(
        self, mock_gemini_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that extension mismatch is corrected and reported."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_gemini_response)
        )

        # Request .jpg but Gemini returns PNG
        save_path = str(tmp_path / "test.jpg")
        result = await generate_image.fn(prompt="a blue square", save_path=save_path)

        assert isinstance(result, dict)
        assert result["path"].endswith(".png")  # Corrected
        assert result["original_path"] == save_path  # Original preserved
        assert len(result["warnings"]) > 0  # Warning present

    @pytest.mark.asyncio
    async def test_verbose_adds_metadata(
        self, mock_gemini_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that verbose=True adds metadata fields."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_gemini_response)
        )

        save_path = str(tmp_path / "test.png")
        result = await generate_image.fn(prompt="a blue square", save_path=save_path, verbose=True)

        assert isinstance(result, dict)
        assert "dimensions" in result
        assert result["dimensions"]["width"] == 100
        assert result["dimensions"]["height"] == 50
        assert "size_bytes" in result
        assert "generation_time_ms" in result
        assert "model" in result

    @pytest.mark.asyncio
    async def test_verbose_false_excludes_metadata(
        self, mock_gemini_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that verbose=False (default) excludes metadata fields."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_gemini_response)
        )

        save_path = str(tmp_path / "test.png")
        result = await generate_image.fn(prompt="a blue square", save_path=save_path, verbose=False)

        assert isinstance(result, dict)
        assert "dimensions" not in result
        assert "size_bytes" not in result
        assert "generation_time_ms" not in result
        assert "model" not in result

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, mock_gemini_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that parent directories are created when they don't exist."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_gemini_response)
        )

        # Nested path where parent dirs don't exist
        nested_path = tmp_path / "nested" / "deep" / "dir" / "test.png"
        assert not nested_path.parent.exists()

        result = await generate_image.fn(prompt="a blue square", save_path=str(nested_path))

        assert isinstance(result, dict)
        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestEditImageStructuredResponse:
    """Tests for edit_image structured response."""

    @pytest.fixture
    def sample_input_image(self, tmp_path) -> Path:
        """Create a sample input image for editing."""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 100), color="green")
        path = tmp_path / "input.png"
        img.save(path, format="PNG")
        return path

    @pytest.fixture
    def mock_edit_response(self):
        """Create a mock Gemini response for edit."""
        import io

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()

        mock_part = type(
            "Part",
            (),
            {
                "inline_data": type(
                    "InlineData", (), {"data": jpeg_bytes, "mime_type": "image/jpeg"}
                )()
            },
        )()
        mock_candidate = type(
            "Candidate", (), {"content": type("Content", (), {"parts": [mock_part]})()}
        )()
        return type("Response", (), {"candidates": [mock_candidate]})()

    @pytest.mark.asyncio
    async def test_returns_dict(
        self, sample_input_image, mock_edit_response, monkeypatch, mock_env_api_key
    ):
        """Test that edit_image returns a dict with path, format, warnings."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_edit_response)
        )

        result = await edit_image.fn(image_path=str(sample_input_image), edit_prompt="make it red")

        assert isinstance(result, dict)
        assert "path" in result
        assert "format" in result
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_extension_correction(
        self, sample_input_image, mock_edit_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that extension mismatch is corrected."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_edit_response)
        )

        # Output path has .png but Gemini returns JPEG
        output_path = str(tmp_path / "output.png")
        result = await edit_image.fn(
            image_path=str(sample_input_image), edit_prompt="make it red", output_path=output_path
        )

        assert isinstance(result, dict)
        assert result["path"].endswith(".jpg")  # Corrected
        assert result["original_path"] == output_path
        assert len(result["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_original_deleted_on_inplace_extension_correction(
        self, sample_input_image, mock_edit_response, monkeypatch, mock_env_api_key
    ):
        """Test that original file is deleted when extension corrected during in-place edit.

        When output_path=None (overwrite original) and Gemini returns a different format,
        the original file should be deleted to avoid leaving orphaned files.
        """
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_edit_response)
        )

        # Verify input exists and is a .png
        assert sample_input_image.exists()
        assert sample_input_image.suffix == ".png"
        original_path = sample_input_image

        # Edit in-place (output_path=None) - Gemini returns JPEG
        result = await edit_image.fn(
            image_path=str(sample_input_image), edit_prompt="make it red", output_path=None
        )

        assert isinstance(result, dict)
        # New file has corrected extension
        assert result["path"].endswith(".jpg")
        # Original path is recorded in response
        assert result["original_path"] == str(original_path)
        # Warning mentions original was deleted
        assert any("original file deleted" in w for w in result["warnings"])
        # New .jpg file exists
        assert Path(result["path"]).exists()
        # Original .png file was deleted
        assert not original_path.exists()

    @pytest.mark.asyncio
    async def test_verbose_mode(
        self, sample_input_image, mock_edit_response, monkeypatch, mock_env_api_key
    ):
        """Test that verbose=True adds metadata."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_edit_response)
        )

        result = await edit_image.fn(
            image_path=str(sample_input_image), edit_prompt="make it red", verbose=True
        )

        assert isinstance(result, dict)
        assert "dimensions" in result
        assert "size_bytes" in result
        assert "generation_time_ms" in result
        assert "model" in result

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, sample_input_image, mock_edit_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that parent directories are created when they don't exist."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_edit_response)
        )

        # Nested path where parent dirs don't exist
        nested_path = tmp_path / "nested" / "deep" / "dir" / "output.jpg"
        assert not nested_path.parent.exists()

        result = await edit_image.fn(
            image_path=str(sample_input_image),
            edit_prompt="make it red",
            output_path=str(nested_path),
        )

        assert isinstance(result, dict)
        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestBlendImagesStructuredResponse:
    """Tests for blend_images structured response."""

    @pytest.fixture
    def sample_images(self, tmp_path) -> list[Path]:
        """Create sample images for blending."""
        from PIL import Image as PILImage

        paths = []
        for i, color in enumerate(["red", "blue"]):
            img = PILImage.new("RGB", (100, 100), color=color)
            path = tmp_path / f"img{i}.png"
            img.save(path, format="PNG")
            paths.append(path)
        return paths

    @pytest.fixture
    def mock_blend_response(self):
        """Create a mock Gemini response for blend."""
        import io

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 100), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        mock_part = type(
            "Part",
            (),
            {
                "inline_data": type(
                    "InlineData", (), {"data": png_bytes, "mime_type": "image/png"}
                )()
            },
        )()
        mock_candidate = type(
            "Candidate", (), {"content": type("Content", (), {"parts": [mock_part]})()}
        )()
        return type("Response", (), {"candidates": [mock_candidate]})()

    @pytest.mark.asyncio
    async def test_returns_dict_when_saving(
        self, sample_images, mock_blend_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that saving returns a dict."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_blend_response)
        )

        output_path = str(tmp_path / "blended.png")
        result = await blend_images.fn(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these",
            output_path=output_path,
        )

        assert isinstance(result, dict)
        assert "path" in result
        assert "format" in result
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_returns_image_when_inline(
        self, sample_images, mock_blend_response, monkeypatch, mock_env_api_key
    ):
        """Test that no output_path returns Image."""
        from unittest.mock import AsyncMock

        from fastmcp import Image

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_blend_response)
        )

        result = await blend_images.fn(
            image_paths=[str(p) for p in sample_images], prompt="blend these"
        )

        assert isinstance(result, Image)

    @pytest.mark.asyncio
    async def test_extension_correction(
        self, sample_images, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test extension correction when format mismatches."""
        import io

        from PIL import Image as PILImage

        # Create JPEG response
        img = PILImage.new("RGB", (100, 100), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()

        mock_part = type(
            "Part",
            (),
            {
                "inline_data": type(
                    "InlineData", (), {"data": jpeg_bytes, "mime_type": "image/jpeg"}
                )()
            },
        )()
        mock_candidate = type(
            "Candidate", (), {"content": type("Content", (), {"parts": [mock_part]})()}
        )()
        mock_response = type("Response", (), {"candidates": [mock_candidate]})()

        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_response)
        )

        # Request .png but get JPEG
        output_path = str(tmp_path / "blended.png")
        result = await blend_images.fn(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these",
            output_path=output_path,
        )

        assert isinstance(result, dict)
        assert result["path"].endswith(".jpg")
        assert result["original_path"] == output_path

    @pytest.mark.asyncio
    async def test_verbose_mode(
        self, sample_images, mock_blend_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test verbose adds metadata."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_blend_response)
        )

        output_path = str(tmp_path / "blended.png")
        result = await blend_images.fn(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these",
            output_path=output_path,
            verbose=True,
        )

        assert isinstance(result, dict)
        assert "dimensions" in result
        assert "size_bytes" in result
        assert "generation_time_ms" in result
        assert "model" in result

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, sample_images, mock_blend_response, tmp_path, monkeypatch, mock_env_api_key
    ):
        """Test that parent directories are created when they don't exist."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "banana_appeal.server._call_gemini_api", AsyncMock(return_value=mock_blend_response)
        )

        # Nested path where parent dirs don't exist
        nested_path = tmp_path / "nested" / "deep" / "dir" / "blended.png"
        assert not nested_path.parent.exists()

        result = await blend_images.fn(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these",
            output_path=str(nested_path),
        )

        assert isinstance(result, dict)
        assert nested_path.exists()
        assert nested_path.parent.exists()
