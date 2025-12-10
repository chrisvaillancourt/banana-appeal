"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from banana_appeal.models import ImageDimensions


class TestImageDimensions:
    """Tests for ImageDimensions model."""

    def test_create_dimensions(self):
        """Test creating ImageDimensions with valid values."""
        dims = ImageDimensions(width=1920, height=1080)
        assert dims.width == 1920
        assert dims.height == 1080

    def test_dimensions_immutable(self):
        """Test that ImageDimensions is immutable."""
        dims = ImageDimensions(width=100, height=100)
        with pytest.raises(ValidationError):
            dims.width = 200

    def test_zero_width_raises_error(self):
        """Test that zero width raises validation error."""
        with pytest.raises(ValidationError):
            ImageDimensions(width=0, height=100)

    def test_zero_height_raises_error(self):
        """Test that zero height raises validation error."""
        with pytest.raises(ValidationError):
            ImageDimensions(width=100, height=0)

    def test_negative_width_raises_error(self):
        """Test that negative width raises validation error."""
        with pytest.raises(ValidationError):
            ImageDimensions(width=-100, height=100)

    def test_negative_height_raises_error(self):
        """Test that negative height raises validation error."""
        with pytest.raises(ValidationError):
            ImageDimensions(width=100, height=-100)


class TestImageOperationResponse:
    """Tests for ImageOperationResponse model."""

    def test_minimal_response(self):
        """Test creating response with only required fields."""
        from banana_appeal.models import ImageOperationResponse

        response = ImageOperationResponse(format="jpeg")
        assert response.format == "jpeg"
        assert response.path is None
        assert response.warnings == []
        assert response.original_path is None
        assert response.dimensions is None

    def test_response_with_path(self):
        """Test response with saved path."""
        from banana_appeal.models import ImageOperationResponse

        response = ImageOperationResponse(
            path="/tmp/image.jpg",
            format="jpeg",
            warnings=[],
        )
        assert response.path == "/tmp/image.jpg"
        assert response.format == "jpeg"

    def test_response_with_correction(self):
        """Test response when extension was corrected."""
        from banana_appeal.models import ImageOperationResponse

        response = ImageOperationResponse(
            path="/tmp/image.jpg",
            format="jpeg",
            warnings=["Gemini returned JPEG image; saved as .jpg (requested .png)"],
            original_path="/tmp/image.png",
        )
        assert response.original_path == "/tmp/image.png"
        assert len(response.warnings) == 1

    def test_verbose_response(self):
        """Test response with verbose fields."""
        from banana_appeal.models import ImageDimensions, ImageOperationResponse

        response = ImageOperationResponse(
            path="/tmp/image.jpg",
            format="jpeg",
            warnings=[],
            dimensions=ImageDimensions(width=1920, height=1080),
            size_bytes=102400,
            generation_time_ms=1234.5,
            model="gemini-2.5-flash-image",
            seed=42,
        )
        assert response.dimensions.width == 1920
        assert response.dimensions.height == 1080
        assert response.size_bytes == 102400
        assert response.generation_time_ms == 1234.5
        assert response.model == "gemini-2.5-flash-image"
        assert response.seed == 42

    def test_to_dict(self):
        """Test converting response to dict for MCP return."""
        from banana_appeal.models import ImageOperationResponse

        response = ImageOperationResponse(
            path="/tmp/image.jpg",
            format="jpeg",
            warnings=[],
        )
        result = response.model_dump(exclude_none=True)
        assert result == {
            "path": "/tmp/image.jpg",
            "format": "jpeg",
            "warnings": [],
        }

    def test_to_dict_excludes_none_verbose_fields(self):
        """Test that None verbose fields are excluded from dict."""
        from banana_appeal.models import ImageOperationResponse

        response = ImageOperationResponse(
            path="/tmp/image.jpg",
            format="jpeg",
            warnings=[],
            # verbose fields left as None
        )
        result = response.model_dump(exclude_none=True)
        assert "dimensions" not in result
        assert "size_bytes" not in result
        assert "generation_time_ms" not in result
