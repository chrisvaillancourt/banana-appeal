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
