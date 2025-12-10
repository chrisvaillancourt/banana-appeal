"""Pydantic models for request/response validation and configuration."""

from __future__ import annotations

import os
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


class ImageFormat(StrEnum):
    """Supported image formats."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"


class ImageDimensions(BaseModel):
    """Image dimensions in pixels."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")


class AspectRatio(StrEnum):
    """Supported aspect ratios for image generation."""

    SQUARE = "1:1"
    PORTRAIT_2_3 = "2:3"
    LANDSCAPE_3_2 = "3:2"
    PORTRAIT_3_4 = "3:4"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_4_5 = "4:5"
    LANDSCAPE_5_4 = "5:4"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_16_9 = "16:9"
    ULTRAWIDE = "21:9"


class ImageResolution(StrEnum):
    """Supported output resolutions (must be uppercase for Gemini API)."""

    LOW = "1K"
    MEDIUM = "2K"
    HIGH = "4K"


# Pro model detection: model name must contain both "gemini" and "pro"
# Pro models support: 2K/4K resolution, up to 14 images for blending, search grounding
# This handles current and future versions (gemini-3-pro, gemini-4-pro, gemini-pro-vision, etc.)


class ServerConfig(BaseModel):
    """Server configuration loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    # API configuration
    api_key: SecretStr = Field(description="Google API key for Gemini")
    model_name: str = Field(
        default="gemini-2.5-flash-image",
        description="Gemini model to use",
    )

    # Retry configuration
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    retry_timeout_seconds: int = Field(
        default=60, ge=10, le=300, description="Total retry timeout in seconds"
    )

    # Prompt limits
    max_prompt_length: int = Field(
        default=10000, ge=100, le=50000, description="Maximum prompt length"
    )

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration from environment variables."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get your API key from https://aistudio.google.com/apikey"
            )

        return cls(
            api_key=SecretStr(api_key),
            model_name=os.environ.get("BANANA_MODEL", "gemini-2.5-flash-image"),
            retry_attempts=int(os.environ.get("BANANA_RETRY_ATTEMPTS", "3")),
            retry_timeout_seconds=int(os.environ.get("BANANA_RETRY_TIMEOUT", "60")),
            max_prompt_length=int(os.environ.get("BANANA_MAX_PROMPT_LENGTH", "10000")),
        )

    @cached_property
    def is_pro_model(self) -> bool:
        """Check if the current model is a Pro model (supports advanced features).

        Pro models support:
        - 2K/4K resolution output
        - Up to 14 images for blending (6 high-fidelity)
        - Google Search grounding

        Detection: model name must contain both "gemini" and "pro".
        This handles current and future versions (gemini-3-pro, gemini-4-pro, etc.).
        """
        model_lower = self.model_name.lower()
        return "gemini" in model_lower and "pro" in model_lower

    @cached_property
    def max_blend_images(self) -> int:
        """Get the maximum number of images for blending based on the model.

        - Flash models (gemini-2.5-flash-image): 3 images max
        - Pro models (gemini-3-pro-image-preview): 14 images max (6 high-fidelity)
        """
        return 14 if self.is_pro_model else 3


class ConfigurationError(Exception):
    """Raised when server configuration is invalid."""


class GenerateImageRequest(BaseModel):
    """Request model for image generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prompt: Annotated[str, Field(min_length=1, description="Text description of the image")]
    save_path: Path | None = Field(default=None, description="Optional path to save the image")
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="Aspect ratio for the generated image (default: 1:1)"
    )
    resolution: ImageResolution | None = Field(
        default=None, description="Output resolution: 1K, 2K, or 4K (default: 1K)"
    )
    seed: int | None = Field(default=None, ge=0, description="Seed for reproducible generation")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_whitespace(cls, v: str) -> str:
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("save_path", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        path = Path(v)
        # Ensure parent directory exists or can be created
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


class EditImageRequest(BaseModel):
    """Request model for image editing."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    image_path: Path = Field(description="Path to the image to edit")
    edit_prompt: Annotated[str, Field(min_length=1, description="Edit instructions")]
    output_path: Path | None = Field(
        default=None, description="Output path (default: overwrite original)"
    )

    @field_validator("edit_prompt")
    @classmethod
    def validate_prompt_not_whitespace(cls, v: str) -> str:
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Edit prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("image_path", "output_path", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Image file not found: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v


class BlendImagesRequest(BaseModel):
    """Request model for image blending."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    image_paths: Annotated[
        list[Path], Field(min_length=2, max_length=14, description="Paths to images to blend")
    ]
    prompt: Annotated[str, Field(min_length=1, description="Blending instructions")]
    output_path: Path | None = Field(default=None, description="Optional output path")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_whitespace(cls, v: str) -> str:
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("image_paths", mode="before")
    @classmethod
    def validate_paths(cls, v: list[str | Path]) -> list[Path]:
        paths = [Path(p) for p in v]
        for p in paths:
            if not p.exists():
                raise ValueError(f"Image file not found: {p}")
            if not p.is_file():
                raise ValueError(f"Path is not a file: {p}")
        return paths

    @field_validator("output_path", mode="before")
    @classmethod
    def validate_output_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        path = Path(v)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ImageResult(BaseModel):
    """Result of an image generation/editing operation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether the operation succeeded")
    data: bytes | None = Field(default=None, description="Image data if not saved to file")
    format: ImageFormat = Field(default=ImageFormat.PNG, description="Image format")
    saved_path: Path | None = Field(default=None, description="Path where image was saved")
    error: str | None = Field(default=None, description="Error message if operation failed")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Ensure result is consistent."""
        if self.success:
            if self.data is None and self.saved_path is None:
                raise ValueError("Successful result must have either data or saved_path")
        else:
            if self.error is None:
                raise ValueError("Failed result must have an error message")
        return self

    @classmethod
    def from_saved(cls, path: Path, fmt: ImageFormat = ImageFormat.PNG) -> Self:
        """Create a result for a saved image."""
        return cls(success=True, saved_path=path, format=fmt)

    @classmethod
    def from_data(cls, data: bytes, fmt: ImageFormat = ImageFormat.PNG) -> Self:
        """Create a result with image data."""
        return cls(success=True, data=data, format=fmt)

    @classmethod
    def from_error(cls, error: str) -> Self:
        """Create a failed result."""
        return cls(success=False, error=error)


class APICallMetrics(BaseModel):
    """Metrics for an API call (for logging/observability)."""

    model_config = ConfigDict(frozen=True)

    operation: str = Field(description="Operation name (generate, edit, blend)")
    model: str = Field(description="Model used")
    prompt_length: int = Field(description="Length of the prompt")
    image_count: int = Field(default=0, description="Number of input images")
    duration_ms: float = Field(description="Duration in milliseconds")
    success: bool = Field(description="Whether the call succeeded")
    retry_count: int = Field(default=0, description="Number of retries")
    error_type: str | None = Field(default=None, description="Error type if failed")
