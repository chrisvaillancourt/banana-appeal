# Structured Responses Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add extension auto-correction and structured responses with optional verbose metadata to all three MCP tools.

**Architecture:** Add new response models to `models.py`, helper functions to `server.py`, then update each tool to use structured responses when saving files.

**Tech Stack:** Pydantic models, PIL for dimensions, pytest for testing.

---

## Task 1: Add ImageDimensions Model

**Files:**
- Modify: `src/banana_appeal/models.py` (add after `ImageFormat` class, ~line 28)
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
class TestImageDimensions:
    """Tests for ImageDimensions model."""

    def test_create_dimensions(self):
        """Test creating valid dimensions."""
        from banana_appeal.models import ImageDimensions

        dims = ImageDimensions(width=1920, height=1080)
        assert dims.width == 1920
        assert dims.height == 1080

    def test_dimensions_immutable(self):
        """Test that dimensions are frozen."""
        from banana_appeal.models import ImageDimensions

        dims = ImageDimensions(width=100, height=100)
        with pytest.raises(Exception):  # ValidationError for frozen model
            dims.width = 200
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::TestImageDimensions -v`

Expected: FAIL with `ImportError: cannot import name 'ImageDimensions'`

**Step 3: Write minimal implementation**

Add to `src/banana_appeal/models.py` after line 27 (after `ImageFormat` class):

```python
class ImageDimensions(BaseModel):
    """Image dimensions in pixels."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::TestImageDimensions -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/models.py tests/test_models.py
git commit -m "feat: add ImageDimensions model"
```

---

## Task 2: Add ImageOperationResponse Model

**Files:**
- Modify: `src/banana_appeal/models.py` (add after `ImageDimensions`, before `ServerConfig`)
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::TestImageOperationResponse -v`

Expected: FAIL with `ImportError: cannot import name 'ImageOperationResponse'`

**Step 3: Write minimal implementation**

Add to `src/banana_appeal/models.py` after `ImageDimensions` class:

```python
class ImageOperationResponse(BaseModel):
    """Structured response from image operations."""

    model_config = ConfigDict(frozen=True)

    # Always included
    path: str | None = Field(default=None, description="Path where image was saved")
    format: str = Field(description="Image format (jpeg, png, etc.)")
    warnings: list[str] = Field(default_factory=list, description="Any warnings or corrections")

    # Only set if correction occurred
    original_path: str | None = Field(
        default=None, description="Original requested path if extension was corrected"
    )

    # Verbose fields (only when verbose=True)
    dimensions: ImageDimensions | None = Field(
        default=None, description="Image dimensions in pixels"
    )
    size_bytes: int | None = Field(default=None, description="Image file size in bytes")
    generation_time_ms: float | None = Field(
        default=None, description="Generation time in milliseconds"
    )
    model: str | None = Field(default=None, description="Model used for generation")
    seed: int | None = Field(default=None, description="Seed used for generation")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::TestImageOperationResponse -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/models.py tests/test_models.py
git commit -m "feat: add ImageOperationResponse model"
```

---

## Task 3: Add Extension Correction Function

**Files:**
- Modify: `src/banana_appeal/server.py` (add after `_log_metrics` function, ~line 166)
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::TestCorrectExtension -v`

Expected: FAIL with `ImportError: cannot import name '_correct_extension'`

**Step 3: Write minimal implementation**

Add to `src/banana_appeal/server.py` after line 166 (after `_log_metrics` function):

```python
# Mapping of image formats to valid file extensions
FORMAT_EXTENSIONS: dict[ImageFormat, set[str]] = {
    ImageFormat.JPEG: {".jpg", ".jpeg"},
    ImageFormat.PNG: {".png"},
    ImageFormat.WEBP: {".webp"},
    ImageFormat.GIF: {".gif"},
}


def _correct_extension(
    save_path: Path,
    actual_format: ImageFormat,
) -> tuple[Path, str | None]:
    """Correct file extension to match actual image format.

    Args:
        save_path: The requested save path
        actual_format: The actual image format from Gemini

    Returns:
        Tuple of (corrected_path, warning_message or None)
    """
    expected_exts = FORMAT_EXTENSIONS.get(actual_format, {".png"})
    current_ext = save_path.suffix.lower()

    # No extension provided
    if not current_ext:
        new_ext = ".jpg" if actual_format == ImageFormat.JPEG else f".{actual_format.value}"
        corrected_path = save_path.with_suffix(new_ext)
        warning = f"No extension provided; saved as {new_ext}"
        return corrected_path, warning

    # Extension matches (case-insensitive)
    if current_ext in expected_exts:
        return save_path, None

    # Extension mismatch - correct it
    new_ext = ".jpg" if actual_format == ImageFormat.JPEG else f".{actual_format.value}"
    corrected_path = save_path.with_suffix(new_ext)
    warning = (
        f"Gemini returned {actual_format.value.upper()} image; "
        f"saved as {new_ext} (requested {current_ext})"
    )

    return corrected_path, warning
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::TestCorrectExtension -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/server.py tests/test_server.py
git commit -m "feat: add extension correction function"
```

---

## Task 4: Add Verbose Fields Helper

**Files:**
- Modify: `src/banana_appeal/server.py` (add after `_correct_extension` function)
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
import pytest


class TestAddVerboseFields:
    """Tests for _add_verbose_fields function."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a minimal valid PNG image for testing."""
        import io

        import PIL.Image as PILImage

        # Create a small test image
        img = PILImage.new("RGB", (100, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def base_response(self):
        """Create a base response for testing."""
        from banana_appeal.models import ImageOperationResponse

        return ImageOperationResponse(
            path="/tmp/test.png",
            format="png",
            warnings=[],
        )

    @pytest.mark.asyncio
    async def test_adds_dimensions(self, sample_image_bytes, base_response):
        """Test that dimensions are correctly extracted."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            response=base_response,
            image_data=sample_image_bytes,
            generation_time_ms=1234.5,
            model_name="test-model",
            seed=None,
        )
        assert result.dimensions is not None
        assert result.dimensions.width == 100
        assert result.dimensions.height == 50

    @pytest.mark.asyncio
    async def test_adds_size_bytes(self, sample_image_bytes, base_response):
        """Test that size_bytes matches image data length."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            response=base_response,
            image_data=sample_image_bytes,
            generation_time_ms=1234.5,
            model_name="test-model",
            seed=None,
        )
        assert result.size_bytes == len(sample_image_bytes)

    @pytest.mark.asyncio
    async def test_adds_generation_time(self, sample_image_bytes, base_response):
        """Test that generation_time_ms is included."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            response=base_response,
            image_data=sample_image_bytes,
            generation_time_ms=5678.9,
            model_name="test-model",
            seed=None,
        )
        assert result.generation_time_ms == 5678.9

    @pytest.mark.asyncio
    async def test_adds_model_name(self, sample_image_bytes, base_response):
        """Test that model name is included."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            response=base_response,
            image_data=sample_image_bytes,
            generation_time_ms=1234.5,
            model_name="gemini-2.5-flash-image",
            seed=None,
        )
        assert result.model == "gemini-2.5-flash-image"

    @pytest.mark.asyncio
    async def test_adds_seed_when_provided(self, sample_image_bytes, base_response):
        """Test that seed is included when provided."""
        from banana_appeal.server import _add_verbose_fields

        result = await _add_verbose_fields(
            response=base_response,
            image_data=sample_image_bytes,
            generation_time_ms=1234.5,
            model_name="test-model",
            seed=42,
        )
        assert result.seed == 42

    @pytest.mark.asyncio
    async def test_preserves_base_fields(self, sample_image_bytes):
        """Test that base response fields are preserved."""
        from banana_appeal.models import ImageOperationResponse
        from banana_appeal.server import _add_verbose_fields

        base = ImageOperationResponse(
            path="/tmp/corrected.jpg",
            format="jpeg",
            warnings=["Some warning"],
            original_path="/tmp/original.png",
        )
        result = await _add_verbose_fields(
            response=base,
            image_data=sample_image_bytes,
            generation_time_ms=1234.5,
            model_name="test-model",
            seed=None,
        )
        assert result.path == "/tmp/corrected.jpg"
        assert result.format == "jpeg"
        assert result.warnings == ["Some warning"]
        assert result.original_path == "/tmp/original.png"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::TestAddVerboseFields -v`

Expected: FAIL with `ImportError: cannot import name '_add_verbose_fields'`

**Step 3: Write minimal implementation**

Add to `src/banana_appeal/server.py` after `_correct_extension` function:

```python
async def _add_verbose_fields(
    response: ImageOperationResponse,
    image_data: bytes,
    generation_time_ms: float,
    model_name: str,
    seed: int | None,
) -> ImageOperationResponse:
    """Add verbose metadata fields to response.

    Args:
        response: Base response to augment
        image_data: Raw image bytes
        generation_time_ms: Time taken to generate image
        model_name: Name of the model used
        seed: Seed used for generation (if any)

    Returns:
        New response with verbose fields populated
    """
    import PIL.Image as PILImage

    # Get dimensions (reads header only, fast)
    img = await to_thread.run_sync(lambda: PILImage.open(io.BytesIO(image_data)))
    width, height = img.size

    return ImageOperationResponse(
        # Copy base fields
        path=response.path,
        format=response.format,
        warnings=response.warnings,
        original_path=response.original_path,
        # Add verbose fields
        dimensions=ImageDimensions(width=width, height=height),
        size_bytes=len(image_data),
        generation_time_ms=generation_time_ms,
        model=model_name,
        seed=seed,
    )
```

Also add imports at top of `server.py` if not already present. Update the imports from `banana_appeal.models`:

```python
from banana_appeal.models import (
    APICallMetrics,
    AspectRatio,
    BlendImagesRequest,
    ConfigurationError,
    EditImageRequest,
    GenerateImageRequest,
    ImageDimensions,  # Add this
    ImageFormat,
    ImageOperationResponse,  # Add this
    ImageResolution,
    ImageResult,
    ServerConfig,
)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::TestAddVerboseFields -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/server.py tests/test_server.py
git commit -m "feat: add verbose fields helper function"
```

---

## Task 5: Update generate_image Tool

**Files:**
- Modify: `src/banana_appeal/server.py` (update `_generate_image_impl` and `generate_image`)
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
class TestGenerateImageStructuredResponse:
    """Tests for generate_image structured response."""

    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini response with image data."""
        import io

        import PIL.Image as PILImage

        # Create test image
        img = PILImage.new("RGB", (256, 256), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_returns_dict_when_saving(self, mock_gemini_response, tmp_path, monkeypatch):
        """Test that saving returns a dict, not a string."""
        from banana_appeal.models import ImageFormat
        from banana_appeal.server import _generate_image_impl

        # Mock the Gemini API call
        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_gemini_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        save_path = tmp_path / "test.jpg"
        result = await _generate_image_impl(
            prompt="test image",
            save_path=str(save_path),
        )

        # Should return ImageOperationResponse, not ImageResult
        from banana_appeal.models import ImageOperationResponse

        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(save_path)
        assert result.format == "jpeg"

    @pytest.mark.asyncio
    async def test_extension_correction_in_response(
        self, mock_gemini_response, tmp_path, monkeypatch
    ):
        """Test that extension correction is reflected in response."""
        from banana_appeal.server import _generate_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_gemini_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        # Request .json but Gemini returns JPEG
        save_path = tmp_path / "test.json"
        result = await _generate_image_impl(
            prompt="test image",
            save_path=str(save_path),
        )

        assert result.path == str(tmp_path / "test.jpg")
        assert result.original_path == str(save_path)
        assert len(result.warnings) == 1
        assert ".json" in result.warnings[0]
        assert ".jpg" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_verbose_adds_metadata(self, mock_gemini_response, tmp_path, monkeypatch):
        """Test that verbose=True adds metadata fields."""
        from banana_appeal.server import _generate_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_gemini_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        save_path = tmp_path / "test.jpg"
        result = await _generate_image_impl(
            prompt="test image",
            save_path=str(save_path),
            verbose=True,
        )

        assert result.dimensions is not None
        assert result.dimensions.width == 256
        assert result.dimensions.height == 256
        assert result.size_bytes == len(mock_gemini_response)
        assert result.generation_time_ms is not None
        assert result.model is not None

    @pytest.mark.asyncio
    async def test_verbose_false_excludes_metadata(
        self, mock_gemini_response, tmp_path, monkeypatch
    ):
        """Test that verbose=False excludes metadata fields."""
        from banana_appeal.server import _generate_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_gemini_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        save_path = tmp_path / "test.jpg"
        result = await _generate_image_impl(
            prompt="test image",
            save_path=str(save_path),
            verbose=False,
        )

        assert result.dimensions is None
        assert result.size_bytes is None
        assert result.generation_time_ms is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::TestGenerateImageStructuredResponse -v`

Expected: FAIL (tests expect new behavior)

**Step 3: Update implementation**

Modify `_generate_image_impl` in `src/banana_appeal/server.py`. Replace the existing function (around line 318-377) with:

```python
async def _generate_image_impl(
    prompt: str,
    save_path: str | None = None,
    aspect_ratio: AspectRatio | None = None,
    resolution: ImageResolution | None = None,
    seed: int | None = None,
    verbose: bool = False,
    ctx: Context | None = None,
) -> ImageResult | ImageOperationResponse:
    """Internal implementation of generate_image.

    Returns:
        ImageResult if no save_path (for inline display)
        ImageOperationResponse if save_path provided (structured response)
    """
    # Validate resolution is supported by the current model
    config = get_config()
    if resolution and resolution != ImageResolution.LOW and not config.is_pro_model:
        error_msg = (
            f"{resolution.value} resolution requires a Pro model "
            f"(e.g., gemini-3-pro-image-preview). "
            f"Current model ({config.model_name}) only supports 1K resolution."
        )
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    try:
        request = GenerateImageRequest(
            prompt=prompt,
            save_path=save_path,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            seed=seed,
        )
    except ValueError as e:
        error_msg = f"Invalid request: {e}"
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    if ctx:
        await ctx.info(f"Generating image: {request.prompt[:50]}...")

    start_time = time.perf_counter()

    try:
        response = await _call_gemini_api(
            request.prompt,
            operation="generate",
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            seed=request.seed,
        )
    except GeminiAPIError as e:
        error_msg = str(e)
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)
    except genai_errors.APIError as e:
        error_msg = f"API error: {e}"
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)
    except Exception as e:
        logger.exception("Unexpected error in generate_image")
        error_msg = f"Unexpected error: {e}"
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)

    generation_time_ms = (time.perf_counter() - start_time) * 1000

    result = _extract_image_data(response)
    if not result:
        error_msg = "No image was generated"
        if save_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)

    image_data, fmt = result

    # If saving to file, return structured response
    if request.save_path:
        warnings: list[str] = []
        original_path: str | None = None

        # Correct extension if needed
        corrected_path, warning = _correct_extension(request.save_path, fmt)
        if warning:
            warnings.append(warning)
            original_path = str(request.save_path)

        await _save_image_async(corrected_path, image_data)

        if ctx:
            await ctx.info(f"Image saved to {corrected_path}")

        response_obj = ImageOperationResponse(
            path=str(corrected_path),
            format=fmt.value,
            warnings=warnings,
            original_path=original_path,
        )

        if verbose:
            response_obj = await _add_verbose_fields(
                response=response_obj,
                image_data=image_data,
                generation_time_ms=generation_time_ms,
                model_name=config.model_name,
                seed=seed,
            )

        return response_obj

    # No save path - return inline image
    return ImageResult.from_data(image_data, fmt)
```

Also update the MCP tool wrapper `generate_image` (around line 510-560):

```python
@mcp.tool
async def generate_image(
    prompt: Annotated[str, Field(min_length=1, description="Text description of the image")],
    save_path: Annotated[str | None, Field(description="Optional path to save the image")] = None,
    aspect_ratio: Annotated[
        str | None,
        Field(description="Aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, or 21:9"),
    ] = None,
    resolution: Annotated[
        str | None,
        Field(description="Output resolution: 1K (default), 2K, or 4K (2K/4K require Pro model)"),
    ] = None,
    seed: Annotated[int | None, Field(description="Seed for reproducible generation")] = None,
    verbose: Annotated[
        bool, Field(description="Return detailed metadata (dimensions, timing, etc.)")
    ] = False,
    ctx: Context | None = None,
) -> Image | dict:
    """Generate an image from a text prompt using Gemini.

    Returns the generated image, or saves it to the specified path.
    """
    # Convert string parameters to enum types with helpful error messages
    try:
        ar = AspectRatio(aspect_ratio) if aspect_ratio else None
    except ValueError:
        valid = ", ".join(r.value for r in AspectRatio)
        return {"error": f"Invalid aspect_ratio '{aspect_ratio}'. Valid options: {valid}"}

    try:
        res = ImageResolution(resolution) if resolution else None
    except ValueError:
        valid = ", ".join(r.value for r in ImageResolution)
        return {"error": f"Invalid resolution '{resolution}'. Valid options: {valid}"}

    result = await _generate_image_impl(
        prompt=prompt,
        save_path=save_path,
        aspect_ratio=ar,
        resolution=res,
        seed=seed,
        verbose=verbose,
        ctx=ctx,
    )

    # Structured response for saved files
    if isinstance(result, ImageOperationResponse):
        return result.model_dump(exclude_none=True)

    # ImageResult for inline display or errors
    if not result.success:
        return {"error": result.error}

    if result.data:
        return Image(data=result.data, format=result.format.value)

    return {"error": "No image data"}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::TestGenerateImageStructuredResponse -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/server.py tests/test_server.py
git commit -m "feat: add structured response to generate_image"
```

---

## Task 6: Update edit_image Tool

**Files:**
- Modify: `src/banana_appeal/server.py` (update `_edit_image_impl` and `edit_image`)
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
class TestEditImageStructuredResponse:
    """Tests for edit_image structured response."""

    @pytest.fixture
    def sample_image_file(self, tmp_path):
        """Create a sample image file for editing."""
        import io

        import PIL.Image as PILImage

        img = PILImage.new("RGB", (100, 100), color="green")
        path = tmp_path / "source.jpg"
        img.save(path, format="JPEG")
        return path

    @pytest.fixture
    def mock_edit_response(self):
        """Create mock edited image bytes."""
        import io

        import PIL.Image as PILImage

        img = PILImage.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_returns_dict(self, sample_image_file, mock_edit_response, tmp_path, monkeypatch):
        """Test that edit_image returns a dict."""
        from banana_appeal.server import _edit_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_edit_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        output_path = tmp_path / "edited.jpg"
        result = await _edit_image_impl(
            image_path=str(sample_image_file),
            edit_prompt="make it red",
            output_path=str(output_path),
        )

        from banana_appeal.models import ImageOperationResponse

        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(output_path)
        assert result.format == "jpeg"

    @pytest.mark.asyncio
    async def test_extension_correction(
        self, sample_image_file, mock_edit_response, tmp_path, monkeypatch
    ):
        """Test extension correction in edit_image."""
        from banana_appeal.server import _edit_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_edit_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        # Request .png output but Gemini returns JPEG
        output_path = tmp_path / "edited.png"
        result = await _edit_image_impl(
            image_path=str(sample_image_file),
            edit_prompt="make it red",
            output_path=str(output_path),
        )

        assert result.path == str(tmp_path / "edited.jpg")
        assert result.original_path == str(output_path)
        assert len(result.warnings) == 1

    @pytest.mark.asyncio
    async def test_verbose_mode(
        self, sample_image_file, mock_edit_response, tmp_path, monkeypatch
    ):
        """Test verbose mode in edit_image."""
        from banana_appeal.server import _edit_image_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_edit_response
                    mime_type = "image/jpeg"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        output_path = tmp_path / "edited.jpg"
        result = await _edit_image_impl(
            image_path=str(sample_image_file),
            edit_prompt="make it red",
            output_path=str(output_path),
            verbose=True,
        )

        assert result.dimensions is not None
        assert result.size_bytes is not None
        assert result.generation_time_ms is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::TestEditImageStructuredResponse -v`

Expected: FAIL (tests expect new behavior)

**Step 3: Update implementation**

Replace `_edit_image_impl` in `src/banana_appeal/server.py`:

```python
async def _edit_image_impl(
    image_path: str,
    edit_prompt: str,
    output_path: str | None = None,
    verbose: bool = False,
    ctx: Context | None = None,
) -> ImageOperationResponse:
    """Internal implementation of edit_image.

    Returns:
        ImageOperationResponse (always saves, so always structured response)
    """
    config = get_config()

    try:
        request = EditImageRequest(
            image_path=image_path,
            edit_prompt=edit_prompt,
            output_path=output_path,
        )
    except ValueError as e:
        return ImageOperationResponse(format="unknown", warnings=[f"Invalid request: {e}"])

    if ctx:
        await ctx.info(f"Editing image: {request.edit_prompt[:50]}...")

    try:
        image = await _load_image_async(request.image_path)
    except Exception as e:
        return ImageOperationResponse(format="unknown", warnings=[f"Failed to load image: {e}"])

    start_time = time.perf_counter()

    try:
        response = await _call_gemini_api(
            [request.edit_prompt, image],
            operation="edit",
            image_count=1,
        )
    except GeminiAPIError as e:
        return ImageOperationResponse(format="unknown", warnings=[f"Error: {e}"])
    except genai_errors.APIError as e:
        return ImageOperationResponse(format="unknown", warnings=[f"API error: {e}"])
    except Exception as e:
        logger.exception("Unexpected error in edit_image")
        return ImageOperationResponse(format="unknown", warnings=[f"Unexpected error: {e}"])

    generation_time_ms = (time.perf_counter() - start_time) * 1000

    result = _extract_image_data(response)
    if not result:
        return ImageOperationResponse(
            format="unknown", warnings=["Image editing failed - no output generated"]
        )

    image_data, fmt = result
    save_to = request.output_path or request.image_path

    warnings: list[str] = []
    original_path: str | None = None

    # Correct extension if needed
    corrected_path, warning = _correct_extension(save_to, fmt)
    if warning:
        warnings.append(warning)
        original_path = str(save_to)

    await _save_image_async(corrected_path, image_data)

    if ctx:
        await ctx.info(f"Edited image saved to {corrected_path}")

    response_obj = ImageOperationResponse(
        path=str(corrected_path),
        format=fmt.value,
        warnings=warnings,
        original_path=original_path,
    )

    if verbose:
        response_obj = await _add_verbose_fields(
            response=response_obj,
            image_data=image_data,
            generation_time_ms=generation_time_ms,
            model_name=config.model_name,
            seed=None,  # edit doesn't use seed
        )

    return response_obj
```

Also update the MCP tool wrapper `edit_image`:

```python
@mcp.tool
async def edit_image(
    image_path: Annotated[str, Field(description="Path to the image to edit")],
    edit_prompt: Annotated[str, Field(min_length=1, description="Edit instructions")],
    output_path: Annotated[
        str | None, Field(description="Output path (default: overwrite original)")
    ] = None,
    verbose: Annotated[
        bool, Field(description="Return detailed metadata (dimensions, timing, etc.)")
    ] = False,
    ctx: Context | None = None,
) -> dict:
    """Edit an existing image using natural language instructions.

    Modifies the image according to the edit prompt and saves the result.
    """
    result = await _edit_image_impl(
        image_path=image_path,
        edit_prompt=edit_prompt,
        output_path=output_path,
        verbose=verbose,
        ctx=ctx,
    )

    return result.model_dump(exclude_none=True)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::TestEditImageStructuredResponse -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/server.py tests/test_server.py
git commit -m "feat: add structured response to edit_image"
```

---

## Task 7: Update blend_images Tool

**Files:**
- Modify: `src/banana_appeal/server.py` (update `_blend_images_impl` and `blend_images`)
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
class TestBlendImagesStructuredResponse:
    """Tests for blend_images structured response."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample image files for blending."""
        import PIL.Image as PILImage

        paths = []
        for i, color in enumerate(["red", "blue"]):
            img = PILImage.new("RGB", (100, 100), color=color)
            path = tmp_path / f"img{i}.jpg"
            img.save(path, format="JPEG")
            paths.append(path)
        return paths

    @pytest.fixture
    def mock_blend_response(self):
        """Create mock blended image bytes."""
        import io

        import PIL.Image as PILImage

        img = PILImage.new("RGB", (100, 100), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_returns_dict_when_saving(
        self, sample_images, mock_blend_response, tmp_path, monkeypatch
    ):
        """Test that blend_images returns dict when output_path provided."""
        from banana_appeal.server import _blend_images_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_blend_response
                    mime_type = "image/png"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        output_path = tmp_path / "blended.png"
        result = await _blend_images_impl(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these images",
            output_path=str(output_path),
        )

        from banana_appeal.models import ImageOperationResponse

        assert isinstance(result, ImageOperationResponse)
        assert result.path == str(output_path)
        assert result.format == "png"

    @pytest.mark.asyncio
    async def test_returns_image_result_without_output_path(
        self, sample_images, mock_blend_response, monkeypatch
    ):
        """Test that blend_images returns ImageResult without output_path."""
        from banana_appeal.server import _blend_images_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_blend_response
                    mime_type = "image/png"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        result = await _blend_images_impl(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these images",
            output_path=None,
        )

        from banana_appeal.models import ImageResult

        assert isinstance(result, ImageResult)
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_verbose_mode(
        self, sample_images, mock_blend_response, tmp_path, monkeypatch
    ):
        """Test verbose mode in blend_images."""
        from banana_appeal.server import _blend_images_impl

        async def mock_call(*args, **kwargs):
            class MockPart:
                class InlineData:
                    data = mock_blend_response
                    mime_type = "image/png"

                inline_data = InlineData()

            class MockContent:
                parts = [MockPart()]

            class MockCandidate:
                content = MockContent()

            class MockResponse:
                candidates = [MockCandidate()]

            return MockResponse()

        monkeypatch.setattr("banana_appeal.server._call_gemini_api", mock_call)

        output_path = tmp_path / "blended.png"
        result = await _blend_images_impl(
            image_paths=[str(p) for p in sample_images],
            prompt="blend these images",
            output_path=str(output_path),
            verbose=True,
        )

        assert result.dimensions is not None
        assert result.size_bytes is not None
        assert result.generation_time_ms is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::TestBlendImagesStructuredResponse -v`

Expected: FAIL (tests expect new behavior)

**Step 3: Update implementation**

Replace `_blend_images_impl` in `src/banana_appeal/server.py`:

```python
async def _blend_images_impl(
    image_paths: list[str],
    prompt: str,
    output_path: str | None = None,
    verbose: bool = False,
    ctx: Context | None = None,
) -> ImageResult | ImageOperationResponse:
    """Internal implementation of blend_images.

    Returns:
        ImageResult if no output_path (for inline display)
        ImageOperationResponse if output_path provided (structured response)
    """
    config = get_config()

    # Check model-specific limit before validation
    max_images = config.max_blend_images
    if len(image_paths) > max_images:
        error_msg = (
            f"Model {config.model_name} supports max {max_images} images for blending, "
            f"but {len(image_paths)} were provided. Use a Pro model for more images."
        )
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    try:
        request = BlendImagesRequest(
            image_paths=image_paths,
            prompt=prompt,
            output_path=output_path,
        )
    except ValueError as e:
        error_msg = f"Invalid request: {e}"
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    if ctx:
        await ctx.info(f"Blending {len(request.image_paths)} images...")

    # Load all images concurrently
    images: list[PIL.Image.Image] = []
    load_errors: list[str] = []

    async def load_image(path: Path) -> None:
        try:
            img = await _load_image_async(path)
            images.append(img)
        except Exception as e:
            load_errors.append(f"{path}: {e}")

    async with anyio.create_task_group() as tg:
        for p in request.image_paths:
            tg.start_soon(load_image, p)

    if load_errors:
        error_msg = f"Failed to load images: {', '.join(load_errors)}"
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    start_time = time.perf_counter()

    try:
        response = await _call_gemini_api(
            [request.prompt, *images],
            operation="blend",
            image_count=len(images),
        )
    except GeminiAPIError as e:
        error_msg = str(e)
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)
    except genai_errors.APIError as e:
        error_msg = f"API error: {e}"
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)
    except Exception as e:
        logger.exception("Unexpected error in blend_images")
        error_msg = f"Unexpected error: {e}"
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[f"Error: {error_msg}"])
        return ImageResult.from_error(error_msg)

    generation_time_ms = (time.perf_counter() - start_time) * 1000

    result = _extract_image_data(response)
    if not result:
        error_msg = "Image blending failed - no output generated"
        if output_path:
            return ImageOperationResponse(format="unknown", warnings=[error_msg])
        return ImageResult.from_error(error_msg)

    image_data, fmt = result

    # If saving to file, return structured response
    if request.output_path:
        warnings: list[str] = []
        original_path: str | None = None

        # Correct extension if needed
        corrected_path, warning = _correct_extension(request.output_path, fmt)
        if warning:
            warnings.append(warning)
            original_path = str(request.output_path)

        await _save_image_async(corrected_path, image_data)

        if ctx:
            await ctx.info(f"Blended image saved to {corrected_path}")

        response_obj = ImageOperationResponse(
            path=str(corrected_path),
            format=fmt.value,
            warnings=warnings,
            original_path=original_path,
        )

        if verbose:
            response_obj = await _add_verbose_fields(
                response=response_obj,
                image_data=image_data,
                generation_time_ms=generation_time_ms,
                model_name=config.model_name,
                seed=None,  # blend doesn't use seed
            )

        return response_obj

    # No output path - return inline image
    return ImageResult.from_data(image_data, fmt)
```

Also update the MCP tool wrapper `blend_images`:

```python
@mcp.tool
async def blend_images(
    image_paths: Annotated[
        list[str], Field(min_length=2, max_length=14, description="Paths to images to blend")
    ],
    prompt: Annotated[str, Field(min_length=1, description="Blending instructions")],
    output_path: Annotated[str | None, Field(description="Optional output path")] = None,
    verbose: Annotated[
        bool, Field(description="Return detailed metadata (dimensions, timing, etc.)")
    ] = False,
    ctx: Context | None = None,
) -> Image | dict:
    """Blend multiple images together with a creative prompt.

    Combines up to 14 images according to the prompt instructions.
    Note: Flash models (gemini-2.5-flash-image) only support 3 images max.
    """
    result = await _blend_images_impl(
        image_paths=image_paths,
        prompt=prompt,
        output_path=output_path,
        verbose=verbose,
        ctx=ctx,
    )

    # Structured response for saved files
    if isinstance(result, ImageOperationResponse):
        return result.model_dump(exclude_none=True)

    # ImageResult for inline display or errors
    if not result.success:
        return {"error": result.error}

    if result.data:
        return Image(data=result.data, format=result.format.value)

    return {"error": "No image data"}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::TestBlendImagesStructuredResponse -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/banana_appeal/server.py tests/test_server.py
git commit -m "feat: add structured response to blend_images"
```

---

## Task 8: Run Full Test Suite and Fix Any Issues

**Step 1: Run all tests**

```bash
uv run pytest -v
```

**Step 2: Run linting and formatting**

```bash
uv run ruff check .
uv run ruff format .
uv run pyrefly check src/banana_appeal/
```

**Step 3: Fix any issues found**

Address any test failures or linting errors.

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: fix linting and test issues"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `README.md` or `docs/` as needed

**Step 1: Update tool documentation**

Add examples of the new structured response format and `verbose` parameter to documentation.

**Step 2: Commit**

```bash
git add docs/ README.md
git commit -m "docs: document structured responses and verbose parameter"
```

---

## Summary

**Total Tasks:** 9
**Estimated Time:** ~2-3 hours

**Key Files Modified:**
- `src/banana_appeal/models.py` - New models
- `src/banana_appeal/server.py` - New helpers and updated tools
- `tests/test_models.py` - Model tests
- `tests/test_server.py` - Server tests

**New Features:**
- Extension auto-correction with warnings
- Structured responses when saving files
- Optional verbose metadata
