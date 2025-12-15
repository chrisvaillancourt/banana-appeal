# Structured Responses & Extension Correction

**Date:** 2025-12-10
**Status:** Approved
**Scope:** All three MCP tools (generate_image, edit_image, blend_images)

## Problem

1. **Extension mismatch**: When users specify `save_path="/tmp/diagram.json"`, Gemini returns JPEG bytes which get saved with the wrong extension. This causes confusion when users try to parse the file.

2. **Opaque return values**: Tools return simple strings like `"Image saved to /tmp/img.jpg"` with no metadata about format, dimensions, or timing.

## Solution

### 1. Extension Auto-Correction

When saving an image, compare the requested extension against the actual format from Gemini. If they don't match:
- Correct the extension automatically
- Add a warning to the response
- Include `original_path` so users know what they requested

### 2. Structured Response

Replace string returns with a structured response when saving to disk:

```python
class ImageDimensions(BaseModel):
    """Image dimensions in pixels."""
    model_config = ConfigDict(frozen=True)

    width: int
    height: int


class ImageOperationResponse(BaseModel):
    """Structured response from image operations."""
    model_config = ConfigDict(frozen=True)

    # Always included
    path: str | None = None
    format: str
    warnings: list[str] = []

    # Only set if correction occurred
    original_path: str | None = None

    # Verbose fields (only when verbose=True)
    dimensions: ImageDimensions | None = None
    size_bytes: int | None = None
    generation_time_ms: float | None = None
    model: str | None = None
    seed: int | None = None
```

### 3. Verbose Flag

Add `verbose: bool = False` parameter to all three tools. When `True`, include additional metadata (dimensions, size, timing, model, seed).

**Rationale:** Minimize token consumption by default while allowing detailed metadata when needed.

## API Changes

### generate_image

```python
@mcp.tool
async def generate_image(
    prompt: Annotated[str, Field(...)],
    save_path: Annotated[str | None, Field(...)] = None,
    aspect_ratio: Annotated[str | None, Field(...)] = None,
    resolution: Annotated[str | None, Field(...)] = None,
    seed: Annotated[int | None, Field(...)] = None,
    verbose: Annotated[bool, Field(
        description="Return detailed metadata (dimensions, timing, etc.)"
    )] = False,
    ctx: Context | None = None,
) -> Image | dict:
```

### edit_image

```python
@mcp.tool
async def edit_image(
    image_path: Annotated[str, Field(...)],
    edit_prompt: Annotated[str, Field(...)],
    output_path: Annotated[str | None, Field(...)] = None,
    verbose: Annotated[bool, Field(...)] = False,
    ctx: Context | None = None,
) -> dict:  # Always saves, always returns structured response
```

### blend_images

```python
@mcp.tool
async def blend_images(
    image_paths: Annotated[list[str], Field(...)],
    prompt: Annotated[str, Field(...)],
    output_path: Annotated[str | None, Field(...)] = None,
    verbose: Annotated[bool, Field(...)] = False,
    ctx: Context | None = None,
) -> Image | dict:
```

## Return Behavior

| Scenario | Return Type |
|----------|-------------|
| `save_path` provided | `dict` (structured response) |
| `save_path` not provided | `Image` (inline data) |

## Example Responses

### Default (no correction)

```json
{"path": "/tmp/img.jpg", "format": "jpeg", "warnings": []}
```

### With correction

```json
{
  "path": "/tmp/img.jpg",
  "format": "jpeg",
  "warnings": ["Gemini returned JPEG image; saved as .jpg (requested .png)"],
  "original_path": "/tmp/img.png"
}
```

### Verbose mode

```json
{
  "path": "/tmp/img.jpg",
  "format": "jpeg",
  "warnings": [],
  "dimensions": {"width": 1376, "height": 768},
  "size_bytes": 689152,
  "generation_time_ms": 3420.5,
  "model": "gemini-2.5-flash-image",
  "seed": null
}
```

## Implementation Details

### Extension Correction Function

```python
FORMAT_EXTENSIONS = {
    ImageFormat.JPEG: {".jpg", ".jpeg"},
    ImageFormat.PNG: {".png"},
    ImageFormat.WEBP: {".webp"},
    ImageFormat.GIF: {".gif"},
}

def _correct_extension(
    save_path: Path,
    actual_format: ImageFormat
) -> tuple[Path, str | None]:
    """Correct file extension to match actual image format.

    Returns:
        (corrected_path, warning_message or None)
    """
    expected_exts = FORMAT_EXTENSIONS.get(actual_format, {".png"})
    current_ext = save_path.suffix.lower()

    # No extension provided
    if not current_ext:
        new_ext = ".jpg" if actual_format == ImageFormat.JPEG else f".{actual_format.value}"
        corrected_path = save_path.with_suffix(new_ext)
        warning = f"No extension provided; saved as {new_ext}"
        return corrected_path, warning

    # Extension matches
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

### Verbose Fields Helper

```python
async def _add_verbose_fields(
    response: ImageOperationResponse,
    image_data: bytes,
    generation_time_ms: float,
    config: ServerConfig,
    seed: int | None,
) -> ImageOperationResponse:
    """Add verbose metadata fields to response."""
    import PIL.Image as PILImage

    # Get dimensions (reads header only, fast)
    img = await to_thread.run_sync(
        lambda: PILImage.open(io.BytesIO(image_data))
    )
    width, height = img.size

    return ImageOperationResponse(
        path=response.path,
        format=response.format,
        warnings=response.warnings,
        original_path=response.original_path,
        dimensions=ImageDimensions(width=width, height=height),
        size_bytes=len(image_data),
        generation_time_ms=generation_time_ms,
        model=config.model_name,
        seed=seed,
    )
```

## Test Cases

### Extension Correction

| Input Extension | Gemini Format | Output Extension | Warning |
|-----------------|---------------|------------------|---------|
| `.json` | JPEG | `.jpg` | Yes |
| `.png` | JPEG | `.jpg` | Yes |
| `.jpg` | JPEG | `.jpg` | No |
| `.jpeg` | JPEG | `.jpeg` | No |
| `.JSON` | JPEG | `.jpg` | Yes (normalized) |
| (none) | JPEG | `.jpg` | Yes |
| (none) | PNG | `.png` | Yes |

### Structured Response

- Default response has `path`, `format`, `warnings` only
- `verbose=True` adds dimensions, size_bytes, timing, model, seed
- `original_path` only present when correction occurred
- All three tools behave consistently

## Future Work

- `banana-appeal-md3`: Add field selection (e.g., `include=["dimensions"]`) for finer-grained control
- `banana-appeal-wjl`: Investigate additional Gemini response metadata (token counts, C2PA details)
