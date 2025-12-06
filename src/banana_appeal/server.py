#!/usr/bin/env python3
"""
Banana Appeal - A-peel-ing image generation powered by Gemini.

This FastMCP server exposes Gemini's image generation capabilities as MCP tools
that Claude Code can call directly.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import anyio
import stamina
from anyio import Path as AsyncPath
from anyio import to_thread
from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import GenerateContentConfig
from pydantic import Field

from banana_appeal.models import (
    APICallMetrics,
    BlendImagesRequest,
    ConfigurationError,
    EditImageRequest,
    GenerateImageRequest,
    ImageFormat,
    ImageResult,
    ServerConfig,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import PIL.Image

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("banana_appeal")


# Global state
_config: ServerConfig | None = None
_client: genai.Client | None = None
_shutdown_event: asyncio.Event | None = None


class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def get_config() -> ServerConfig:
    """Get the validated server configuration."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
        logger.info(
            "Configuration loaded",
            extra={
                "model": _config.model_name,
                "retry_attempts": _config.retry_attempts,
                "retry_timeout": _config.retry_timeout_seconds,
            },
        )
    return _config


def get_client() -> genai.Client:
    """Get the singleton Gemini client."""
    global _client
    if _client is None:
        config = get_config()
        _client = genai.Client(api_key=config.api_key.get_secret_value())
        logger.info("Gemini client initialized")
    return _client


def _is_retryable_error(exc: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exc, GeminiAPIError):
        return exc.retryable
    if isinstance(exc, genai_errors.APIError):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is not None:
            # Only retry on server errors (5xx) and rate limits (429)
            # Do NOT retry on client errors (4xx except 429)
            return status >= 500 or status == 429
        # No status code - likely a connection error, retry
        return True
    # Retry on connection/timeout errors
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _is_non_retryable_client_error(exc: BaseException) -> bool:
    """Check if this is a client error that should NOT be retried."""
    if isinstance(exc, genai_errors.APIError):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is not None and 400 <= status < 500 and status != 429:
            return True
    return False


async def _save_image_async(path: Path, data: bytes) -> None:
    """Save image data to file asynchronously."""
    await AsyncPath(path).write_bytes(data)
    logger.debug("Image saved", extra={"path": str(path), "size_bytes": len(data)})


async def _load_image_async(path: Path) -> PIL.Image.Image:
    """Load image from file asynchronously."""
    import PIL.Image as PILImage

    data = await AsyncPath(path).read_bytes()
    logger.debug("Image loaded", extra={"path": str(path), "size_bytes": len(data)})
    return await to_thread.run_sync(lambda: PILImage.open(io.BytesIO(data)))


def _extract_image_data(response: Any) -> tuple[bytes, ImageFormat] | None:
    """Extract image data and format from Gemini response."""
    if not response.candidates:
        return None
    content = response.candidates[0].content
    if not content or not content.parts:
        return None
    for part in content.parts:
        if part.inline_data and part.inline_data.data:
            mime_type = part.inline_data.mime_type or "image/png"
            fmt_str = mime_type.split("/")[-1] if "/" in mime_type else "png"
            try:
                fmt = ImageFormat(fmt_str)
            except ValueError:
                fmt = ImageFormat.PNG
            return part.inline_data.data, fmt
    return None


def _log_metrics(metrics: APICallMetrics) -> None:
    """Log API call metrics."""
    log_data = metrics.model_dump(exclude_none=True)
    if metrics.success:
        logger.info("API call completed", extra=log_data)
    else:
        logger.warning("API call failed", extra=log_data)


async def _call_gemini_api(
    contents: str | list[Any],
    operation: str,
    image_count: int = 0,
) -> Any:
    """Call Gemini API with automatic retries on transient errors.

    Uses native async from google-genai SDK.
    """
    config = get_config()
    client = get_client()
    start_time = time.perf_counter()
    retry_count = 0
    prompt_length = len(contents) if isinstance(contents, str) else len(str(contents[0]))

    generation_config = GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    @stamina.retry(
        on=_is_retryable_error,
        attempts=config.retry_attempts,
        timeout=dt.timedelta(seconds=config.retry_timeout_seconds),
    )
    async def _make_request() -> Any:
        nonlocal retry_count
        try:
            # Use native async API
            response = await client.aio.models.generate_content(
                model=config.model_name,
                contents=contents,
                config=generation_config,
            )
            return response
        except genai_errors.APIError as e:
            if _is_non_retryable_client_error(e):
                # Don't retry client errors (bad request, unauthorized, etc.)
                raise GeminiAPIError(f"Client error: {e}", retryable=False) from e
            retry_count += 1
            logger.warning(
                "Retrying API call",
                extra={
                    "operation": operation,
                    "retry_count": retry_count,
                    "error": str(e),
                },
            )
            raise

    try:
        response = await _make_request()
        duration_ms = (time.perf_counter() - start_time) * 1000

        _log_metrics(
            APICallMetrics(
                operation=operation,
                model=config.model_name,
                prompt_length=prompt_length,
                image_count=image_count,
                duration_ms=duration_ms,
                success=True,
                retry_count=retry_count,
            )
        )
        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        error_type = type(e).__name__

        _log_metrics(
            APICallMetrics(
                operation=operation,
                model=config.model_name,
                prompt_length=prompt_length,
                image_count=image_count,
                duration_ms=duration_ms,
                success=False,
                retry_count=retry_count,
                error_type=error_type,
            )
        )
        raise


# Initialize FastMCP server with lifespan for graceful shutdown
@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncGenerator[None]:
    """Manage server lifecycle with graceful shutdown."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # Validate configuration on startup
    try:
        config = get_config()
        _ = get_client()  # Ensure client can be created
        logger.info(
            "Banana Appeal server starting",
            extra={"model": config.model_name, "version": "0.1.0"},
        )
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Set up signal handlers for graceful shutdown
    def handle_shutdown(sig: signal.Signals) -> None:
        logger.info(f"Received {sig.name}, initiating graceful shutdown...")
        if _shutdown_event:
            _shutdown_event.set()

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    try:
        yield
    finally:
        logger.info("Banana Appeal server shutting down")
        # Cleanup
        global _client, _config
        _client = None
        _config = None


mcp = FastMCP(
    name="banana-appeal",
    instructions="A-peel-ing image generation using Gemini. "
    "Supports text-to-image generation, image editing, and multi-image blending.",
    version="0.1.0",
    lifespan=lifespan,
)


async def _generate_image_impl(
    prompt: str,
    save_path: str | None = None,
    ctx: Context | None = None,
) -> ImageResult:
    """Internal implementation of generate_image."""
    try:
        request = GenerateImageRequest(prompt=prompt, save_path=save_path)
    except ValueError as e:
        return ImageResult.from_error(f"Invalid request: {e}")

    if ctx:
        await ctx.info(f"Generating image: {request.prompt[:50]}...")

    try:
        response = await _call_gemini_api(request.prompt, operation="generate")
    except GeminiAPIError as e:
        return ImageResult.from_error(str(e))
    except genai_errors.APIError as e:
        return ImageResult.from_error(f"API error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in generate_image")
        return ImageResult.from_error(f"Unexpected error: {e}")

    result = _extract_image_data(response)
    if not result:
        return ImageResult.from_error("No image was generated")

    image_data, fmt = result

    if request.save_path:
        await _save_image_async(request.save_path, image_data)
        if ctx:
            await ctx.info(f"Image saved to {request.save_path}")
        return ImageResult.from_saved(request.save_path, fmt)

    return ImageResult.from_data(image_data, fmt)


async def _edit_image_impl(
    image_path: str,
    edit_prompt: str,
    output_path: str | None = None,
    ctx: Context | None = None,
) -> ImageResult:
    """Internal implementation of edit_image."""
    try:
        request = EditImageRequest(
            image_path=image_path,
            edit_prompt=edit_prompt,
            output_path=output_path,
        )
    except ValueError as e:
        return ImageResult.from_error(f"Invalid request: {e}")

    if ctx:
        await ctx.info(f"Editing image: {request.edit_prompt[:50]}...")

    try:
        image = await _load_image_async(request.image_path)
    except Exception as e:
        return ImageResult.from_error(f"Failed to load image: {e}")

    try:
        response = await _call_gemini_api(
            [request.edit_prompt, image],
            operation="edit",
            image_count=1,
        )
    except GeminiAPIError as e:
        return ImageResult.from_error(str(e))
    except genai_errors.APIError as e:
        return ImageResult.from_error(f"API error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in edit_image")
        return ImageResult.from_error(f"Unexpected error: {e}")

    result = _extract_image_data(response)
    if not result:
        return ImageResult.from_error("Image editing failed - no output generated")

    image_data, fmt = result
    save_to = request.output_path or request.image_path
    await _save_image_async(save_to, image_data)

    if ctx:
        await ctx.info(f"Edited image saved to {save_to}")

    return ImageResult.from_saved(save_to, fmt)


async def _blend_images_impl(
    image_paths: list[str],
    prompt: str,
    output_path: str | None = None,
    ctx: Context | None = None,
) -> ImageResult:
    """Internal implementation of blend_images."""
    try:
        request = BlendImagesRequest(
            image_paths=image_paths,
            prompt=prompt,
            output_path=output_path,
        )
    except ValueError as e:
        return ImageResult.from_error(f"Invalid request: {e}")

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
        return ImageResult.from_error(f"Failed to load images: {', '.join(load_errors)}")

    try:
        response = await _call_gemini_api(
            [request.prompt, *images],
            operation="blend",
            image_count=len(images),
        )
    except GeminiAPIError as e:
        return ImageResult.from_error(str(e))
    except genai_errors.APIError as e:
        return ImageResult.from_error(f"API error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in blend_images")
        return ImageResult.from_error(f"Unexpected error: {e}")

    result = _extract_image_data(response)
    if not result:
        return ImageResult.from_error("Image blending failed - no output generated")

    image_data, fmt = result

    if request.output_path:
        await _save_image_async(request.output_path, image_data)
        if ctx:
            await ctx.info(f"Blended image saved to {request.output_path}")
        return ImageResult.from_saved(request.output_path, fmt)

    return ImageResult.from_data(image_data, fmt)


# MCP tool wrappers - convert internal results to MCP-compatible format


@mcp.tool
async def generate_image(
    prompt: Annotated[str, Field(min_length=1, description="Text description of the image")],
    save_path: Annotated[str | None, Field(description="Optional path to save the image")] = None,
    ctx: Context | None = None,
) -> Image | str:
    """Generate an image from a text prompt using Gemini.

    Returns the generated image, or saves it to the specified path.
    """
    result = await _generate_image_impl(prompt=prompt, save_path=save_path, ctx=ctx)

    if not result.success:
        return f"Error: {result.error}"

    if result.saved_path:
        return f"Image saved to {result.saved_path}"

    if result.data:
        return Image(data=result.data, format=result.format.value)

    return "Error: No image data"


@mcp.tool
async def edit_image(
    image_path: Annotated[str, Field(description="Path to the image to edit")],
    edit_prompt: Annotated[str, Field(min_length=1, description="Edit instructions")],
    output_path: Annotated[
        str | None, Field(description="Output path (default: overwrite original)")
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Edit an existing image using natural language instructions.

    Modifies the image according to the edit prompt and saves the result.
    """
    result = await _edit_image_impl(
        image_path=image_path,
        edit_prompt=edit_prompt,
        output_path=output_path,
        ctx=ctx,
    )

    if not result.success:
        return f"Error: {result.error}"

    return f"Edited image saved to {result.saved_path}"


@mcp.tool
async def blend_images(
    image_paths: Annotated[
        list[str], Field(min_length=2, max_length=14, description="Paths to images to blend")
    ],
    prompt: Annotated[str, Field(min_length=1, description="Blending instructions")],
    output_path: Annotated[str | None, Field(description="Optional output path")] = None,
    ctx: Context | None = None,
) -> Image | str:
    """Blend multiple images together with a creative prompt.

    Combines up to 14 images according to the prompt instructions.
    """
    result = await _blend_images_impl(
        image_paths=image_paths,
        prompt=prompt,
        output_path=output_path,
        ctx=ctx,
    )

    if not result.success:
        return f"Error: {result.error}"

    if result.saved_path:
        return f"Blended image saved to {result.saved_path}"

    if result.data:
        return Image(data=result.data, format=result.format.value)

    return "Error: No image data"


def main() -> None:
    """Entry point for the MCP server."""
    try:
        # Validate configuration before starting
        _ = get_config()
    except ConfigurationError as e:
        logger.error(f"Startup failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mcp.run()


if __name__ == "__main__":
    main()
