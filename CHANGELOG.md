# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-06

### Added

- Initial release of Banana Appeal MCP server
- **Tools:**
  - `generate_image` - Text-to-image generation using Gemini
  - `edit_image` - Edit existing images with natural language instructions
  - `blend_images` - Combine up to 14 reference images with a creative prompt
- **Configuration:**
  - `GOOGLE_API_KEY` - Required API key for Gemini
  - `BANANA_MODEL` - Configurable model name (default: gemini-2.0-flash-preview-image-generation)
  - `BANANA_RETRY_ATTEMPTS` - Number of retry attempts (default: 3)
  - `BANANA_RETRY_TIMEOUT` - Total retry timeout in seconds (default: 60)
  - `BANANA_MAX_PROMPT_LENGTH` - Maximum prompt length (default: 10000)
- **Observability:**
  - Structured logging with operation metrics (duration, retry count, error type)
  - API call tracking for debugging and monitoring
- **Reliability:**
  - Smart retry logic - retries 5xx and 429 errors, fails fast on 4xx client errors
  - Graceful shutdown with SIGTERM/SIGINT signal handling
  - Startup validation - fails fast if GOOGLE_API_KEY is missing
- **Validation:**
  - Pydantic models for all inputs with comprehensive validation
  - `ImageResult` response model with factory methods
  - Whitespace stripping and empty prompt detection
- **Testing:**
  - 46 unit tests covering all tools and edge cases
  - Build verification tests
  - Mocked Gemini API for fast, reliable tests
- **CI/CD:**
  - GitHub Actions workflows for testing, linting, and publishing
  - Automatic PyPI publishing on release
  - Coverage reporting with Codecov
- **Documentation:**
  - README with badges, installation, and usage instructions
  - CONTRIBUTING guide with conventional commits
  - MIT License

### Technical

- Native async Gemini API integration via `client.aio.models.generate_content()`
- Singleton client pattern to avoid connection overhead
- Concurrent image loading with anyio task groups
- FastMCP server with lifespan management
- Full type hints with py.typed marker
- Python 3.12, 3.13, and 3.14 support

### Fixed

- Response parsing for new Gemini SDK structure (`response.candidates[0].content.parts`)
- Removed non-existent `stamina.RetryingExhausted` exception handling
- Environment variable name corrected from `GEMINI_API_KEY` to `GOOGLE_API_KEY`

[Unreleased]: https://github.com/chrisvaillancourt/banana-appeal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chrisvaillancourt/banana-appeal/releases/tag/v0.1.0
