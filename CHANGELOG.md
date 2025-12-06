# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-05

### Added

- Initial release of Banana Appeal
- `generate_image` tool for text-to-image generation using Gemini 3 Pro Image
- `edit_image` tool for editing existing images with natural language instructions
- `blend_images` tool for combining up to 14 reference images
- FastMCP server implementation for Claude Code integration
- Automatic retry handling with exponential backoff via stamina
- Pydantic validation for all tool inputs
- Comprehensive test suite with pytest
- GitHub Actions CI/CD workflows
- Semantic versioning with commitizen

### Technical

- Async implementation using anyio for concurrent image loading
- Support for Python 3.12, 3.13, and 3.14
- Full type hints throughout the codebase

[Unreleased]: https://github.com/chrisvaillancourt/banana-appeal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chrisvaillancourt/banana-appeal/releases/tag/v0.1.0
