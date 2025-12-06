# Agent Guide

This file provides guidance to any agent (e.g. Claude Code) when working with code in this repository.

## Project Overview

Banana Appeal is an MCP (Model Context Protocol) server that exposes Google Gemini's image generation capabilities to Claude Code. Built with FastMCP and the google-genai SDK.

## Common Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest                              # All tests
uv run pytest tests/test_server.py         # Single file
uv run pytest -m "not integration"         # Skip integration tests

# Code quality
uv run ruff check .                        # Lint
uv run ruff check --fix .                  # Auto-fix
uv run ruff format .                       # Format
uv run pyrefly check src/banana_appeal/    # Type check

# Run server
uv run fastmcp dev src/banana_appeal/server.py  # Dev mode with auto-reload
uv run banana-appeal                             # Production mode
```

## Architecture

The server exposes three MCP tools:
- `generate_image` - Text-to-image generation
- `edit_image` - Modify existing images with natural language
- `blend_images` - Combine up to 14 images with a creative prompt

**Key files:**
- `src/banana_appeal/server.py` - FastMCP server, tool definitions, Gemini API integration with stamina retries
- `src/banana_appeal/models.py` - Pydantic models for validation and configuration

**Configuration:** Loaded from environment variables via `ServerConfig.from_env()`. Required: `GOOGLE_API_KEY`. Optional: `BANANA_MODEL`, `BANANA_RETRY_ATTEMPTS`, `BANANA_RETRY_TIMEOUT`, `BANANA_MAX_PROMPT_LENGTH`.

## Issue Tracking

This project uses [beads](https://github.com/steveyegge/beads) (`bd` command) for issue tracking instead of markdown TODOs.

## Commit Guidelines

Uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning:
- `feat:` - New feature (minor bump)
- `fix:` - Bug fix (patch bump)
- `feat!:` or `BREAKING CHANGE:` - Breaking change (major bump)

Use `uv run cz commit` for guided commit formatting.


## Issue Tracking

This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs or TodoWrite.

```bash
bd ready              # See available work
bd create "Title" --type=task|bug|feature
bd update <id> --status=in_progress
bd close <id>
bd sync               # Sync with remote
```

### Workflow

1. Check ready work: `bd ready`
2. Claim task: `bd update <id> --status=in_progress`
3. Complete work: `bd close <id>`
4. Sync: `bd sync`
