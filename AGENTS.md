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

**Gemini capabilities:** See [docs/gemini-capabilities.md](docs/gemini-capabilities.md) for detailed documentation on Gemini's image generation features, prompting strategies, and coding workflow use cases.

## Commit Guidelines

Uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning:
- `feat:` - New feature (minor bump)
- `fix:` - Bug fix (patch bump)
- `feat!:` or `BREAKING CHANGE:` - Breaking change (major bump)

Use `uv run cz commit` for guided commit formatting.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
