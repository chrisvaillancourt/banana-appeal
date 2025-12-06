# Contributing to Banana Appeal

Thank you for your interest in contributing to Banana Appeal! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- A Google AI API key (for integration testing)

### Getting Started

1. Clone the repository:

```bash
git clone https://github.com/chrisvaillancourt/banana-appeal.git
cd banana-appeal
```

2. Install dependencies:

```bash
uv sync --dev
```

3. Set up your API key (for integration tests):

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/banana_appeal --cov-report=html

# Run specific test file
uv run pytest tests/test_server.py

# Run only fast tests (skip integration)
uv run pytest -m "not integration"
```

### Code Quality

```bash
# Run linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run pyrefly check src/banana_appeal/
```

### Running the MCP Server Locally

```bash
# Development mode with auto-reload
uv run fastmcp dev src/banana_appeal/server.py

# Production mode
uv run banana-appeal
```

## Issue Tracking

This project uses [beads](https://github.com/steveyegge/beads) for issue tracking.

```bash
bd ready              # See available work
bd create --title="Fix bug" --type=bug
bd update <id> --status=in_progress
bd close <id>
bd sync               # Sync with remote
```

## Commit Guidelines

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature (triggers minor version bump)
- `fix`: A bug fix (triggers patch version bump)
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Breaking Changes

For breaking changes, add `BREAKING CHANGE:` in the commit footer or append `!` after the type:

```
feat!: remove deprecated API endpoint

BREAKING CHANGE: The /v1/old endpoint has been removed.
```

### Examples

```bash
# Feature
git commit -m "feat(server): add support for batch image processing"

# Bug fix
git commit -m "fix(models): correct validation for empty prompts"

# Documentation
git commit -m "docs: update API reference for blend_images"
```

### Using Commitizen

You can use commitizen to help write properly formatted commits:

```bash
uv run cz commit
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes following the coding standards
3. Add or update tests as needed
4. Ensure all tests pass: `uv run pytest`
5. Ensure code quality checks pass: `uv run ruff check . && uv run ruff format --check .`
6. Submit a pull request with a clear description

### PR Title Format

Use the same format as commit messages:

```
feat(server): add batch processing support
```

## Release Process

Releases are automated via GitHub Actions:

1. When commits are merged to `main`, commitizen analyzes commit messages
2. If there are version-bumping commits, it:
   - Updates version in `pyproject.toml` and `__init__.py`
   - Updates `CHANGELOG.md`
   - Creates a git tag
   - Publishes to PyPI

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue or start a discussion on GitHub.
