<p align="center">
  <img src="docs/assets/logo.png" alt="Banana Appeal Logo" width="300">
</p>

<h1 align="center">Banana Appeal</h1>

<p align="center">
  <strong>A-peel-ing image generation MCP server powered by Gemini 3 Pro Image</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/banana-appeal/"><img src="https://img.shields.io/pypi/v/banana-appeal?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/banana-appeal/"><img src="https://img.shields.io/pypi/pyversions/banana-appeal" alt="Python versions"></a>
  <a href="https://github.com/chrisvaillancourt/banana-appeal/actions/workflows/ci.yml"><img src="https://github.com/chrisvaillancourt/banana-appeal/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/chrisvaillancourt/banana-appeal/blob/main/LICENSE"><img src="https://img.shields.io/github/license/chrisvaillancourt/banana-appeal" alt="License"></a>
</p>

---

Banana Appeal is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that brings Google's Gemini image generation capabilities to Claude Code and other MCP-compatible tools.

## Features

- **Text-to-image generation** - Generate images from text prompts
- **Image editing** - Modify existing images using natural language instructions
- **Multi-image blending** - Combine up to 14 reference images with creative prompts
- **Automatic retries** - Built-in exponential backoff for API reliability
- **Type-safe** - Full Pydantic validation for all inputs

## Installation

### From PyPI

```bash
pip install banana-appeal
```

### From Source

```bash
git clone https://github.com/chrisvaillancourt/banana-appeal.git
cd banana-appeal
uv sync
```

## Configuration

### API Key

Set the `GOOGLE_API_KEY` environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Claude Code Integration

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "banana-appeal": {
      "type": "stdio",
      "command": "uvx",
      "args": ["banana-appeal"],
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Or if installed locally with uv:

```json
{
  "mcpServers": {
    "banana-appeal": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/banana-appeal", "run", "banana-appeal"],
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Usage

Once configured, Claude Code will have access to these tools:

### generate_image

Generate an image from a text prompt.

```
Generate an image of a sunset over mountains
```

### edit_image

Edit an existing image using natural language.

```
Edit /path/to/image.png: add a rainbow in the sky
```

### blend_images

Blend multiple images with a creative prompt.

```
Blend these images into a single artwork: /path/to/img1.png /path/to/img2.png
```

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/chrisvaillancourt/banana-appeal.git
cd banana-appeal
uv sync --dev
```

### Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/banana_appeal
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run pyrefly check src/banana_appeal/
```

### Development Server

```bash
# Run in dev mode with auto-reload
uv run fastmcp dev src/banana_appeal/server.py
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - The MCP framework powering this server
- [Google Gemini](https://ai.google.dev/) - The AI model behind image generation
- [Anthropic Claude](https://claude.ai/) - For creating MCP and Claude Code
