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

## Commit Guidelines

Uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning:
- `feat:` - New feature (minor bump)
- `fix:` - Bug fix (patch bump)
- `feat!:` or `BREAKING CHANGE:` - Breaking change (major bump)

Use `uv run cz commit` for guided commit formatting.


## Issue Tracking

This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs or TodoWrite.

### Session Protocol

**At Session Start:**
1. Check for updates: `bd info --whats-new`
2. Find ready work: `bd ready --json`

**During Work:**
- Create issues for discovered work with `--deps discovered-from:<parent-id>`
- Always include meaningful descriptions explaining "why", "what", and "how"
- Use `--json` flags for programmatic parsing

**At Session End:**
> **CRITICAL**: Run `bd sync` to flush changes and push. Work is not done until synced.

### Commands Reference

```bash
# Discovery
bd ready --json                    # Find unblocked work
bd list --status=open --json       # All open issues
bd show <id> --json                # Issue details
bd stale --days 30 --json          # Find forgotten issues

# Creating issues
bd create "Title" --description="Context" --type=bug|feature|task --json
bd create "Title" --deps discovered-from:<parent-id> --json  # Link to parent

# Issue lifecycle
bd update <id> --status=in_progress --json   # Claim work
bd close <id> --reason="Done" --json         # Complete
bd close <id1> <id2> ...                     # Close multiple

# Dependencies
bd dep add <issue> <depends-on>    # Add dependency
bd dep tree <id>                   # View hierarchy
bd blocked                         # Show blocked issues

# Maintenance
bd sync                            # Force sync (ALWAYS run at session end)
bd duplicates                      # Find duplicates
```

### Issue Types and Priorities

**Types:** `bug`, `feature`, `task`, `epic`, `chore`

**Priorities:**
- `0` - Critical (production down)
- `1` - High (blocking work)
- `2` - Medium (default)
- `3` - Low (nice to have)
- `4` - Backlog (someday)

### Planning Work with Dependencies

#### Cognitive Trap: Temporal Language

Phrases like "Phase 1," "Step 1," or "before" can **invert dependency direction** in your thinking. Your brain interprets temporal sequencing as dependency flow, leading to backwards logic.

**The problem:** Saying "Phase 1 comes before Phase 2" makes you think `bd dep add phase1 phase2`, but this incorrectly states that phase1 depends on phase2.

**The solution:** Use requirement-based language focused on what tasks *need*, not when they occur. Name tasks by their actual function rather than execution order.

#### Dependency Direction

Dependencies express "needs" not "comes before":

```bash
# WRONG (temporal thinking): "Phase 1 comes before Phase 2"
bd dep add phase1 phase2

# CORRECT (requirement thinking): "Phase 2 needs Phase 1"
bd dep add phase2 phase1
```

#### Example: Multi-Task Dependencies

```bash
# Create tasks named by function, not order
bd create "Implement conversation region" -t task -p 1
bd create "Add header-line status display" -t task -p 1
bd create "Render tool calls inline" -t task -p 2

# Establish dependencies based on requirements
bd dep add header-line conversation-region    # header needs conversation region
bd dep add tool-calls conversation-region     # tool calls need conversation region
```

Verify with `bd blocked` - tasks should be blocked by their actual prerequisites.

### Best Practices

1. **Always include descriptions** - Issues without context waste future time
2. **Use `--json` flags** - Essential for programmatic parsing by agents
3. **Run `bd sync` at session end** - Prevents data loss
4. **Check `bd ready` first** - Shows unblocked work automatically
5. **Link discovered issues** - Use `--deps discovered-from:<id>` to track where issues came from

### Managing Planning Documents

Store AI-generated planning materials (PLAN.md, ARCHITECTURE.md, DESIGN.md, etc.) in a `history/` directory rather than the repository root.

```
history/
├── PLAN-feature-x.md
├── ARCHITECTURE-v2.md
└── DESIGN-auth-refactor.md
```

**Benefits:**
- Keeps repository root clean and focused on production code
- Clearly distinguishes ephemeral planning from permanent documentation
- Preserves planning history for reference
- Can be excluded from version control if needed (add `history/` to `.gitignore`)
