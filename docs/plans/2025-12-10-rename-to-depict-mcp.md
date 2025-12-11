# Rename Project: banana-appeal → depict-mcp

**Date:** 2025-12-10
**Status:** Approved, pending implementation

## Summary

Rename the project from `banana-appeal` to `depict-mcp` to follow MCP naming conventions and better describe the project's purpose as a generic image generation MCP server.

## MCP Naming Conventions (Reference)

- Use kebab-case for package names
- Include "mcp" in the name for discoverability
- Two common patterns: `<name>-mcp` (40%) or `mcp-<name>` (35%)
- Max 64 characters
- Avoid "server" in the name

## Beads Migration

**Prefix change:** `banana-appeal-` → `d-`

The `d-` prefix is the shortest valid option (2 chars), saving tokens in every issue reference.

```bash
bd rename-prefix d- --dry-run  # Preview
bd rename-prefix d-            # Apply
```

This updates:
- All 24 issue IDs
- All dependency references
- Text references in descriptions

**Run this first**, before any file renames.

## Source Code Changes

### Directory Rename

```
src/banana_appeal/ → src/depict_mcp/
```

### pyproject.toml

| Field | Old | New |
|-------|-----|-----|
| `name` | `banana-appeal` | `depict-mcp` |
| `project.scripts` | `banana-appeal = "banana_appeal.server:main"` | `depict-mcp = "depict_mcp.server:main"` |
| URLs | `chrisvaillancourt/banana-appeal` | `chrisvaillancourt/depict-mcp` |
| `known-first-party` | `["banana_appeal"]` | `["depict_mcp"]` |
| `source` | `["src/banana_appeal"]` | `["src/depict_mcp"]` |

### Test Files

Update imports in:
- `tests/test_server.py`
- `tests/test_models.py`
- `tests/test_build.py`
- `tests/conftest.py`
- `tests/__init__.py`

Change: `from banana_appeal` → `from depict_mcp`

## Documentation Updates

| File | Changes |
|------|---------|
| `README.md` | Project name, install commands, usage |
| `AGENTS.md` | Project description |
| `CONTRIBUTING.md` | Package references |
| `CHANGELOG.md` | Update header |
| `docs/gemini-capabilities.md` | Package references |

## GitHub Workflows

| File | Changes |
|------|---------|
| `.github/workflows/ci.yml` | Package name |
| `.github/workflows/publish.yml` | Package name, PyPI refs |

## Execution Order

1. `bd rename-prefix d-` (before file changes)
2. `mv src/banana_appeal src/depict_mcp`
3. Update file contents (pyproject.toml, tests, docs, workflows)
4. `uv sync --dev` (regenerate lock file)
5. Verify: `uv run pytest && uv run ruff check . && bd list --status=open`
6. Commit: `git commit -m "chore!: rename project from banana-appeal to depict-mcp"`

## Notes

- The `!` in commit message indicates breaking change
- Users will need to update imports and install commands
- GitHub repository rename is separate (do manually via GitHub settings)
