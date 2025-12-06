# Agent Guide

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
