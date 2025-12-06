# Beads Setup and Configuration Guide

This document covers the setup, configuration, and usage of [beads](https://github.com/steveyegge/beads) (`bd` command) for issue tracking in this project.

## Overview

Beads is an AI-supervised issue tracker that stores issues in git-tracked JSONL files. It's designed to work seamlessly with AI coding assistants like Claude Code, keeping issue state in sync with your codebase.

### How It Works

1. Issues are stored in a SQLite database (`.beads/*.db`)
2. Changes automatically export to `.beads/issues.jsonl` (git-tracked)
3. Git hooks ensure JSONL stays in sync with commits
4. A background daemon handles real-time synchronization

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   bd commands   │────▶│  SQLite Database │────▶│  issues.jsonl   │
│  (create, etc)  │     │   (.beads/*.db)  │     │  (git-tracked)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               ▲                        │
                               │                        ▼
                        ┌──────────────┐         ┌─────────────┐
                        │    Daemon    │         │  Git Hooks  │
                        │ (background) │         │ (pre-commit)│
                        └──────────────┘         └─────────────┘
```

## Installation

```bash
# macOS
brew install steveyegge/tap/bd

# Or via curl
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/install.sh | bash
```

### Initialize in a Project

```bash
cd your-project
bd init
```

This creates:
- `.beads/` directory with database and config
- Git hooks for synchronization

## Daemon Configuration

The daemon runs in the background to handle automatic synchronization.

### Daemon Modes

| Mode | Sync Latency | CPU Usage | Best For |
|------|--------------|-----------|----------|
| **Polling** (default) | ~5 seconds | Higher | Compatibility |
| **Event-driven** | < 500ms | ~60% less | Performance |

Event-driven mode uses platform-native file watching (FSEvents on macOS, inotify on Linux).

### Starting the Daemon

```bash
# Basic (polling mode)
bd daemon --start

# Event-driven mode (recommended for local dev)
BEADS_DAEMON_MODE=events bd daemon --start

# With auto-commit and auto-push
BEADS_DAEMON_MODE=events bd daemon --start --auto-commit --auto-push
```

### Daemon Flags

| Flag | Purpose |
|------|---------|
| `--auto-commit` | Automatically commit beads changes |
| `--auto-push` | Automatically push commits |
| `--interval` | Sync check interval (default: 5s) |
| `--foreground` | Run in foreground (for systemd) |

### When to Avoid Event-Driven Mode

- Network filesystems (NFS, SMB)
- Container environments
- Resource-constrained systems

Set `BEADS_WATCHER_FALLBACK=true` (default) to fall back to polling if file watching fails.

## Commit Workflows

### Option 1: Sync Branch (Team Projects)

Best for teams where issue changes should be reviewed or kept separate from code.

**Configuration** (`.beads/config.yaml`):
```yaml
sync-branch: "beads-sync"
```

**How it works:**
1. Issue changes commit to `beads-sync` branch
2. Daemon pushes `beads-sync` to remote
3. Merge to main via PR or `bd sync`

**Pros:**
- Issue changes can be code-reviewed
- Avoids conflicts in multi-developer environments
- Clear separation of concerns

**Cons:**
- Extra branch to manage
- Requires manual merge or `bd sync`

### Option 2: Direct Commit (Solo Projects)

Best for solo developers who want simplicity.

**Configuration** (`.beads/config.yaml`):
```yaml
# sync-branch: "beads-sync"  # Commented out or removed
```

**How it works:**
1. Pre-commit hook auto-stages `.beads/*.jsonl` files
2. Issue changes commit alongside your code changes
3. Single branch, single workflow

**Pros:**
- Simpler workflow
- Issue changes always in sync with code
- No extra branches

**Cons:**
- Less control over when issue changes are committed
- Not ideal for teams

### Switching Between Workflows

To switch from sync-branch to direct commit:
```bash
# Edit .beads/config.yaml and comment out sync-branch
# Restart daemon
bd daemon --stop
bd daemon --start
```

To switch to sync-branch:
```bash
# Edit .beads/config.yaml and set sync-branch
bd daemon --stop
bd daemon --start
```

## Git Hooks

Beads installs three git hooks:

| Hook | Purpose |
|------|---------|
| **pre-commit** | Flushes pending changes to JSONL, auto-stages beads files (if no sync-branch) |
| **post-merge** | Imports JSONL changes after `git pull` or merge |
| **pre-push** | Exports database to JSONL before push |

### Pre-Commit Hook Behavior

- **With sync-branch**: Skips auto-staging (changes go to sync branch)
- **Without sync-branch**: Auto-stages all `.beads/*.jsonl` files

Verify hooks are installed:
```bash
bd doctor
```

## Claude Code Integration

### CLI + Hooks (Recommended)

This is the default mode. Claude Code uses `bd` commands directly.

The session start hook (`bd prime`) injects workflow context into Claude's system prompt.

### Plugin (Optional)

Provides `/bd-*` slash commands in Claude Code.

```bash
# Install plugin
/plugin marketplace add steveyegge/beads
/plugin install beads
```

Slash commands:
- `/bd-ready` - Find tasks with no blockers
- `/bd-create` - Create new issues
- `/bd-update` - Update issue status
- `/bd-close` - Close issues
- `/bd-show` - Show issue details

## Common Commands

### Finding Work

```bash
bd ready              # Issues with no blockers
bd list --status=open # All open issues
bd list --status=in_progress
bd show <id>          # Detailed view
```

### Managing Issues

```bash
bd create --title="Fix bug" --type=bug
bd update <id> --status=in_progress
bd close <id>
bd close <id> --reason="explanation"
```

### Dependencies

```bash
bd dep add <issue> <depends-on>  # Add dependency
bd blocked                        # Show blocked issues
```

### Sync and Maintenance

```bash
bd sync            # Force sync with remote
bd sync --status   # Check sync status
bd doctor          # Check for issues
bd doctor --fix    # Auto-fix issues
```

## Configuration Reference

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BEADS_DAEMON_MODE` | `poll` or `events` | `poll` |
| `BEADS_AUTO_START_DAEMON` | Auto-start daemon | `true` |
| `BEADS_NO_DAEMON` | Disable daemon | `false` |
| `BEADS_SYNC_BRANCH` | Override sync branch | - |
| `BEADS_WATCHER_FALLBACK` | Fall back to polling | `true` |

### Config File (`.beads/config.yaml`)

Key settings:
```yaml
# Sync branch for team collaboration
sync-branch: "beads-sync"

# Issue prefix (auto-detected from directory name)
# issue-prefix: "myproject"

# Disable daemon
# no-daemon: false
```

### Database Config

View/set via CLI:
```bash
bd config list
bd config set <key> <value>
bd config get <key>
```

## Troubleshooting

### Check Health

```bash
bd doctor
```

### Common Issues

**Daemon not running:**
```bash
bd daemon --start
```

**Stale daemon after upgrade:**
```bash
bd daemons killall
bd daemon --start
```

**Changes not committing:**
1. Check if sync-branch is configured
2. Verify git hooks are installed: `bd doctor`
3. Check daemon status: `bd daemon --status`

**JSONL out of sync:**
```bash
bd sync --flush-only  # Force export to JSONL
```

## Project-Specific Configuration

This project uses:

| Setting | Value | Reason |
|---------|-------|--------|
| Daemon mode | Event-driven | Fast sync, lower CPU |
| Sync branch | Disabled | Solo project, simpler workflow |
| Auto-commit | Via pre-commit hook | Changes commit with code |

To replicate this setup:
```bash
# Comment out sync-branch in .beads/config.yaml
BEADS_DAEMON_MODE=events bd daemon --start
```

## Changing Configuration Later

### Adding Team Members

If the project grows to multiple developers:
1. Enable sync-branch in `.beads/config.yaml`
2. Restart daemon
3. Team members clone and run `bd doctor` to verify setup

### Switching Daemon Modes

```bash
bd daemon --stop
BEADS_DAEMON_MODE=poll bd daemon --start  # or events
```

### Disabling Auto-Push

If you want auto-commit but manual push control:
```bash
bd daemon --stop
BEADS_DAEMON_MODE=events bd daemon --start --auto-commit
# Note: no --auto-push
```

## Resources

- [Beads GitHub](https://github.com/steveyegge/beads)
- [Full Documentation](https://github.com/steveyegge/beads/tree/main/docs)
- [Daemon Guide](https://github.com/steveyegge/beads/blob/main/docs/DAEMON.md)
- [Claude Integration](https://github.com/steveyegge/beads/blob/main/docs/CLAUDE_INTEGRATION.md)
