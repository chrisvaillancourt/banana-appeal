# Python 3.14 Documentation & Resources for Claude Code

### Core Official Documentation

| Resource | URL |
|----------|-----|
| **What's New in Python 3.14** | https://docs.python.org/3/whatsnew/3.14.html |
| **Python 3.14 Documentation Home** | https://docs.python.org/3.14/ |
| **Deprecations Index** | https://docs.python.org/3/deprecations/index.html |

---

### Major New Features - PEPs & Docs

#### Template Strings (t-strings) - PEP 750
| Resource | URL |
|----------|-----|
| PEP 750 Specification | https://peps.python.org/pep-0750/ |
| `string.templatelib` module docs | https://docs.python.org/3/library/string.templatelib.html |
| PEP 750 Examples Repository | https://github.com/t-strings/pep750-examples |
| PEP 787 (t-strings in subprocess) | https://peps.python.org/pep-0787/ |

#### Deferred Annotations - PEP 649/749
| Resource | URL |
|----------|-----|
| PEP 649 (Deferred Evaluation) | https://peps.python.org/pep-0649/ |
| PEP 749 (Implementation Details) | https://peps.python.org/pep-0749/ |
| `annotationlib` module docs | https://docs.python.org/3/library/annotationlib.html |

#### Subinterpreters - PEP 734
| Resource | URL |
|----------|-----|
| PEP 734 Specification | https://peps.python.org/pep-0734/ |
| `concurrent.interpreters` module docs | https://docs.python.org/3/library/concurrent.interpreters.html |
| PEP 554 (Background/History) | https://peps.python.org/pep-0554/ |

#### Zstandard Compression - PEP 784
| Resource | URL |
|----------|-----|
| PEP 784 Specification | https://peps.python.org/pep-0784/ |
| `compression.zstd` module docs | https://docs.python.org/3/library/compression.zstd.html |
| Backport for older Python | https://github.com/rogdham/backports.zstd |

#### Free-threaded Python (No GIL) - PEP 779
| Resource | URL |
|----------|-----|
| PEP 779 (Criteria for Support) | https://peps.python.org/pep-0779/ |
| PEP 703 (Making GIL Optional) | https://peps.python.org/pep-0703/ |
| Free-Threading Guide | https://py-free-threading.github.io/ |

#### Other Syntax/Language Changes
| Resource | URL |
|----------|-----|
| PEP 758 (except without parentheses) | https://peps.python.org/pep-0758/ |
| PEP 762 (New REPL) | https://peps.python.org/pep-0762/ |

---

### Tutorials & Deep Dives

| Resource | URL |
|----------|-----|
| Real Python: Python 3.14 New Features | https://realpython.com/python314-new-features/ |
| Better Stack: What's New in 3.14 | https://betterstack.com/community/guides/scaling-python/python-3-14-new-features/ |
| Astral Blog: Python 3.14 (Ruff/uv) | https://astral.sh/blog/python-3.14 |

---

### Quick Reference Summary

**Key Features to Highlight:**
1. **t-strings** (`t"Hello {name}"`) - Template strings for safe interpolation
2. **Deferred annotations** - No more quoting forward references
3. **`concurrent.interpreters`** - Subinterpreters in stdlib for parallelism
4. **`compression.zstd`** - Zstandard compression built-in
5. **Free-threaded builds** - Officially supported (no GIL, opt-in)
6. **PyREPL** - Syntax highlighting, autocompletion in the REPL
7. **PEP 758** - `except ValueError, TypeError:` without parentheses
8. **`InterpreterPoolExecutor`** - New executor in `concurrent.futures`
9. **Incremental GC** - Smoother garbage collection for lower latency
10. **Improved error messages** - More helpful syntax and runtime errors
