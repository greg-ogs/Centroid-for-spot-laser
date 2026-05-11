"""Top-level shim — preserves ``python main.py`` for backwards compatibility.

The real implementation lives at :mod:`source.benchmarks.main`. Prefer
``python -m source.benchmarks.main`` going forward.
"""
from source.benchmarks.main import main

if __name__ == "__main__":
    main()
