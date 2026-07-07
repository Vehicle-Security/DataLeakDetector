"""Public import surface for the rewritten DataLeakDetector package.

Only stable entry points are exported here. Keeping this file small makes it
clear that the package root is a contract boundary rather than a second copy of
the stage implementations.
"""

from __future__ import annotations

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
