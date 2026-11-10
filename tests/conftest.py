"""Pytest configuration and environment fixtures for FinSight test suites."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for test discovery and module imports
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
