"""Pytest configuration for lakehouse architecture tests.

Puts the MineralVision_Enhanced package root on sys.path so that
`lakehouse_architecture` is importable as a top-level package.
"""

import os
import sys

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
