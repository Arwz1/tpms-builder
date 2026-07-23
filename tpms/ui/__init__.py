"""Qt user interface.

Imports from ``features`` and ``io``; nothing in those packages imports from here. That
one-way dependency is what lets every feature run headless from ``tpms/cli.py``.
"""

from tpms.ui import theme

__all__ = ["theme"]
