#!/usr/bin/env python3
"""Launch TPMS Builder.

    python run.py                 open the GUI
    python run.py part.stl        open the GUI with a file loaded
    python run.py setup.json      open the GUI with saved settings

For headless generation see ``python -m tpms.cli --help``.
"""

import sys

from tpms.app import main

if __name__ == "__main__":
    sys.exit(main())
