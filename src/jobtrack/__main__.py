"""Allow ``python -m jobtrack`` alongside the installed ``jobtrack`` script."""

from __future__ import annotations

import sys

from jobtrack.cli import main

if __name__ == "__main__":
    sys.exit(main())
