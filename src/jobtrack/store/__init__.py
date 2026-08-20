"""store module — persistence, linking, and derived status. See CONTRACTS.md §6.

The public surface is :class:`Store` (the only writer of SQLite) plus the two pure
functions in :mod:`jobtrack.store.linker`. Import from here rather than reaching into
``db`` or ``repo``.
"""

from __future__ import annotations

from jobtrack.store.db import DEFAULT_GHOST_AFTER_DAYS, SCHEMA_VERSION, Store
from jobtrack.store.linker import LINK_WINDOW_DAYS, derive_status, match_application

__all__ = [
    "DEFAULT_GHOST_AFTER_DAYS",
    "LINK_WINDOW_DAYS",
    "SCHEMA_VERSION",
    "Store",
    "derive_status",
    "match_application",
]
