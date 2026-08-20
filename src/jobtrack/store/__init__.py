"""store module: SQLite persistence, migrations, and message->application linking.

See CONTRACTS.md §6. ``Store`` is the sole writer of SQLite; ``linker`` is pure and DB-free.
"""

from __future__ import annotations

from jobtrack.store.db import SCHEMA_VERSION
from jobtrack.store.linker import LINK_WINDOW_DAYS, derive_status, match_application
from jobtrack.store.repo import Store

__all__ = [
    "LINK_WINDOW_DAYS",
    "SCHEMA_VERSION",
    "Store",
    "derive_status",
    "match_application",
]
