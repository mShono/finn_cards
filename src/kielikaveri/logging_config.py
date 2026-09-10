"""Root logging setup - console (for `journalctl`) plus a rotated file.

Split into two handlers with different formats because journald already
timestamps every line it receives: repeating %(asctime)s in the console
format double-stamps each line and buries the short, actually useful part
(cache hits, token usage, ingest results) under two timestamps. The file
handler has no such prefix, so it keeps the full timestamp.
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Import alone installs the trace_id LogRecordFactory (see module docstring
# there) - every record from here on carries %(trace_id)s, which the
# formatters below reference directly.
from kielikaveri import logging_context  # noqa: F401

# httpx/openai log one INFO line per HTTP call ("HTTP Request: POST ...
# 200 OK") that drowns out the app's own summaries. Quiet unless someone
# is actually debugging at DEBUG level. "httpx2" (not "httpx") is not a
# typo - openai>=3 vendors its own httpx2 module/logger internally; the
# real "httpx" package is still a dependency but unused by the SDK itself.
_NOISY_LOGGERS = ("httpx", "httpx2", "openai", "aiogram")


def setup_logging(log_level: str = "INFO", log_file: str = "") -> None:
    root = logging.getLogger()
    root.setLevel(log_level)
    # Idempotent for tests/reloads - otherwise re-running setup_logging
    # stacks duplicate handlers and every line prints twice.
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: trace=%(trace_id)s %(message)s")
    )
    root.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            path, when="midnight", backupCount=14, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: trace=%(trace_id)s %(message)s")
        )
        root.addHandler(file_handler)

    if log_level != "DEBUG":
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
