"""Per-update trace id, threaded through logging via contextvars.

aiogram runs each incoming Update in its own asyncio task, so a ContextVar
set at the top of that task (bot/middleware.py's RequestLoggingMiddleware)
stays correct across every awaited call inside it - LLM requests, FST
lookups, DB writes - without passing trace_id through every function
signature along the way.

Injected via a LogRecordFactory, not a logging.Filter: a Filter only runs on
the handlers it's attached to, so anything that reads LogRecords a different
way (pytest's caplog fixture attaches its own capture handler and never sees
handler-level filters) would see records with no trace_id at all. A record
factory runs once at record creation, before any handler or filter, so
every record - ours, a library's, or one caplog captures in a test - carries
%(trace_id)s.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:8]


_base_factory = logging.getLogRecordFactory()


def _trace_id_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    record = _base_factory(*args, **kwargs)
    record.trace_id = trace_id_var.get()
    return record


# Installed once at import time (idempotent - re-installing the same factory
# is harmless) so it's active for every entry point that ever imports this
# module, tests included, regardless of whether setup_logging() has run yet.
logging.setLogRecordFactory(_trace_id_record_factory)
