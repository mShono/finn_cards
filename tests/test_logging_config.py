import logging

import pytest

from kielikaveri.logging_config import setup_logging


@pytest.fixture(autouse=True)
def restore_logging_state():
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    noisy_levels_before = {
        name: logging.getLogger(name).level for name in ("httpx", "httpx2", "openai", "aiogram")
    }
    yield
    root.handlers[:] = handlers_before
    root.setLevel(level_before)
    for name, level in noisy_levels_before.items():
        logging.getLogger(name).setLevel(level)


def test_creates_log_file_and_parent_dirs(tmp_path):
    log_file = tmp_path / "nested" / "kielikaveri.log"

    setup_logging("INFO", str(log_file))

    assert log_file.exists()


def test_without_log_file_only_console_handler():
    setup_logging("INFO", "")

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_root_level_applied(tmp_path):
    setup_logging("WARNING", str(tmp_path / "kielikaveri.log"))

    assert logging.getLogger().level == logging.WARNING


def test_info_level_quiets_noisy_loggers(tmp_path):
    setup_logging("INFO", str(tmp_path / "kielikaveri.log"))

    assert logging.getLogger("httpx").level == logging.WARNING
    # openai>=3 vendors its own httpx2, not the real httpx package - missing
    # this one from _NOISY_LOGGERS means every OpenAI HTTP call logs an INFO
    # line in production regardless of level (found while auditing the
    # logging architecture, 10.09.2026).
    assert logging.getLogger("httpx2").level == logging.WARNING
    assert logging.getLogger("openai").level == logging.WARNING
    assert logging.getLogger("aiogram").level == logging.WARNING


def test_debug_level_leaves_noisy_loggers_alone(tmp_path):
    logging.getLogger("httpx").setLevel(logging.NOTSET)

    setup_logging("DEBUG", str(tmp_path / "kielikaveri.log"))

    assert logging.getLogger("httpx").level == logging.NOTSET


def test_rerunning_setup_does_not_stack_handlers(tmp_path):
    log_file = str(tmp_path / "kielikaveri.log")

    setup_logging("INFO", log_file)
    setup_logging("INFO", log_file)

    assert len(logging.getLogger().handlers) == 2
