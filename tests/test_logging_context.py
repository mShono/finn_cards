import logging

from kielikaveri.logging_context import new_trace_id, trace_id_var


def test_new_trace_id_are_distinct():
    assert new_trace_id() != new_trace_id()


def test_default_trace_id_outside_any_request():
    assert trace_id_var.get() == "-"


def test_log_records_carry_current_trace_id():
    logger = logging.getLogger("kielikaveri.test.logging_context")
    token = trace_id_var.set("abc123ff")
    try:
        record = logger.makeRecord("x", logging.INFO, __file__, 1, "msg", (), None)
        assert record.trace_id == "abc123ff"
    finally:
        trace_id_var.reset(token)


def test_log_records_reflect_trace_id_changes_between_records():
    logger = logging.getLogger("kielikaveri.test.logging_context")

    token1 = trace_id_var.set("trace-one")
    record1 = logger.makeRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    trace_id_var.reset(token1)

    token2 = trace_id_var.set("trace-two")
    record2 = logger.makeRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    trace_id_var.reset(token2)

    assert record1.trace_id == "trace-one"
    assert record2.trace_id == "trace-two"


def test_log_records_default_to_dash_with_no_trace_set():
    logger = logging.getLogger("kielikaveri.test.logging_context")
    assert trace_id_var.get() == "-"
    record = logger.makeRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    assert record.trace_id == "-"
