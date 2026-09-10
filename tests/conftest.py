import re


def log_fields(message: str) -> dict[str, str]:
    """Parse an `event=x key=val key2=val2` log message into a dict, event
    included under the "event" key - shared by every test asserting on the
    application's structured log lines instead of matching full strings.
    """
    return dict(re.findall(r"(\S+?)=(\S+)", message))
