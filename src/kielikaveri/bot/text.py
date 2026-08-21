from __future__ import annotations

TELEGRAM_MESSAGE_LIMIT = 4096


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split `text` into chunks Telegram will accept as separate messages.

    Prefers to break on the last newline before `limit` so paragraphs and
    list items stay whole; falls back to a hard cut only when a single
    line is itself longer than `limit`.
    """
    if not text:
        return []

    chunks: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)
    return chunks
