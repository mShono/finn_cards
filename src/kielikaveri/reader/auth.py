"""Who is calling the reader's /api: Telegram's initData, checked against
BOT_TOKEN (https://core.telegram.org/bots/webapps#validating-data-received-via-the-web-app).

The Mini App sends it as `Authorization: tma <initData>` - a header, not the
URL, so it never lands in an access log. The string itself is a credential
for its lifetime and is never logged; only the outcome and the user id are.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aiogram.utils.web_app import safe_parse_webapp_init_data

SCHEME = "tma "


class AuthError(Exception):
    """Why a request was refused - a short reason for the log, no initData."""


def user_id_from_header(
    header: str | None,
    bot_token: str,
    whitelist: set[int],
    max_age: timedelta,
    now: datetime | None = None,
) -> int:
    if not header or not header.startswith(SCHEME):
        raise AuthError("missing")
    try:
        data = safe_parse_webapp_init_data(bot_token, header.removeprefix(SCHEME))
    except ValueError as exc:
        raise AuthError("bad_signature") from exc
    # A valid signature never expires by itself: without this, a leaked
    # initData would open the reader forever.
    now = now or datetime.now(UTC)
    if now - data.auth_date > max_age:
        raise AuthError("expired")
    if data.user is None:
        raise AuthError("no_user")
    if data.user.id not in whitelist:
        raise AuthError("not_whitelisted")
    return data.user.id
