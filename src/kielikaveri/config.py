from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required to actually run the bot (kielikaveri.bot.main) - left optional
    # here so scripts like import_cards, and tests, don't need a real token.
    bot_token: str = ""
    whitelist_user_ids: str = ""

    openai_api_key: str = ""
    # Strong model, not nano - plan 3.2: nano-tier models don't hold up on
    # Finnish morphology and give a useless error breakdown.
    openai_text_model: str = "gpt-5.6-terra"
    openai_stt_model: str = "whisper-1"
    openai_tts_model: str = "tts-1"
    # tts-1 at 1.0 reads short sentences too fast to follow at B2 level.
    openai_tts_speed: float = 0.8
    # A button the user is waiting on - past this, failing fast beats waiting.
    openai_tts_timeout_seconds: float = 15.0
    openai_timeout_seconds: float = 60.0

    # Plan 3.8 point 3: the in-process circuit breaker. More calls than this
    # within the window always means a bug (a human can't drive it by hand),
    # so it trips. The window is rolling: calls go through again once the
    # old ones age out of it - it caps the rate, not the total.
    breaker_max_calls: int = 60
    breaker_window_minutes: int = 10

    # The background book translator (translate/worker.py) has its own
    # breaker: one chapter is ~15 calls, glossary included, and must neither
    # trip the bot's breaker nor be starved by it. A chapter takes minutes,
    # so 60 calls in 10 minutes still means a retry loop, not reading.
    translate_breaker_max_calls: int = 60
    translate_breaker_window_minutes: int = 10
    # The total the rate window can't cap: rolling 24 hours, ~10 chapters.
    # Past it every chapter fails with "try tomorrow" until calls age out.
    translate_daily_max_calls: int = 150
    # Each SDK retry of a timed-out 300 s request may be billed, and the
    # breaker sees only one check for all of them - fewer than the bot's 3.
    translate_max_retries: int = 1
    # A 6k-char block comes back as ~6k chars of Finnish - minutes, not the
    # 60 s an /add reply gets.
    translate_timeout_seconds: float = 300.0

    # The reader (word lookup, TTS) has its own breaker too (plan
    # 2026-11-20-kirjakaveri-reader, И6): a learner easily taps 60 words in
    # 10 minutes at reading pace, and that must not starve /add and TTS in
    # the bot. 150 in 10 minutes is still faster than anyone reads.
    reader_breaker_max_calls: int = 150
    reader_breaker_window_minutes: int = 10
    # The total the reader's rate window can't cap (lookup and TTS cache
    # misses together, rolling 24 hours) - like translate_daily_max_calls.
    reader_daily_max_calls: int = 400
    # mp3 of tapped words (reader/speech.py), one file per word, model,
    # voice and speed. Relative to the working directory (the repo root
    # under deploy/kielikaveri.service); git-ignored.
    reader_tts_cache_dir: str = "data/tts_cache"
    # Telegram lets bots download up to 20 MB; a long novel is ~1-2 MB.
    max_book_bytes: int = 5_000_000

    # The reader Mini App (plan 2026-11-20-kirjakaveri-reader, "П2"). The
    # aiohttp app listens on localhost only; Caddy in front of it holds the
    # HTTPS certificate. READER_URL is the public https address the menu
    # button opens - empty leaves Telegram's default menu button.
    reader_url: str = ""
    reader_host: str = "127.0.0.1"
    reader_port: int = 8080
    # Telegram signs initData once, when the Mini App opens. Older than this
    # is refused - a day covers a long reading session.
    reader_init_data_max_age_hours: int = 24

    database_url: str = "sqlite+aiosqlite:///./kielikaveri.db"

    log_level: str = "INFO"
    # Relative to the process's working directory (deploy/kielikaveri.service
    # sets that to the repo root) - rotated daily, 14 days kept. Empty
    # disables the file handler and logs to stdout only.
    log_file: str = "logs/kielikaveri.log"

    # Hour (0-23, Europe/Helsinki) a new study day starts for `due`
    # scheduling and the daily new-card limit - see srs/queue.py.
    day_boundary_hour: int = 4

    # /learn session shape.
    session_max_cards: int = 20
    session_max_minutes: int = 10
    # Defensive cap: never-reviewed cards of any type admitted per study day.
    daily_new_limit: int = 10
    # How many *grammar forms* the curriculum may open per study day. Kept
    # separate from daily_new_limit on purpose - a new word and a new case
    # are different units of work, and one noun has a dozen forms waiting
    # (see srs/curriculum.py).
    daily_new_forms: int = 4

    # Above this many overdue cards, /learn offers to postpone the backlog
    # instead of dumping it all into one session.
    debt_threshold: int = 100
    debt_postpone_days: int = 7

    @property
    def whitelist(self) -> set[int]:
        return {int(user_id) for user_id in self.whitelist_user_ids.split(",") if user_id.strip()}


def load_settings() -> Settings:
    return Settings()
