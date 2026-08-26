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
    openai_timeout_seconds: float = 60.0

    # Plan 3.8 point 3: the in-process circuit breaker. More calls than this
    # within the window always means a bug (a human can't drive it by hand),
    # so it trips and stops hitting the API until restart.
    breaker_max_calls: int = 60
    breaker_window_minutes: int = 10

    database_url: str = "sqlite+aiosqlite:///./kielikaveri.db"

    # Hour (0-23, Europe/Helsinki) a new study day starts for `due`
    # scheduling and the daily new-card limit - see srs/queue.py.
    day_boundary_hour: int = 4

    # /learn session shape.
    session_max_cards: int = 20
    session_max_minutes: int = 10
    daily_new_limit: int = 10

    # Above this many overdue cards, /learn offers to postpone the backlog
    # instead of dumping it all into one session.
    debt_threshold: int = 100
    debt_postpone_days: int = 7

    @property
    def whitelist(self) -> set[int]:
        return {int(user_id) for user_id in self.whitelist_user_ids.split(",") if user_id.strip()}


def load_settings() -> Settings:
    return Settings()
