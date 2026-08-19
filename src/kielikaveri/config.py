from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required to actually run the bot (kielikaveri.bot.main) - left optional
    # here so scripts like import_cards, and tests, don't need a real token.
    bot_token: str = ""
    whitelist_user_ids: str = ""

    openai_api_key: str = ""
    openai_text_model: str = "gpt-4o-mini"
    openai_stt_model: str = "whisper-1"
    openai_tts_model: str = "tts-1"

    database_url: str = "sqlite+aiosqlite:///./kielikaveri.db"

    # Hour (0-23, local server time) a new study day starts for `due`
    # scheduling - used from phase 2 onward.
    day_boundary_hour: int = 4

    @property
    def whitelist(self) -> set[int]:
        return {int(user_id) for user_id in self.whitelist_user_ids.split(",") if user_id.strip()}


def load_settings() -> Settings:
    return Settings()
