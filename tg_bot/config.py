from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    api_id: int | None
    api_hash: str = field(repr=False)
    gemini_key: str = field(repr=False)
    model: str
    chats: set[int]
    chats_error: bool = False
    control_bot_token: str = field(default="", repr=False)
    auto_reply_enabled: bool = False
    reply_backend: str = "gemini"
    reply_model: str = ""
    style_profile_path: str = "data/style/profile.json"
    owner_name_aliases: tuple[str, ...] = ()
    reply_context_before: int = 5
    reply_context_after: int = 3
    reply_wait_seconds: int = 1
    reply_config_error: str = ""
    auto_reply_stickers_enabled: bool = False
    owner_command_sender_ids: tuple[int, ...] = ()

    @classmethod
    def load(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        try: api_id = int(os.getenv("TELEGRAM_API_ID", "")) or None
        except ValueError: api_id = None
        ids = set()
        chats_error = False
        for v in os.getenv("ALLOWED_CHAT_IDS", "").split(","):
            v = v.strip()
            if v:
                try: ids.add(int(v))
                except ValueError: chats_error = True
        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
        try:
            auto_raw = os.getenv("AUTO_REPLY_ENABLED", "false").strip().casefold()
            if auto_raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
                raise ValueError
            auto_reply = auto_raw in {"1", "true", "yes", "on"}
            sticker_default = "true" if auto_reply else "false"
            sticker_raw = os.getenv(
                "AUTO_REPLY_STICKERS_ENABLED", sticker_default,
            ).strip().casefold()
            if sticker_raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
                raise ValueError
            auto_reply_stickers = sticker_raw in {"1", "true", "yes", "on"}
            backend = os.getenv("REPLY_BACKEND", "gemini").strip().casefold() or "gemini"
            reply_model = os.getenv("REPLY_MODEL", "").strip() or model
            profile_path = os.getenv("STYLE_PROFILE_PATH", "data/style/profile.json").strip()
            aliases = tuple(
                item.strip()[:80] for item in os.getenv("OWNER_NAME_ALIASES", "").split(",")
                if item.strip()
            )[:20]
            owner_command_sender_ids = _parse_owner_command_sender_ids(
                os.getenv("OWNER_COMMAND_SENDER_IDS", ""),
            )
            numeric_settings = {
                "reply_context_before": ("REPLY_CONTEXT_BEFORE", 5, 0, 10),
                "reply_context_after": ("REPLY_CONTEXT_AFTER", 3, 0, 5),
                "reply_wait_seconds": ("REPLY_WAIT_SECONDS", 1, 0, 15),
            }
            numeric = {}
            for field_name, (env_name, default, minimum, maximum) in numeric_settings.items():
                raw = os.getenv(env_name, str(default)).strip()
                value = int(raw)
                if not minimum <= value <= maximum:
                    raise ValueError
                numeric[field_name] = value
            if backend != "gemini" or not profile_path or not reply_model:
                raise ValueError
            config_error = ""
        except (ValueError, TypeError):
            auto_reply = False
            backend = "gemini"
            reply_model = model
            profile_path = "data/style/profile.json"
            aliases = ()
            numeric = {
                "reply_context_before": 5, "reply_context_after": 3,
                "reply_wait_seconds": 1,
            }
            auto_reply_stickers = False
            config_error = "reply_config_invalid"
        return cls(
            api_id, os.getenv("TELEGRAM_API_HASH", "").strip(),
            os.getenv("GEMINI_API_KEY", "").strip(), model, ids, chats_error,
            os.getenv("CONTROL_BOT_TOKEN", "").strip(), auto_reply, backend,
            reply_model, profile_path, aliases, **numeric, reply_config_error=config_error,
            auto_reply_stickers_enabled=auto_reply_stickers,
            owner_command_sender_ids=owner_command_sender_ids,
        )


def _parse_owner_command_sender_ids(raw: str) -> tuple[int, ...]:
    """Parse a fail-closed allowlist of positive Telegram user IDs."""
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return ()
    parsed = []
    try:
        for value in values:
            number = int(value)
            if number <= 0:
                return ()
            parsed.append(number)
    except (TypeError, ValueError):
        return ()
    return tuple(dict.fromkeys(parsed))
