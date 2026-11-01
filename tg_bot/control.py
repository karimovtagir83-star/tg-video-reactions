"""Optional private Telegram controls for the group-reply worker."""

import asyncio
import json
from urllib.request import Request, urlopen

from telethon import TelegramClient, events
from telethon.sessions import MemorySession


COMMANDS = (
    ("start", "Открыть пульт"),
    ("commands", "Показать все команды"),
    ("status", "Показать режимы бота"),
    ("talk_on", "Включить обычные ответы"),
    ("talk_off", "Выключить обычные ответы"),
    ("chat_on", "Синоним команды talk_on"),
    ("chat_off", "Синоним команды talk_off"),
    ("links_on", "Включить ответы на ссылки"),
    ("links_off", "Выключить ответы на ссылки"),
    ("pause", "Поставить оба режима на паузу"),
    ("resume", "Снять общую паузу"),
)


def _register_command_menu(token, owner_id):
    """Publish private-chat slash suggestions without exposing the bot token."""
    payload = {
        "commands": [
            {"command": command, "description": description}
            for command, description in COMMANDS
        ],
        "scope": {"type": "chat", "chat_id": int(owner_id)},
        "language_code": "ru",
    }
    request = Request(
        f"https://api.telegram.org/bot{token}/setMyCommands",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
    except Exception:
        return False
    return isinstance(result, dict) and result.get("ok") is True


async def register_command_menu(token, owner_id):
    """Update Telegram's slash menu; a failure does not stop the control bot."""
    if (not isinstance(token, str) or not token.strip()
            or not isinstance(owner_id, int) or owner_id <= 0):
        return False
    return await asyncio.to_thread(_register_command_menu, token.strip(), owner_id)


def create_client(cfg):
    """Create an in-memory bot session; the BotFather token is supplied at start."""
    return TelegramClient(MemorySession(), cfg.api_id, cfg.api_hash, catch_up=False)


def install_handler(
    bot, state, owner_id, send_lock, on_pause=None, status_provider=None,
    on_ordinary_toggle=None, ordinary_available=True,
):
    """Install owner-only private commands and return the async event handler."""

    async def handle(event):
        if not getattr(event, "is_private", False) or event.sender_id != owner_id:
            return

        raw_text = getattr(event, "raw_text", "") or ""
        command = raw_text.strip().split(maxsplit=1)[0].split("@", 1)[0].lower() if raw_text.strip() else ""
        talk_on = {"/talk_on", "/chat_on"}
        talk_off = {"/talk_off", "/chat_off"}
        links_on = {"/links_on"}
        links_off = {"/links_off"}
        if command not in (
            {"/start", "/commands", "/pause", "/resume", "/status"}
            | talk_on | talk_off | links_on | links_off
        ):
            return

        if command in talk_on | talk_off:
            if command in talk_on and not ordinary_available:
                await event.respond(
                    "Обычное общение отключено настройкой сервера; режим ссылок не менялся."
                )
                return
            enabled = command in talk_on
            async with send_lock:
                was_enabled, _ = state.ordinary_mode_snapshot(
                    default_enabled=ordinary_available,
                )
                state.set_ordinary_enabled(
                    enabled, default_enabled=ordinary_available,
                )
            if was_enabled != enabled and on_ordinary_toggle is not None:
                try:
                    on_ordinary_toggle(enabled)
                except Exception:
                    pass
            action = "включён" if enabled else "выключен"
            await event.respond(
                f"Режим обычного общения {action}. Режим ответов на ссылки не менялся."
            )
            return

        if command in links_on | links_off:
            enabled = command in links_on
            async with send_lock:
                state.set_links_enabled(enabled, default_enabled=True)
            action = "включены" if enabled else "выключены"
            await event.respond(
                f"Ответы на ссылки TikTok, Reels и Shorts {action}. "
                "Режим обычного общения не менялся."
            )
            return

        if command == "/pause":
            # Serialize with Telegram sends: once /pause is acknowledged, no
            # subsequent send can pass the paused/generation guard.
            async with send_lock:
                paused, _ = state.set_paused(True)
            if on_pause is not None:
                try:
                    on_pause()
                except Exception:
                    pass
            await event.respond("Пауза включена. Новые задачи не обрабатываются.")
            return

        if command == "/resume":
            async with send_lock:
                was_paused, _ = state.pause_snapshot()
                state.set_paused(False)
            if was_paused:
                await event.respond(
                    "Общая пауза снята. Бот снова работает в прежних режимах; "
                    "обычное общение и ответы на ссылки не переключались. "
                    "Старые задачи из очереди не отправляются."
                )
            else:
                await event.respond(
                    "Общая пауза уже выключена. Режимы общения и ссылок не менялись."
                )
            return

        if command == "/commands":
            lines = ["Команды пульта:"]
            lines.extend(f"/{name} — {description}" for name, description in COMMANDS)
            await event.respond("\n".join(lines))
            return

        paused, _ = state.pause_snapshot()
        status = "на паузе" if paused else "работает"
        ordinary_enabled, _ = state.ordinary_mode_snapshot(
            default_enabled=ordinary_available,
        )
        links_enabled, _ = state.link_mode_snapshot(default_enabled=True)
        if not ordinary_available:
            ordinary_status = "выключено настройкой сервера"
        else:
            ordinary_status = "включено" if ordinary_enabled else "выключено"
        detail = (
            f"Обычное общение: {ordinary_status}; "
            f"ответы на ссылки TikTok, Reels и Shorts: "
            f"{'включены' if links_enabled else 'выключены'}; "
            f"общая пауза: {'включена' if paused else 'выключена'}."
        )
        if status_provider is not None:
            try:
                detail = str(status_provider())[:240]
            except Exception:
                detail = ""
        if command == "/start":
            await event.respond(
                f"Пульт управления: {status}. {detail} "
                "Набери /, чтобы выбрать команду, или отправь /commands для полного списка."
            )
        else:
            await event.respond(f"Статус: {status}. {detail}".strip())

    bot.add_event_handler(handle, events.NewMessage(incoming=True))
    return handle
