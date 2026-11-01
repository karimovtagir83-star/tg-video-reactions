import asyncio
import fcntl
import inspect
import math
import os
import random  # Kept as a legacy test and plugin injection point; no sticker lottery.
import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from telethon import events
from telethon.tl.types import MessageEntityMention, User

from .telegram import ROOT, client
from .links import extract_urls, supported_url, has_url_candidate
from .metadata import fetch, MetadataError
from .state import State
from .ai import draft, laugh_only
from .control import (
    create_client as create_control_client,
    install_handler as install_control_handler,
    register_command_menu,
)
from .context import (
    CONTEXT_TTL_SECONDS, ContextBuffer, MessageSnapshot,
    sanitize_window, snapshot_from_message,
)
from .triggers import explicit_triggers, is_addressed_identity_question, thread_key
from .style import load_private_profile
from .reply_ai import (
    format_safe_fallback, generate_reply, REPLY_VALIDATION_FAILURE_CODES,
)
from .voice import load_voice_config
from .stickers import (
    StickerPackCatalog as RecentStickerCatalog, sticker_document_key,
)
from .participants import (
    begins_with_participant_address,
    combine_participant_artifacts, load_private_participant_summaries,
    match_participant_question,
    participant_label_for_reference, participant_label_for_sender,
    participant_reference_is_ambiguous, participant_reference_is_known,
    participant_reference_matches_sender,
    participant_profiles_path,
    participant_summary_for_sender,
    participant_summary_path,
)

SAFE_ERROR_CODES = {
    "metadata_unavailable", "metadata_timeout", "metadata_too_large",
    "redirect_unsupported", "unsupported_media", "ai_unavailable",
    "gemini_key_missing",
}
ORDINARY_DEADLINE_SECONDS = 60
# Allow one validator repair to finish within Gemini's 30-second HTTP timeout.
ORDINARY_GENERATION_TIMEOUT = 35
ORDINARY_MODEL_CONCURRENCY = 3
ORDINARY_LOOKUP_TIMEOUT = 10
FOLLOWUP_IDLE_TTL_SECONDS = 10 * 60
# Retained as a legacy settings window; stickers are no longer gated by it.
STICKER_COOLDOWN_SECONDS = 15 * 60
CONTEXT_PRUNE_INTERVAL_SECONDS = 15 * 60
PARTICIPANT_ARTIFACT_CACHE_SECONDS = 60
PARTICIPANT_ARTIFACT_NEGATIVE_CACHE_SECONDS = 5
ORDINARY_FALLBACK = "Не понял, о чём речь. Уточни?"
_NEEDS_OWNER_FALLBACKS = (
    "Тут лучше уточнить детали перед ответом.",
    "Сначала нужны точные детали, а то можно неверно понять.",
    "Тут не буду решать наугад; уточни, что именно имеется в виду.",
    "Нужен контекст поточнее, чтобы не ошибиться.",
    "Тут лучше остановиться и уточнить факты.",
)
ORDINARY_FALLBACK_BY_REASON = {
    "context_unclear": "Не понял, что ты имеешь в виду. Уточни вопрос?",
    "needs_owner": _NEEDS_OWNER_FALLBACKS[0],
    "unsuitable_content": "Не буду додумывать ответ по этому сообщению. Уточни, что именно обсудить?",
    "unsafe_explosives": "С изготовлением взрывных устройств не помогу. Могу объяснить, что делать при обнаружении подозрительного предмета.",
}
ORDINARY_FALLBACK_REASON_CODES = frozenset({
    "profile_unavailable", "reply_config_invalid", "context_unavailable",
    "context_private_data", "profile_or_context_unavailable", "context_too_large",
    "needs_owner", "context_unclear", "unsuitable_content",
    "unsafe_explosives",
    "reply_model_unavailable", "reply_invalid_schema", "reply_validation_failed",
    "reply_generation_timeout", "reply_generation_error", "reply_invalid_result",
    "reply_timeout",
}) | frozenset(
    f"reply_validation_failed_{code}"
    for code in REPLY_VALIDATION_FAILURE_CODES
)
_OWNER_ACTION_CLARIFICATION = (
    "Кому ответить? Пришли команду ответом на его сообщение или на мой ответ ему."
)
_ACTION_STOP_RE = re.compile(
    r"(?iu)^\s*(?:"
    r"(?:ну\s+)?(?:стоп|отмена|пауза|подожди|погоди)(?:$|[,!.?…])|"
    r"(?:отмени|отменяю|останови(?:сь)?|приостанови|передумал[а]?)(?:\b|$)|"
    r"(?:нет[, ]+)?(?:лучше\s+)?(?:перепиши|исправь|поправь|переформулируй|"
    r"переделай|поменяй|замени|измени|напиши\s+иначе|ответь\s+иначе|"
    r"ответь\s+(?:короче|кратко)|скажи\s+короче|"
    r"сделай\s+(?:покороче|короче|иначе))\b|"
    r"(?:пожалуйста[, ]+)?(?:пока\s+)?не\s+(?:надо\s+)?"
    r"(?:отправляй|отправлять|пиши|писать|отвечай|отвечать)\b|"
    r"не\s+(?:надо|нужно)\s*[.!?…]*$"
    r")"
)
_ACTION_META_RESPONSE_RE = re.compile(
    r"(?iu)^\s*(?:(?:да|ага|ок(?:ей)?|хорошо|конечно|понял[а]?|"
    r"без\s+проблем|давай)[,!\s]+)?(?:я\s+)?"
    r"(?:(?:обязательно|сейчас|щас|позже|потом|завтра|скоро|тогда|"
    r"ему|ей|им)\s+){0,3}"
    r"(?:напишу|отвечу|скажу|спрошу|передам|перешлю)"
    r"(?:\s+(?:ему|ей|им|тебе|вам|сейчас|завтра|позже|потом|скоро|"
    r"тогда|сам|сама|сами))*"
    r"[.!…]*\s*$"
)
_ACTION_META_REFUSAL_RE = re.compile(
    r"(?iu)^\s*(?:извини[, ]+)?(?:я\s+)?(?:не\s+(?:могу|буду|хочу|"
    r"стану)\s+(?:ему\s+|ей\s+|им\s+|тебе\s+|вам\s+)?"
    r"(?:ответить|отвечать|написать|писать|сказать|спросить|отправить)|"
    r"не\s+знаю[, ]+(?:что\s+)?(?:ему|ей|им)\s+"
    r"(?:ответить|написать|сказать|спросить))\b"
)
_ACTION_META_REDIRECT_RE = re.compile(
    r"(?iu)^\s*(?:(?:нет|ну|может)[, ]+)?(?:"
    r"(?:сам|сама|сами|самостоятельно)\s+"
    r"(?:ему|ей|им|его|её|ее|их|у\s+него|у\s+неё|у\s+нее)\s+"
    r"(?:напиши(?:те)?|пиши(?:те)?|ответь(?:те)?|отвечай(?:те)?|"
    r"спроси(?:те)?|спрашивай(?:те)?|скажи(?:те)?)|"
    r"(?:напиши(?:те)?|пиши(?:те)?|ответь(?:те)?|отвечай(?:те)?|"
    r"спроси(?:те)?|спрашивай(?:те)?|скажи(?:те)?)\s+"
    r"(?:ему|ей|им|его|её|ее|их|у\s+него|у\s+неё|у\s+нее)\s+"
    r"(?:сам|сама|сами|самостоятельно)|"
    r"(?:сам|сама|сами|самостоятельно)\s+"
    r"(?:напиши(?:те)?|пиши(?:те)?|ответь(?:те)?|отвечай(?:те)?|"
    r"спроси(?:те)?|спрашивай(?:те)?|скажи(?:те)?)\s+"
    r"(?:ему|ей|им|его|её|ее|их)|"
    r"пусть\s+(?:(?:он|она|они|ты)\s+)?(?:сам|сама|сами)\s+"
    r"(?:ему|ей|им|его|её|ее|их)\s+"
    r"(?:напишет|ответит|спросит|скажет|напишешь|ответишь|спросишь|скажешь)"
    r")\b"
)


def _is_explicit_action_stop(text):
    """Match a follow-up that clearly cancels, pauses, or edits a queued action."""
    normalized = " ".join(str(text or "").split()).strip()
    return bool(normalized and _ACTION_STOP_RE.search(normalized))


def _is_action_meta_response(text):
    """Reject a generated promise/refusal about doing the requested reply."""
    normalized = " ".join(str(text or "").split()).strip()
    return bool(normalized and (
        _ACTION_META_RESPONSE_RE.fullmatch(normalized)
        or _ACTION_META_REFUSAL_RE.search(normalized)
        or _ACTION_META_REDIRECT_RE.search(normalized)
    ))


@dataclass(frozen=True)
class ActionRequest:
    command_msg_id: int
    chat_id: int
    thread_id: int
    target_message_id: int
    target_sender_id: int
    requester_sender_id: int
    trigger_reason: str
    generation: tuple[int, int]
    deadline: float
    recipient_username: str | None = None


def _is_owner_reply_command(text, aliases=(), artifact=None):
    """Match only the legacy owner-only, unnamed reply command."""
    normalized = " ".join(str(text or "").split()).strip()
    candidates = [normalized]
    for alias in aliases or ():
        if not isinstance(alias, str) or not alias.strip():
            continue
        prefix = re.escape(" ".join(alias.split()))
        candidates.append(re.sub(
            r"(?iu)^" + prefix + r"\s*[,!—–:-]?\s+", "", normalized,
            count=1,
        ))
    return any(re.fullmatch(r"(?iu)ответь\s+ему[.!?…]*", item)
               for item in candidates)


def _owner_action_match(text, aliases=()):
    normalized = " ".join(str(text or "").split()).strip()
    candidates = [normalized]
    for alias in aliases or ():
        if not isinstance(alias, str) or not alias.strip():
            continue
        prefix = re.escape(" ".join(alias.split()))
        candidates.append(re.sub(
            r"(?iu)^" + prefix + r"\s*[,!—–:-]?\s+", "", normalized, count=1,
        ))
    for candidate in candidates:
        match = re.fullmatch(
            r"(?iu)(?:ответь|ответить|напиши|написать|скажи|сказать|спроси|спросить)"
            r"(?:\s+(.+))?", candidate.rstrip(" .!?…"),
        )
        if match is not None:
            return match
    return None


def _owner_action_recipient_label(text, artifact, aliases=()):
    """Resolve only a locally known participant named in an action request."""
    match = _owner_action_match(text, aliases)
    if match is None or not match.group(1):
        return None
    reference = match.group(1).strip()
    words = reference.split()
    if (match.group(0).casefold().startswith(("спроси ", "спросить "))
            and words[0].casefold() == "у"):
        reference = " ".join(words[1:])
        words = reference.split()
        if not words:
            return None
    if words[0].strip(" ,.!?;:—–-").casefold() in {
            "ему", "ей", "им", "него", "нее", "неё", "них"}:
        return None
    for count in range(min(5, len(words)), 0, -1):
        candidate = " ".join(words[:count]).strip(" ,.!?;:—–-")
        if participant_reference_is_ambiguous(candidate, artifact):
            if participant_reference_is_known(candidate, artifact):
                return reference
            continue
        label = participant_label_for_reference(candidate, artifact)
        if label is not None:
            return label
    return None


def _recipient_first_action_label(text, artifact, aliases=()):
    """Resolve a known local recipient before an explicit action verb."""
    normalized = " ".join(str(text or "").split()).strip()
    for alias in sorted(
            (item for item in aliases or ()
             if isinstance(item, str) and item.strip()),
            key=len, reverse=True):
        prefix = re.escape(" ".join(alias.split()))
        match = re.match(
            r"(?iu)^\s*" + prefix + r"\s*[,!—–:-]\s*(.+)$",
            normalized,
        )
        if match:
            normalized = match.group(1).strip()
            break

    tokens = normalized.split()
    action_verb = re.compile(
        r"(?iu)^(?:ответь|ответить|напиши|написать|скажи|сказать|"
        r"спроси|спросить)\b(?:\s+.*)?$"
    )
    for count in range(min(5, len(tokens) - 1), 0, -1):
        reference = " ".join(tokens[:count]).strip(" ,.!?;:—–-")
        label = participant_label_for_reference(reference, artifact)
        if label is None:
            continue
        remainder = " ".join(tokens[count:]).strip()
        if action_verb.fullmatch(remainder):
            return label
    return None


def _group_action_intent(text, artifact, aliases=()):
    """Return an explicit action marker and its locally resolved recipient reference."""
    if _owner_action_match(text, aliases) is not None:
        return True, _owner_action_recipient_label(text, artifact, aliases)
    recipient = _recipient_first_action_label(text, artifact, aliases)
    return (True, recipient) if recipient is not None else (False, None)


def _log_skip(reason, message_id):
    # Video diagnostics never contain message text, metadata, or URLs.
    print(f"skip: {reason} message={message_id}")


def _log_fallback(reason, message_id):
    print(f"fallback: {reason} message={message_id}")


def _log_ordinary(reason, task_id, *, model_ms=None, total_ms=None):
    # Ordinary-path logs contain a private random task token, never Telegram IDs.
    timing = ""
    if model_ms is not None:
        timing += f" model_ms={max(0, int(model_ms))}"
    if total_ms is not None:
        timing += f" total_ms={max(0, int(total_ms))}"
    print(f"ordinary: {reason} task={task_id}{timing}")


def _fallback_for_reason(reason, *, recent=()):
    code = (reason if isinstance(reason, str)
            and reason in ORDINARY_FALLBACK_REASON_CODES else "unknown")
    if code == "needs_owner":
        blocked = {item.casefold() for item in recent[-2:] if isinstance(item, str)}
        text = next((item for item in _NEEDS_OWNER_FALLBACKS
                     if item.casefold() not in blocked), _NEEDS_OWNER_FALLBACKS[0])
        return code, text
    return code, ORDINARY_FALLBACK_BY_REASON.get(code, ORDINARY_FALLBACK)


async def _prune_ordinary_context(state, context, allowed_chats, enabled):
    """Periodically enforce the local context TTL while the bot is idle."""
    while True:
        await asyncio.sleep(CONTEXT_PRUNE_INTERVAL_SECONDS)
        try:
            active = bool(enabled())
            state.prune_context_snapshots(allowed_chats, enabled=active)
            if active:
                context.prune_expired()
            else:
                context.clear()
        except Exception:
            print("ordinary: context_prune_unavailable")


async def run(cfg, send=False, show_drafts=False):
    if cfg.chats_error:
        raise RuntimeError("allowlist_invalid")
    if not cfg.chats:
        raise RuntimeError("allowlist_empty")
    if not cfg.api_id or not cfg.api_hash:
        raise RuntimeError("telegram_config_missing")
    if send and show_drafts:
        raise RuntimeError("show_drafts_not_allowed_with_send")
    if show_drafts and (send or not __import__("sys").stdin.isatty()
                        or not __import__("sys").stdout.isatty()):
        raise RuntimeError("show_drafts_requires_local_tty")

    old_umask = os.umask(0o077)
    data_dir = ROOT / "data"
    data_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(data_dir, 0o700)
    lock = open(data_dir / "run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        os.umask(old_umask)
        raise RuntimeError("already_running") from None

    telegram = client(cfg)
    control_bot = None
    state = State(data_dir / "state.sqlite3")
    ordinary_requested = bool(getattr(cfg, "auto_reply_enabled", False))
    state.ordinary_mode_snapshot(default_enabled=ordinary_requested)
    state.link_mode_snapshot(default_enabled=True)

    def ordinary_is_enabled():
        enabled, _ = state.ordinary_mode_snapshot(
            default_enabled=ordinary_requested,
        )
        return ordinary_requested and enabled

    state.prune_context_snapshots(cfg.chats, enabled=ordinary_is_enabled())
    video_queue = asyncio.Queue()
    own_id = None
    own_username = None
    send_lock = asyncio.Lock()
    ordinary_model_slots = asyncio.Semaphore(ORDINARY_MODEL_CONCURRENCY)
    context = ContextBuffer(max_groups=max(100, len(cfg.chats)))
    recent_stickers = (
        RecentStickerCatalog()
        if ordinary_requested and getattr(cfg, "auto_reply_stickers_enabled", False)
        else None
    )
    last_sticker_pack_by_chat: dict[int, str] = {}
    needs_owner_fallback_history: dict[int, list[str]] = {}
    if ordinary_is_enabled():
        try:
            context.hydrate(
                MessageSnapshot(
                    int(chat), int(message),
                    int(sender) if sender is not None else None,
                    bool(own), int(thread),
                    int(reply_to) if reply_to is not None else None,
                    float(timestamp), str(text),
                )
                for chat, message, sender, own, thread, reply_to, timestamp, text
                in state.load_context_snapshots(cfg.chats)
            )
        except Exception:
            # Continue with live context if a damaged local DB cannot hydrate.
            print("ordinary: context_restore_unavailable")
    history_attempted: set[tuple[int, int]] = set()
    active_ordinary: set[asyncio.Task] = set()
    ordinary_task_triggers: dict[asyncio.Task, tuple[int, int]] = {}
    active_owner_actions: set[asyncio.Task] = set()
    started_owner_actions: set[asyncio.Task] = set()
    owner_actions_by_thread: dict[
        tuple[int, int], dict[int, asyncio.Task]
    ] = {}
    owner_action_requesters: dict[tuple[int, int, int], int] = {}
    followup_sessions: dict[tuple[int, int], tuple[float, int]] = {}
    other_participant_conversation_anchors: dict[tuple[int, int], int] = {}
    collecting_ordinary: set[asyncio.Task] = set()
    started_ordinary: set[asyncio.Task] = set()
    ordinary_by_thread: dict[
        tuple[int, int], tuple[asyncio.Task, int, str, int | None]
    ] = {}
    ordinary_sending: set[tuple[int, int]] = set()
    superseded: set[tuple[int, int]] = set()
    send_not_before = 0.0
    control_enabled = bool(getattr(cfg, "control_bot_token", ""))
    profile = None
    profile_error = ""
    voice_aliases = tuple(load_voice_config().get("identity", {}).get("aliases", []))
    participant_artifacts: dict[int, tuple[float, dict | None]] = {}

    def remember_snapshot(snapshot):
        if not ordinary_is_enabled() or snapshot is None:
            return False
        added = context.add(snapshot)
        try:
            state.save_context_snapshot(snapshot, cfg.chats)
        except Exception:
            # The message stays available in RAM; never log its content or IDs.
            print("ordinary: context_persist_unavailable")
        return added

    def remember_confirmed_reply(sent_message, chat_id, trigger_id, thread_id):
        if not ordinary_is_enabled() or sent_message is None:
            return
        snapshot = snapshot_from_message(sent_message, chat_id, own_id)
        if snapshot is None:
            return
        # A successful send_message is confirmed and was explicitly sent as a
        # group message. Preserve the actual reply relation in shared state.
        snapshot = replace(
            snapshot, sender_id=own_id, own=True,
            thread_id=int(thread_id),
            reply_to_id=int(trigger_id) if trigger_id is not None else None,
        )
        remember_snapshot(snapshot)

    def remember_confirmed_sticker(sent_message, chat_id, trigger_id,
                                   thread_id, emoji):
        if not ordinary_is_enabled() or sent_message is None:
            return
        try:
            message_id = int(getattr(sent_message, "id"))
        except (TypeError, ValueError):
            return
        date = getattr(sent_message, "date", None)
        try:
            timestamp = float(date.timestamp()) if date is not None else time.time()
        except Exception:
            timestamp = time.time()
        remember_snapshot(MessageSnapshot(
            int(chat_id), message_id, int(own_id) if own_id is not None else None,
            True, int(thread_id),
            int(trigger_id) if trigger_id is not None else None, timestamp,
            f"[стикер {emoji}]",
        ))
    if ordinary_requested:
        try:
            profile = load_private_profile(getattr(cfg, "style_profile_path", "data/style/profile.json"))
        except Exception:
            profile_error = "profile_unavailable"
        if getattr(cfg, "reply_config_error", ""):
            profile = None
            profile_error = "reply_config_invalid"

    def participant_artifact(chat_id):
        """Use shared and group-local summaries; retry missing files promptly."""
        chat_id = int(chat_id)
        now = time.monotonic()
        cached = participant_artifacts.get(chat_id)
        if cached is not None:
            cached_at, artifact = cached
            ttl = (PARTICIPANT_ARTIFACT_CACHE_SECONDS if artifact is not None
                   else PARTICIPANT_ARTIFACT_NEGATIVE_CACHE_SECONDS)
            if now - cached_at < ttl:
                return artifact
        global_artifact = None
        group_artifact = None
        try:
            global_artifact = load_private_participant_summaries(
                participant_profiles_path(), 0,
            )
        except Exception:
            pass
        try:
            group_artifact = load_private_participant_summaries(
                participant_summary_path(chat_id), chat_id,
            )
        except Exception:
            pass
        artifact = combine_participant_artifacts(global_artifact, group_artifact)
        participant_artifacts[chat_id] = (now, artifact)
        return artifact

    def video_generation_snapshot():
        _paused, global_generation = state.pause_snapshot()
        links_enabled, links_generation = state.link_mode_snapshot(
            default_enabled=True,
        )
        return global_generation, links_generation, links_enabled

    def video_generation_is_active(generation):
        try:
            global_generation, links_generation, enabled_at_intake = generation
        except (TypeError, ValueError):
            return False
        paused, current_global_generation = state.pause_snapshot()
        links_enabled, current_links_generation = state.link_mode_snapshot(
            default_enabled=True,
        )
        return (
            not paused and enabled_at_intake and links_enabled
            and current_global_generation == global_generation
            and current_links_generation == links_generation
        )

    def queue_video(ev, generation):
        if not generation[2]:
            return False
        video_queue.put_nowait((ev, generation))
        return True

    def ordinary_generation_snapshot():
        _paused, global_generation = state.pause_snapshot()
        _enabled, ordinary_generation = state.ordinary_mode_snapshot(
            default_enabled=ordinary_requested,
        )
        return global_generation, ordinary_generation

    def ordinary_generation_is_active(generation):
        if not ordinary_requested:
            return False
        try:
            global_generation, ordinary_generation = generation
        except (TypeError, ValueError):
            return False
        paused, current_global_generation = state.pause_snapshot()
        enabled, current_ordinary_generation = state.ordinary_mode_snapshot(
            default_enabled=ordinary_requested,
        )
        return (
            not paused and enabled
            and current_global_generation == global_generation
            and current_ordinary_generation == ordinary_generation
        )

    async def wait_for_send_gate(deadline=None):
        """Wait for process-wide Telegram FloodWait backoff, respecting a task deadline."""
        loop = asyncio.get_running_loop()
        while True:
            target = send_not_before
            remaining = send_not_before - loop.time()
            if remaining <= 0:
                return target
            if deadline is not None and loop.time() + remaining >= deadline:
                return None
            await asyncio.sleep(remaining)
            if send_not_before <= target:
                # asyncio.sleep has completed the requested backoff. Returning
                # the waited target also makes the gate stable under injected
                # test sleepers whose clock does not advance.
                return target

    def skip_for_inactive_video(ev, route, already_claimed, generation):
        if already_claimed:
            state.status(ev.chat_id, ev.message.id, "skip")
        elif not state.claim(ev.chat_id, ev.message.id, route, "skip", kind="video"):
            _log_skip("duplicate_message", ev.message.id)
            return
        global_generation = None
        try:
            global_generation, links_generation, enabled_at_intake = generation
        except (TypeError, ValueError):
            links_generation = None
            enabled_at_intake = False
        paused, current_global_generation = state.pause_snapshot()
        links_enabled, current_links_generation = state.link_mode_snapshot(
            default_enabled=True,
        )
        reason = (
            "paused" if paused or (
                global_generation is not None
                and current_global_generation != global_generation
            )
            else "links_disabled" if (
                not enabled_at_intake or not links_enabled
                or current_links_generation != links_generation
            )
            else "mode_changed"
        )
        _log_skip(reason, ev.message.id)

    def ordinary_status_text():
        counts = state.ordinary_counts()
        if not ordinary_requested:
            mode = "выключены настройкой сервера"
        elif ordinary_is_enabled() and profile is None:
            mode = "включены, но профиль недоступен"
        else:
            mode = "включены" if ordinary_is_enabled() else "выключены"
        paused, _ = state.pause_snapshot()
        links_enabled, _ = state.link_mode_snapshot(default_enabled=True)
        links = "включены" if links_enabled else "выключены"
        return (
            f"Обычное общение: {mode}; ответы на ссылки TikTok, Reels и Shorts: {links}; "
            f"общая пауза: {'включена' if paused else 'выключена'}; "
            f"отправлено обычных ответов за 24 часа: {counts['sent_last_24h']}."
        )

    def cancel_ordinary_for_pause():
        followup_sessions.clear()
        other_participant_conversation_anchors.clear()
        for task in tuple(active_ordinary | active_owner_actions):
            if not task.done():
                task.cancel()

    def cancel_ordinary_for_target(chat_id, thread_id, target_message_id):
        for task, trigger in tuple(ordinary_task_triggers.items()):
            if trigger == (int(chat_id), int(target_message_id)) and not task.done():
                task.cancel()

    def cancel_owner_actions_after_owner_message(
        chat_id, thread_id, message_id, sender_id, message_text,
    ):
        if (sender_id is None or sender_id == own_id
                or not _is_explicit_action_stop(message_text)):
            return
        key = (int(chat_id), int(thread_id))
        actions = owner_actions_by_thread.get(key, {})
        owner_ids = set(getattr(cfg, "owner_command_sender_ids", ()) or ())
        for command_id, task in tuple(actions.items()):
            requester_id = owner_action_requesters.get(
                (key[0], key[1], int(command_id)),
            )
            if (int(message_id) > command_id
                    and (sender_id == requester_id or sender_id in owner_ids)
                    and not task.done()):
                task.cancel()

    def owner_action_cancelled_by_followup(
        chat_id, thread_id, command_id, requester_id,
    ):
        cancel_senders = set(
            getattr(cfg, "owner_command_sender_ids", ()) or ()
        ) | {int(requester_id)}
        cancel_senders.discard(own_id)
        return any(
            item.message_id > int(command_id)
            and item.sender_id in cancel_senders
            and item.sender_id != own_id and not item.own
            and _is_explicit_action_stop(item.text)
            for item in context.messages(int(chat_id), int(thread_id))
        )

    def on_ordinary_toggle(enabled):
        if enabled:
            try:
                state.prune_context_snapshots(cfg.chats, enabled=True)
                context.hydrate(
                    MessageSnapshot(
                        int(chat), int(message),
                        int(sender) if sender is not None else None,
                        bool(own), int(thread),
                        int(reply_to) if reply_to is not None else None,
                        float(timestamp), str(text),
                    )
                    for chat, message, sender, own, thread, reply_to, timestamp, text
                    in state.load_context_snapshots(cfg.chats)
                )
            except Exception:
                print("ordinary: context_restore_unavailable")
            return
        cancel_ordinary_for_pause()
        followup_sessions.clear()
        context.clear()
        state.prune_context_snapshots(cfg.chats, enabled=False)

    def lookup_budget(deadline):
        return max(0.0, min(
            ORDINARY_LOOKUP_TIMEOUT, deadline - asyncio.get_running_loop().time(),
        ))

    async def get_history_before(ev, message, thread_id, amount, *, deadline, task_id):
        key = (int(ev.chat_id), thread_id)
        if amount <= 0 or key in history_attempted:
            return []
        prior = [item for item in context.messages(ev.chat_id, thread_id)
                 if item.message_id < message.id]
        if len(prior) >= amount:
            return []
        history_attempted.add(key)
        iterator = getattr(telegram, "iter_messages", None)
        if iterator is None:
            history_attempted.discard(key)
            _log_ordinary("history_unavailable", task_id)
            return []
        snapshots = []
        try:
            async with asyncio.timeout(lookup_budget(deadline)):
                async for item in iterator(
                    ev.chat_id, limit=min(20, amount + 10), max_id=message.id,
                    reverse=False,
                ):
                    if getattr(item, "id", 0) >= message.id or thread_key(item) != thread_id:
                        continue
                    snap = snapshot_from_message(item, ev.chat_id, own_id)
                    if snap is not None:
                        snapshots.append(snap)
                        remember_snapshot(snap)
                    if len(snapshots) >= amount:
                        break
        except asyncio.CancelledError:
            history_attempted.discard(key)
            raise
        except asyncio.TimeoutError:
            history_attempted.discard(key)
            _log_ordinary("history_timeout", task_id)
            return []
        except Exception:
            history_attempted.discard(key)
            _log_ordinary("history_unavailable", task_id)
            return []
        return sorted(snapshots, key=lambda item: (item.timestamp, item.message_id))[-amount:]

    async def owner_replied_after(ev, trigger_id, thread_id, *, tail_check,
                                  deadline, task_id,
                                  exclude_message_ids=()):
        excluded = {int(value) for value in exclude_message_ids if value is not None}
        if context.owner_replied_after(
            ev.chat_id, thread_id, trigger_id, own_id,
            exclude_message_ids=excluded,
        ):
            return True
        if not tail_check:
            return False
        iterator = getattr(telegram, "iter_messages", None)
        if iterator is None:
            return None
        try:
            count = 0
            async with asyncio.timeout(lookup_budget(deadline)):
                async for item in iterator(
                    ev.chat_id, limit=20, min_id=trigger_id, reverse=True,
                ):
                    count += 1
                    try:
                        item_id = int(getattr(item, "id", 0))
                    except (TypeError, ValueError):
                        item_id = 0
                    try:
                        item_chat_id = getattr(item, "chat_id", None)
                        item_chat_id = (
                            int(item_chat_id) if item_chat_id is not None else None
                        )
                        parent_id = getattr(item, "reply_to_msg_id", None)
                        parent_id = int(parent_id) if parent_id is not None else None
                    except (TypeError, ValueError):
                        item_chat_id = parent_id = None
                    same_chat = item_chat_id is None or item_chat_id == int(ev.chat_id)
                    if (item_id > int(trigger_id) and item_id not in excluded
                            and same_chat and parent_id == int(trigger_id)
                            and thread_key(item) == thread_id
                            and getattr(item, "sender_id", None) == own_id):
                        return True
                    if count >= 20:
                        _log_ordinary("owner_tail_truncated", task_id)
                        return None
            return False
        except asyncio.TimeoutError:
            _log_ordinary("owner_tail_timeout", task_id)
            return None
        except Exception:
            _log_ordinary("owner_tail_lookup_error", task_id)
            return None

    def context_snapshot(chat_id, thread_id, message_id):
        return next((item for item in context.messages(chat_id, thread_id)
                     if item.message_id == int(message_id)), None)

    def active_followup_anchor(chat_id, thread_id):
        key = (int(chat_id), int(thread_id))
        session = followup_sessions.get(key)
        if session is None:
            return None
        expires_at, message_id = session
        if asyncio.get_running_loop().time() >= expires_at:
            followup_sessions.pop(key, None)
            other_participant_conversation_anchors.pop(key, None)
            return None
        snapshot = context_snapshot(key[0], key[1], message_id)
        if snapshot is None or not snapshot.own:
            followup_sessions.pop(key, None)
            other_participant_conversation_anchors.pop(key, None)
            return None
        if other_participant_conversation_anchors.get(key) != int(message_id):
            other_participant_conversation_anchors.pop(key, None)
        return int(message_id)

    def open_followup_session(chat_id, thread_id, message_id):
        try:
            message_id = int(message_id)
        except (TypeError, ValueError):
            return
        if message_id <= 0 or not ordinary_is_enabled():
            return
        now = asyncio.get_running_loop().time()
        for key, (expires_at, _anchor_id) in tuple(followup_sessions.items()):
            if expires_at <= now:
                followup_sessions.pop(key, None)
                other_participant_conversation_anchors.pop(key, None)
        key = (int(chat_id), int(thread_id))
        other_participant_conversation_anchors.pop(key, None)
        followup_sessions[key] = (
            now + FOLLOWUP_IDLE_TTL_SECONDS, message_id,
        )

    async def resolve_owner_action_target(
        ev, deadline, task_id, *, recipient_reference=None,
        requester_sender_id=None,
    ):
        """Resolve a group's explicit recipient separately from its requester."""
        command = ev.message
        chat_id = int(ev.chat_id)
        thread_id = thread_key(command)
        command_owner_ids = set(getattr(cfg, "owner_command_sender_ids", ()) or ())
        owner_actor_ids = command_owner_ids | {own_id}
        try:
            requester_sender_id = int(
                requester_sender_id if requester_sender_id is not None
                else getattr(command, "sender_id"),
            )
        except (TypeError, ValueError):
            return None
        target_excluded_ids = {requester_sender_id}
        if own_id is not None:
            target_excluded_ids.add(int(own_id))

        def recent_snapshot(snapshot):
            now = time.time()
            return bool(
                snapshot is not None and snapshot.thread_id == thread_id
                and math.isfinite(float(snapshot.timestamp))
                and now - CONTEXT_TTL_SECONDS <= snapshot.timestamp <= now + 300
            )

        def valid_snapshot(snapshot, sender_id=None):
            return bool(
                recent_snapshot(snapshot) and not snapshot.own
                and (sender_id is None or snapshot.sender_id == sender_id)
                and snapshot.sender_id not in target_excluded_ids
            )

        async def fetch_message(message_id):
            try:
                async with asyncio.timeout(lookup_budget(deadline)):
                    item = await telegram.get_messages(chat_id, ids=int(message_id))
            except asyncio.CancelledError:
                raise
            except Exception:
                return None
            if isinstance(item, (list, tuple)):
                return item[0] if len(item) == 1 else None
            return item

        async def verify_target(target, snapshot):
            if (target is None or getattr(target, "action", None) is not None
                    or getattr(target, "fwd_from", None) is not None
                    or not valid_snapshot(snapshot)):
                return None
            try:
                target_id = int(getattr(target, "id"))
                target_sender_id = int(getattr(target, "sender_id"))
                target_chat = int(getattr(target, "chat_id"))
            except (TypeError, ValueError):
                return None
            if (target_id >= int(command.id) or target_chat != chat_id
                    or target_sender_id in target_excluded_ids
                    or snapshot.message_id != target_id
                    or snapshot.sender_id != target_sender_id
                    or thread_key(target) != thread_id):
                return None
            try:
                async with asyncio.timeout(lookup_budget(deadline)):
                    sender = await target.get_sender()
            except asyncio.CancelledError:
                raise
            except Exception:
                return None
            if (not isinstance(sender, User) or getattr(sender, "bot", False)
                    or getattr(sender, "id", None) != target_sender_id):
                return None
            summary = participant_summary_for_sender(
                sender, participant_artifact(chat_id),
            )
            username = getattr(sender, "username", None)
            if (not isinstance(username, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{4,31}", username)):
                username = None
            return target_id, target_sender_id, snapshot, summary, username

        async def resolve_named_recipient(label):
            artifact = participant_artifact(chat_id)
            recent = [
                item for item in context.messages(chat_id, thread_id)
                if item.message_id < int(command.id)
                and item.sender_id is not None and item.text
                and valid_snapshot(item)
            ]
            sender_ids = tuple(dict.fromkeys(item.sender_id for item in recent))

            async def lookup_sender(sender_id):
                try:
                    async with asyncio.timeout(lookup_budget(deadline)):
                        sender = await telegram.get_entity(int(sender_id))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return None
                if getattr(sender, "id", None) != sender_id:
                    return None
                if getattr(sender, "bot", False):
                    return False
                return sender

            if not sender_ids:
                return None
            senders = await asyncio.gather(
                *(lookup_sender(sender_id) for sender_id in sender_ids),
                return_exceptions=True,
            )
            lookup_incomplete = any(
                sender is None or isinstance(sender, BaseException)
                for sender in senders
            )
            words = label.split()
            for count in range(min(5, len(words)), 0, -1):
                candidate = " ".join(words[:count]).strip(" ,.!?;:—–-")
                if participant_reference_is_ambiguous(candidate, artifact):
                    return None
                # A locally unique profile alias or explicit @username remains
                # identifiable when unrelated senders fail lookup. An unknown
                # display-name fallback does not: failed senders could share it.
                if (lookup_incomplete
                        and participant_label_for_reference(candidate, artifact) is None
                        and not candidate.startswith("@")):
                    return None
                matching_senders = {
                    sender_id for sender_id, sender in zip(sender_ids, senders)
                    if (sender is not None and sender is not False
                        and not isinstance(sender, BaseException)
                        and participant_reference_matches_sender(
                            candidate, sender, artifact,
                        ))
                }
                if not matching_senders:
                    continue
                if len(matching_senders) != 1:
                    return None
                matches = [item for item in recent if item.sender_id in matching_senders]
                target_snapshot = max(matches, key=lambda item: item.message_id)
                target = await fetch_message(target_snapshot.message_id)
                return await verify_target(target, target_snapshot)
            return None

        async def resolve_named_question(question, question_snapshot, bot_reply_id):
            artifact = participant_artifact(chat_id)
            query = match_participant_question(question, artifact)
            if (query is None or not query.should_trigger or not query.label
                    or question_snapshot is None
                    or question_snapshot.sender_id not in owner_actor_ids
                    or not recent_snapshot(question_snapshot)):
                return None
            try:
                question_id = int(getattr(question, "id"))
                question_sender_id = int(getattr(question, "sender_id"))
                question_chat_id = int(getattr(question, "chat_id"))
            except (TypeError, ValueError):
                return None
            if (question_id != question_snapshot.message_id
                    or question_sender_id != question_snapshot.sender_id
                    or question_chat_id != chat_id
                    or thread_key(question) != thread_id
                    or getattr(question, "action", None) is not None
                    or getattr(question, "fwd_from", None) is not None):
                return None

            # A direct reply is the strongest source. The broader local buffer
            # is used only to identify the sender; the model still receives its
            # fixed, selected context window below.
            if question_snapshot.reply_to_id is not None:
                direct_snapshot = context_snapshot(
                    chat_id, thread_id, question_snapshot.reply_to_id,
                )
                if valid_snapshot(direct_snapshot):
                    direct = await fetch_message(direct_snapshot.message_id)
                    if direct is not None:
                        try:
                            async with asyncio.timeout(lookup_budget(deadline)):
                                direct_sender = await direct.get_sender()
                        except Exception:
                            direct_sender = None
                        if (isinstance(direct_sender, User)
                                and participant_label_for_sender(direct_sender, artifact)
                                == query.label):
                            verified = await verify_target(direct, direct_snapshot)
                            if verified is not None:
                                return verified

            # Recipient resolution may inspect all buffered messages in this
            # same thread without expanding the separate Gemini context window.
            frozen = [
                item for item in context.messages(chat_id, thread_id)
                if item.message_id < int(bot_reply_id)
            ]
            candidates = [
                item for item in frozen
                if item.message_id != question_snapshot.message_id
                and item.sender_id is not None and item.sender_id not in owner_actor_ids
                and not item.own and item.text
                and valid_snapshot(item, item.sender_id)
            ]
            sender_ids = tuple(dict.fromkeys(item.sender_id for item in candidates))

            async def sender_matches(sender_id):
                try:
                    async with asyncio.timeout(lookup_budget(deadline)):
                        sender = await telegram.get_entity(int(sender_id))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return None
                if getattr(sender, "id", None) != sender_id:
                    return None
                if getattr(sender, "bot", False):
                    return False
                return participant_label_for_sender(sender, artifact) == query.label

            if sender_ids:
                matched = await asyncio.gather(
                    *(sender_matches(sender_id) for sender_id in sender_ids),
                    return_exceptions=True,
                )
                matching_senders = {
                    sender_id for sender_id, result in zip(sender_ids, matched)
                    if result is True
                }
                matches = [item for item in candidates
                           if item.sender_id in matching_senders]
            else:
                matches = []
            if not matches or len({item.sender_id for item in matches}) != 1:
                return None
            target_snapshot = max(matches, key=lambda item: item.message_id)
            candidate = await fetch_message(target_snapshot.message_id)
            return await verify_target(candidate, target_snapshot)

        if recipient_reference is not None:
            return await resolve_named_recipient(recipient_reference)

        if not getattr(command, "reply_to_msg_id", None):
            return None
        try:
            async with asyncio.timeout(lookup_budget(deadline)):
                parent = await ev.get_reply_message()
        except asyncio.CancelledError:
            raise
        except Exception:
            _log_ordinary("owner_action_parent_unavailable", task_id)
            return None
        if (parent is None or getattr(parent, "action", None) is not None
                or getattr(parent, "fwd_from", None) is not None):
            return None
        parent_id = getattr(parent, "id", None)
        parent_chat = getattr(parent, "chat_id", None)
        if (parent_id is None or parent_chat is None
                or int(parent_chat) != chat_id
                or int(parent_id) >= int(command.id)
                or thread_key(parent) != thread_id):
            return None

        parent_sender_id = getattr(parent, "sender_id", None)
        if parent_sender_id in owner_actor_ids:
            # An owner message or the bot's reply can point to the participant
            # or to an owner question naming that participant.
            target_id = getattr(parent, "reply_to_msg_id", None)
            parent_snapshot = context_snapshot(chat_id, thread_id, parent_id)
            if (parent_snapshot is None
                    or parent_snapshot.sender_id != parent_sender_id
                    or not recent_snapshot(parent_snapshot)
                    or (parent_sender_id == own_id and not parent_snapshot.own)):
                return None
            if target_id is None:
                if parent_sender_id in command_owner_ids:
                    return await resolve_named_question(parent, parent_snapshot, parent_id)
                return None
            if (parent_snapshot.reply_to_id != int(target_id)
                    or not 0 < int(target_id) < int(parent_id)):
                return None
            target = await fetch_message(target_id)
            if target is None:
                return None
            target_snapshot = context_snapshot(chat_id, thread_id, target_id)
            try:
                target_sender_id = int(getattr(target, "sender_id"))
            except (TypeError, ValueError):
                return None
            if target_sender_id in owner_actor_ids:
                return await resolve_named_question(
                    target, target_snapshot, parent_id,
                )
            return await verify_target(target, target_snapshot)
        else:
            target = parent
        try:
            target_sender_id = int(getattr(target, "sender_id"))
        except (TypeError, ValueError):
            return None
        if target_sender_id in target_excluded_ids:
            return None
        return await verify_target(target, context_snapshot(chat_id, thread_id, getattr(target, "id", 0)))

    def owner_action_window(request, trigger_snapshot):
        before = max(0, int(getattr(cfg, "reply_context_before", 5)))
        after = max(0, int(getattr(cfg, "reply_context_after", 3)))
        frozen = [
            item for item in context.messages(request.chat_id, request.thread_id)
            if item.message_id < request.command_msg_id
        ]
        if not any(item.message_id == request.target_message_id
                   and item.sender_id == request.target_sender_id for item in frozen):
            raise ValueError("owner_action_context_missing")
        selected = ContextBuffer(max_groups=1, max_messages_per_group=50)
        selected.hydrate(frozen)
        selected.add(trigger_snapshot)
        window = selected.make_window(
            request.chat_id, request.thread_id, request.command_msg_id,
            own_id, before, after,
            trigger_reason=request.trigger_reason,
            trigger_snapshot=trigger_snapshot,
            required_context_ids=(request.target_message_id,),
        )
        if not window.get("messages"):
            raise ValueError("owner_action_context_missing")
        return sanitize_window(window)

    async def owner_action_worker(ev, generation, deadline, command_id,
                                  thread_id, task_id, request=None,
                                  trigger_snapshot=None, participant_summary=None,
                                  recipient_vocatives=(), requester_sender_id=None):
        nonlocal send_not_before
        if request is not None:
            generation = request.generation
            deadline = request.deadline
            command_id = request.command_msg_id
            thread_id = request.thread_id
        task = asyncio.current_task()
        started_owner_actions.add(task)
        active_owner_actions.add(task)
        chat_id = int(ev.chat_id)
        if request is not None:
            requester_sender_id = request.requester_sender_id
        try:
            requester_sender_id = int(
                requester_sender_id
                if requester_sender_id is not None
                else getattr(ev.message, "sender_id"),
            )
        except (TypeError, ValueError):
            requester_sender_id = None
        loop = asyncio.get_running_loop()
        reservation_active = False
        sending_started = False
        active_part = 0
        text = _OWNER_ACTION_CLARIFICATION
        reply_to_id = int(command_id)
        action_target = None
        send_username = None

        def finish_skip(reason, *, final_status="skip"):
            nonlocal reservation_active
            if reservation_active:
                state.finish_ordinary_reply(chat_id, command_id, "released")
                reservation_active = False
            state.status(chat_id, command_id, "skip")
            state.finish_owner_action(chat_id, command_id, final_status)
            _log_ordinary(reason, task_id)

        try:
            if not ordinary_generation_is_active(generation):
                finish_skip("owner_action_paused", final_status="cancelled")
                return
            allowed, reserve_reason = state.reserve_ordinary_reply(
                chat_id, command_id, dry_run=not send,
            )
            if not allowed:
                finish_skip("owner_action_" + reserve_reason)
                return
            reservation_active = True

            if request is not None:
                if profile is None:
                    finish_skip("owner_action_profile_unavailable")
                    return
                try:
                    window = owner_action_window(request, trigger_snapshot)
                except Exception:
                    finish_skip("owner_action_context_unavailable")
                    return
                action_target = request
                reply_to_id = request.target_message_id
                model_deadline = min(
                    deadline, loop.time() + ORDINARY_GENERATION_TIMEOUT,
                )
                acquired = False
                try:
                    async with asyncio.timeout_at(model_deadline):
                        await ordinary_model_slots.acquire()
                        acquired = True
                        if not ordinary_generation_is_active(generation):
                            finish_skip("owner_action_paused", final_status="cancelled")
                            return
                        result = await generate_reply(
                            cfg.gemini_key,
                            getattr(cfg, "reply_model", "") or cfg.model,
                            profile, window,
                            participant_summary=participant_summary,
                            recipient_vocatives=recipient_vocatives,
                        )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    finish_skip("owner_action_generation_timeout")
                    return
                except Exception:
                    finish_skip("owner_action_generation_error")
                    return
                finally:
                    if acquired:
                        ordinary_model_slots.release()
                if (getattr(result, "reason", "") != "ok"
                        or not isinstance(getattr(result, "text", None), str)
                        or not result.text.strip()):
                    finish_skip("owner_action_generation_declined")
                    return
                text = result.text
                if _is_action_meta_response(text):
                    finish_skip("owner_action_meta_response")
                    return

            if loop.time() >= deadline:
                finish_skip("owner_action_deadline")
                return
            if not ordinary_generation_is_active(generation):
                finish_skip("owner_action_paused", final_status="cancelled")
                return
            if not send:
                state.status(chat_id, command_id, "dry")
                state.finish_ordinary_reply(chat_id, command_id, "dry")
                reservation_active = False
                state.finish_owner_action(
                    chat_id, command_id,
                    "dry" if request is not None else "ambiguous",
                )
                _log_ordinary("owner_action_dry_run_ready", task_id)
                return
            if not ev.is_group or chat_id not in cfg.chats:
                finish_skip("owner_action_send_guard")
                return

            while True:
                if loop.time() >= deadline:
                    finish_skip("owner_action_deadline")
                    return
                waited_gate = await wait_for_send_gate(deadline)
                if waited_gate is None:
                    finish_skip("owner_action_deadline")
                    return
                try:
                    async with asyncio.timeout_at(deadline):
                        await send_lock.acquire()
                except asyncio.TimeoutError:
                    finish_skip("owner_action_send_lock_timeout")
                    return
                gate_wait = False
                flood_wait = False
                send_username = None
                try:
                    if send_not_before > waited_gate:
                        gate_wait = True
                    elif not ordinary_generation_is_active(generation):
                        finish_skip("owner_action_paused", final_status="cancelled")
                        return
                    elif not ev.is_group or chat_id not in cfg.chats:
                        finish_skip("owner_action_send_guard")
                        return
                    elif owner_action_cancelled_by_followup(
                        chat_id, thread_id, command_id, requester_sender_id,
                    ):
                        finish_skip("owner_action_requester_followup", final_status="cancelled")
                        return
                    else:
                        if action_target is not None:
                            try:
                                async with asyncio.timeout(lookup_budget(deadline)):
                                    current_target = await telegram.get_messages(
                                        chat_id, ids=action_target.target_message_id,
                                    )
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                current_target = None
                            if isinstance(current_target, (list, tuple)):
                                current_target = (
                                    current_target[0] if len(current_target) == 1 else None
                                )
                            try:
                                current_target_id = int(getattr(current_target, "id"))
                                current_sender_id = int(getattr(current_target, "sender_id"))
                                current_chat_id = int(getattr(current_target, "chat_id"))
                            except (TypeError, ValueError):
                                finish_skip("owner_action_target_changed")
                                return
                            if (current_target_id != action_target.target_message_id
                                    or current_sender_id != action_target.target_sender_id
                                    or current_chat_id != chat_id
                                    or thread_key(current_target) != thread_id
                                    or getattr(current_target, "action", None) is not None
                                    or getattr(current_target, "fwd_from", None) is not None):
                                finish_skip("owner_action_target_changed")
                                return
                            send_username = None
                            if action_target.recipient_username:
                                try:
                                    async with asyncio.timeout(lookup_budget(deadline)):
                                        current_sender = await current_target.get_sender()
                                except asyncio.CancelledError:
                                    raise
                                except Exception:
                                    current_sender = None
                                current_username = getattr(
                                    current_sender, "username", None,
                                )
                                if (isinstance(current_sender, User)
                                        and not getattr(current_sender, "bot", False)
                                        and getattr(current_sender, "id", None)
                                        == action_target.target_sender_id
                                        and current_username
                                        == action_target.recipient_username):
                                    send_username = action_target.recipient_username
                        if (loop.time() >= deadline
                                or not ordinary_generation_is_active(generation)):
                            finish_skip(
                                "owner_action_deadline" if loop.time() >= deadline
                                else "owner_action_paused",
                                final_status="cancelled",
                            )
                            return
                        if send_not_before > waited_gate:
                            gate_wait = True
                        else:
                            allowed, reserve_reason = state.start_ordinary_part(
                                chat_id, command_id, 1,
                            )
                            if not allowed:
                                finish_skip("owner_action_" + reserve_reason)
                                return
                            active_part = 1
                            sending_started = True
                            state.status(chat_id, command_id, "sending")
                            state.finish_owner_action(chat_id, command_id, "sending")
                            # No await is allowed between the final local gates
                            # and the one group send.
                            if (loop.time() >= deadline
                                    or not ordinary_generation_is_active(generation)):
                                state.finish_ordinary_part(
                                    chat_id, command_id, 1, "released",
                                )
                                active_part = 0
                                sending_started = False
                                finish_skip(
                                    "owner_action_deadline" if loop.time() >= deadline
                                    else "owner_action_paused",
                                    final_status="cancelled",
                                )
                                return
                            if send_not_before > waited_gate:
                                state.defer_ordinary_part(chat_id, command_id, 1)
                                active_part = 0
                                sending_started = False
                                gate_wait = True
                            else:
                                formatting_entities = None
                                outbound_text = text
                                if send_username:
                                    mention = "@" + send_username
                                    if len(outbound_text) + len(mention) + 1 <= 350:
                                        outbound_text = mention + " " + outbound_text
                                        formatting_entities = [MessageEntityMention(
                                            offset=0, length=len(mention),
                                        )]
                                try:
                                    async with asyncio.timeout_at(deadline):
                                        sent_message = await telegram.send_message(
                                            chat_id, outbound_text, reply_to=reply_to_id,
                                            parse_mode=None,
                                            formatting_entities=formatting_entities,
                                            link_preview=False,
                                        )
                                    try:
                                        outbound_id = int(getattr(sent_message, "id"))
                                    except (TypeError, ValueError):
                                        outbound_id = None
                                    overall = state.finish_ordinary_part(
                                        chat_id, command_id, 1, "sent",
                                    )
                                    state.status(chat_id, command_id, overall)
                                    state.finish_owner_action(
                                        chat_id, command_id,
                                        "sent" if request is not None else "clarification_sent",
                                        outbound=outbound_id,
                                    )
                                    reservation_active = False
                                    sending_started = False
                                    active_part = 0
                                    remember_confirmed_reply(
                                        sent_message, chat_id, reply_to_id, thread_id,
                                    )
                                    if request is not None and outbound_id is not None:
                                        open_followup_session(
                                            chat_id, thread_id, outbound_id,
                                        )
                                except Exception as exc:
                                    if type(exc).__name__ == "FloodWaitError":
                                        seconds = max(1, int(getattr(exc, "seconds", 1)))
                                        send_not_before = max(
                                            send_not_before, loop.time() + seconds,
                                        )
                                        state.defer_ordinary_part(
                                            chat_id, command_id, 1,
                                        )
                                        state.status(chat_id, command_id, "pending")
                                        state.finish_owner_action(
                                            chat_id, command_id, "pending",
                                        )
                                        active_part = 0
                                        sending_started = False
                                        flood_wait = True
                                    else:
                                        state.finish_ordinary_part(
                                            chat_id, command_id, 1, "uncertain",
                                        )
                                        state.status(chat_id, command_id, "uncertain")
                                        state.finish_owner_action(
                                            chat_id, command_id, "uncertain",
                                        )
                                        reservation_active = False
                                        active_part = 0
                                        sending_started = False
                                        _log_ordinary("owner_action_send_uncertain", task_id)
                                        return
                finally:
                    send_lock.release()
                if gate_wait or flood_wait:
                    continue
                break
            _log_ordinary("owner_action_sent", task_id)
        except asyncio.CancelledError:
            if sending_started and active_part:
                try:
                    state.finish_ordinary_part(
                        chat_id, command_id, active_part, "uncertain",
                    )
                    state.status(chat_id, command_id, "uncertain")
                    state.finish_owner_action(chat_id, command_id, "uncertain")
                    reservation_active = False
                except Exception:
                    pass
                _log_ordinary("owner_action_send_uncertain", task_id)
            else:
                finish_skip("owner_action_cancelled", final_status="cancelled")
        except Exception as exc:
            if sending_started and active_part:
                try:
                    state.finish_ordinary_part(
                        chat_id, command_id, active_part, "uncertain",
                    )
                    state.status(chat_id, command_id, "uncertain")
                    state.finish_owner_action(chat_id, command_id, "uncertain")
                    reservation_active = False
                except Exception:
                    pass
                _log_ordinary("owner_action_send_uncertain", task_id)
            else:
                finish_skip("owner_action_error_" + type(exc).__name__)
        finally:
            actions = owner_actions_by_thread.get((chat_id, thread_id))
            if actions is not None and actions.get(int(command_id)) is task:
                actions.pop(int(command_id), None)
                if not actions:
                    owner_actions_by_thread.pop((chat_id, thread_id), None)
            active_owner_actions.discard(task)
            started_owner_actions.discard(task)

    async def ordinary_worker(ev, generation, trigger_reasons, extra_parent,
                              participant_summary, recipient_vocatives, trigger_snapshot,
                              task_id, deadline, key, conversation_anchor_id=None):
        nonlocal send_not_before
        started_ordinary.add(asyncio.current_task())
        loop = asyncio.get_running_loop()
        worker_started_at = loop.time()
        model_ms = 0
        message = ev.message
        chat_id = int(ev.chat_id)
        trigger_id = int(message.id)
        thread_id = thread_key(message)
        reservation_active = False
        sending_started = False
        active_part = 0
        first_outbound_id = None
        task_key = (chat_id, trigger_id)

        def finish_skip(reason, *, superseded_task=False):
            nonlocal reservation_active
            if reservation_active:
                state.finish_ordinary_reply(chat_id, trigger_id, "released")
                reservation_active = False
            if not superseded_task and task_key not in superseded:
                progress = state.ordinary_progress(chat_id, trigger_id)
                final_status = progress["status"]
                if final_status not in {"partial", "sent", "uncertain"}:
                    final_status = "skip"
                state.status(chat_id, trigger_id, final_status)
            _log_ordinary(reason, task_id)

        try:
            if not ordinary_generation_is_active(generation):
                finish_skip("paused")
                return

            allowed, reservation_reason = state.reserve_ordinary_reply(
                chat_id, trigger_id, dry_run=not send,
            )
            if not allowed:
                finish_skip(reservation_reason)
                return
            reservation_active = True

            fallback_reason = None
            repair_timeout = False
            parts = None
            result = None
            sticker_part = None
            if profile is None:
                fallback_reason = profile_error or "profile_unavailable"
            else:
                before_count = max(0, int(getattr(cfg, "reply_context_before", 5)))
                after_count = max(0, int(getattr(cfg, "reply_context_after", 3)))
                wait_seconds = max(0, int(getattr(cfg, "reply_wait_seconds", 1)))
                remaining = deadline - loop.time()
                if remaining <= 0:
                    fallback_reason = "reply_generation_timeout"
                else:
                    try:
                        await context.wait_for_after(
                            chat_id, thread_id, trigger_id, after_count,
                            min(float(wait_seconds), remaining),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        fallback_reason = "context_unavailable"
                # Latest-wins only combines one sender's triggers during this wait. Once
                # history/model work begins, a new trigger gets its own task.
                collecting_ordinary.discard(asyncio.current_task())
                if not ordinary_generation_is_active(generation):
                    finish_skip("paused")
                    return

                if fallback_reason is None:
                    if loop.time() >= deadline:
                        fallback_reason = "reply_generation_timeout"
                    else:
                        try:
                            history = await get_history_before(
                                ev, message, thread_id, before_count,
                                deadline=deadline, task_id=task_id,
                            )
                            window = context.make_window(
                                chat_id, thread_id, trigger_id, own_id, before_count, after_count,
                                extra=extra_parent, history=history,
                                trigger_snapshot=trigger_snapshot,
                                trigger_reason=("_".join(trigger_reasons)
                                                if trigger_reasons else "mention"),
                                pinned_context_ids=(
                                    (conversation_anchor_id,)
                                    if "conversation" in trigger_reasons
                                    and conversation_anchor_id is not None else ()
                                ),
                            )
                            if not window.get("messages"):
                                fallback_reason = "context_unavailable"
                            else:
                                window = sanitize_window(window)
                        except asyncio.CancelledError:
                            raise
                        except (TypeError, ValueError):
                            fallback_reason = "context_private_data"
                        except Exception:
                            fallback_reason = "context_unavailable"

                if fallback_reason is None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        fallback_reason = "reply_generation_timeout"
                    else:
                        model_started_at = loop.time()
                        model_deadline = model_started_at + min(
                            ORDINARY_GENERATION_TIMEOUT, remaining,
                        )
                        acquired = False
                        repair_state = {"started": False}
                        sticker_emojis = []
                        try:
                            async with asyncio.timeout_at(model_deadline):
                                await ordinary_model_slots.acquire()
                                acquired = True
                                if not ordinary_generation_is_active(generation):
                                    finish_skip("paused")
                                    return
                                if recent_stickers is not None:
                                    catalog_refresh_count = getattr(
                                        recent_stickers, "refresh_count", 0,
                                    )
                                    try:
                                        catalog_deadline = min(
                                            model_deadline, loop.time() + 6.0,
                                        )
                                        async with asyncio.timeout_at(catalog_deadline):
                                            await recent_stickers.get(telegram)
                                        sticker_emojis = list(recent_stickers.emojis)
                                    except asyncio.CancelledError:
                                        raise
                                    except Exception:
                                        # Recent-sticker lookup is optional; text replies
                                        # remain available when Telegram cannot serve it.
                                        sticker_emojis = []
                                    if (not sticker_emojis
                                            and getattr(recent_stickers, "refresh_count", 0)
                                            != catalog_refresh_count):
                                        _log_ordinary(
                                            "sticker_catalog_" + getattr(
                                                recent_stickers, "catalog_status", "unavailable",
                                            ),
                                            task_id,
                                        )
                                reply_kwargs = {}
                                if participant_summary is not None:
                                    reply_kwargs["participant_summary"] = participant_summary
                                if recipient_vocatives:
                                    reply_kwargs["recipient_vocatives"] = recipient_vocatives
                                if sticker_emojis:
                                    reply_kwargs["sticker_catalog"] = sticker_emojis
                                try:
                                    parameters = inspect.signature(generate_reply).parameters
                                except (TypeError, ValueError):
                                    parameters = {}
                                if "on_repair_start" in parameters:
                                    reply_kwargs["on_repair_start"] = (
                                        lambda: repair_state.__setitem__("started", True)
                                    )
                                result = await generate_reply(
                                    cfg.gemini_key,
                                    getattr(cfg, "reply_model", "") or cfg.model,
                                    profile, window, **reply_kwargs,
                                )
                        except asyncio.CancelledError:
                            raise
                        except asyncio.TimeoutError:
                            if repair_state["started"]:
                                repair_timeout = True
                            else:
                                fallback_reason = "reply_generation_timeout"
                        except Exception:
                            fallback_reason = "reply_generation_error"
                        finally:
                            if acquired:
                                ordinary_model_slots.release()
                            model_ms = (loop.time() - model_started_at) * 1000

                        if not ordinary_generation_is_active(generation):
                            finish_skip("paused")
                            return
                        if repair_timeout:
                            # A repair draft timed out after the first draft was
                            # rejected. Do not replace it with the repeated generic
                            # fallback that caused prior duplicate replies.
                            finish_skip("reply_validation_retry_timeout")
                            return
                        if fallback_reason is None:
                            result_text = getattr(result, "text", None)
                            result_reason = getattr(result, "reason", "reply_invalid_result")
                            if (isinstance(result_reason, str)
                                    and (result_reason.startswith("reply_validation_retry_failed_")
                                         or (result_reason.startswith("reply_validation_failed_")
                                             and result_reason != "reply_validation_failed"))):
                                finish_skip(result_reason)
                                return
                            if not isinstance(result_text, str) or not result_text or result_reason != "ok":
                                fallback_reason = (
                                    result_reason if result_reason != "ok"
                                    else "reply_invalid_result"
                                )
                            else:
                                parts = [result_text]
                                follow_up = getattr(result, "follow_up", None)
                                requested_emoji = getattr(result, "sticker_emoji", None)
                                candidate_sticker = (
                                    recent_stickers.find(requested_emoji)
                                    if (requested_emoji and recent_stickers is not None
                                        and requested_emoji in sticker_emojis)
                                    else None
                                )
                                last_pack = last_sticker_pack_by_chat.get(chat_id)
                                candidate_pack_key = (
                                    candidate_sticker.pack_key
                                    or str(candidate_sticker.pack_priority)
                                    if candidate_sticker is not None else None
                                )
                                if (candidate_sticker is not None
                                        and last_pack == candidate_pack_key):
                                    find_alternative = getattr(
                                        recent_stickers, "find_alternative", None,
                                    )
                                    if callable(find_alternative):
                                        try:
                                            alternative = find_alternative(
                                                requested_emoji,
                                                exclude_pack_keys={last_pack},
                                            )
                                        except TypeError:
                                            alternative = find_alternative(requested_emoji)
                                        candidate_sticker = alternative or candidate_sticker
                                if candidate_sticker is not None:
                                    last_sticker_key, same_sticker_run = (
                                        state.last_sticker_sent(chat_id)
                                    )
                                    candidate_key = sticker_document_key(
                                        candidate_sticker.document,
                                        candidate_sticker.emoji,
                                    )
                                    if (candidate_key == last_sticker_key
                                            and same_sticker_run >= 2):
                                        find_alternative = getattr(
                                            recent_stickers, "find_alternative", None,
                                        )
                                        if callable(find_alternative):
                                            try:
                                                candidate_sticker = find_alternative(
                                                    requested_emoji,
                                                    exclude_document_ids={last_sticker_key},
                                                    exclude_pack_keys=(
                                                        {last_pack} if last_pack is not None else set()
                                                    ),
                                                )
                                            except TypeError:
                                                candidate_sticker = find_alternative(
                                                    requested_emoji,
                                                    exclude_document_ids={last_sticker_key},
                                                )
                                        else:
                                            candidate_sticker = None
                                if candidate_sticker is not None:
                                    sticker_allowed, sticker_limit_reason = (
                                        state.sticker_send_allowed(
                                            chat_id,
                                            sticker_document_key(
                                                candidate_sticker.document,
                                                candidate_sticker.emoji,
                                            ),
                                            reply_offset=1,
                                        )
                                    )
                                    if not sticker_allowed:
                                        _log_ordinary(sticker_limit_reason, task_id)
                                        candidate_sticker = None
                                if candidate_sticker is not None:
                                    sticker_part = candidate_sticker
                                if sticker_part is not None:
                                    part_two_allowed, part_two_reason = state.reserve_ordinary_followup(
                                        chat_id, trigger_id, dry_run=not send,
                                    )
                                    if not part_two_allowed:
                                        _log_ordinary(
                                            "sticker_downgraded_" + part_two_reason,
                                            task_id,
                                        )
                                        sticker_part = None
                                if sticker_part is not None:
                                    follow_up = None
                                    parts.append(sticker_part.emoji)
                                if follow_up:
                                    followup_allowed, followup_reason = state.reserve_ordinary_followup(
                                        chat_id, trigger_id, dry_run=not send,
                                    )
                                    if followup_allowed:
                                        parts.append(follow_up)
                                    else:
                                        _log_ordinary(
                                            "follow_up_downgraded_" + followup_reason,
                                            task_id,
                                        )

            if fallback_reason is not None:
                collecting_ordinary.discard(asyncio.current_task())
                if "conversation" in trigger_reasons:
                    # A continuation is opportunistic. If context/model
                    # validation cannot produce a grounded reply, stay silent
                    # instead of emitting a canned clarification.
                    finish_skip("conversation_silent_" + str(fallback_reason))
                    return
                recent_safe_replies = needs_owner_fallback_history.get(chat_id, [])
                fallback_code, fallback_text = _fallback_for_reason(
                    fallback_reason, recent=recent_safe_replies,
                )
                fallback_text, fallback_failure = format_safe_fallback(fallback_text)
                if fallback_failure or not fallback_text:
                    finish_skip("fallback_validation_failed_" + (fallback_failure or "empty"))
                    return
                if fallback_code == "needs_owner":
                    recent_safe_replies.append(_fallback_for_reason(
                        fallback_reason, recent=recent_safe_replies,
                    )[1])
                    needs_owner_fallback_history[chat_id] = recent_safe_replies[-2:]
                _log_ordinary(
                    "reply_fallback_" + fallback_code, task_id, model_ms=model_ms,
                )
                parts = [fallback_text]
                sticker_part = None

            if not ordinary_generation_is_active(generation):
                finish_skip("paused")
                return
            owner_check = await owner_replied_after(
                ev, trigger_id, thread_id, tail_check=False,
                deadline=deadline, task_id=task_id,
            )
            if loop.time() >= deadline:
                finish_skip("deadline_expired")
                return
            if owner_check is True:
                finish_skip("owner_already_replied")
                return
            if not send:
                state.status(chat_id, trigger_id, "dry")
                state.finish_ordinary_reply(chat_id, trigger_id, "dry")
                reservation_active = False
                _log_ordinary("dry_run_ready", task_id)
                if show_drafts:
                    for part_number, part_text in enumerate(parts, start=1):
                        if part_number == 2 and sticker_part is not None:
                            print(f"Черновик стикера [{task_id}]: {sticker_part.emoji}")
                        else:
                            print(f"Черновик обычного ответа {part_number}/{len(parts)} [{task_id}]: {part_text}")
                return
            if not ev.is_group or chat_id not in cfg.chats:
                finish_skip("send_guard")
                return

            for part_number, part_text in enumerate(parts, start=1):
                if part_number == 2 and first_outbound_id is None:
                    finish_skip("first_outbound_id_unavailable")
                    return
                while True:
                    loop = asyncio.get_running_loop()
                    if loop.time() >= deadline:
                        finish_skip("deadline_expired")
                        return
                    waited_gate = await wait_for_send_gate(deadline)
                    if waited_gate is None:
                        finish_skip("deadline_expired")
                        return
                    gate_wait = False
                    flood_wait = False
                    try:
                        async with asyncio.timeout_at(deadline):
                            await send_lock.acquire()
                    except asyncio.TimeoutError:
                        finish_skip("send_lock_timeout")
                        return
                    try:
                        if send_not_before > waited_gate:
                            gate_wait = True
                        elif not ordinary_generation_is_active(generation):
                            finish_skip("paused")
                            return
                        elif not ev.is_group or chat_id not in cfg.chats:
                            finish_skip("send_guard")
                            return
                        else:
                            excluded = {first_outbound_id} if part_number == 2 else set()
                            owner_check = await owner_replied_after(
                                ev, trigger_id, thread_id, tail_check=True,
                                deadline=deadline, task_id=task_id,
                                exclude_message_ids=excluded,
                            )
                            if loop.time() >= deadline:
                                finish_skip("deadline_expired")
                                return
                            if not ordinary_generation_is_active(generation):
                                finish_skip("paused")
                                return
                            if owner_check is True:
                                finish_skip("owner_already_replied")
                                return
                            if owner_check is None:
                                finish_skip("owner_tail_unavailable")
                                return
                            if send_not_before > waited_gate:
                                gate_wait = True
                            else:
                                if part_number == 2 and sticker_part is not None:
                                    sticker_allowed, sticker_limit_reason = (
                                        state.sticker_send_allowed(
                                            chat_id,
                                            sticker_document_key(
                                                sticker_part.document,
                                                sticker_part.emoji,
                                            ),
                                        )
                                    )
                                    if not sticker_allowed:
                                        if not state.skip_ordinary_followup(
                                                chat_id, trigger_id):
                                            finish_skip("sticker_reservation_unavailable")
                                            return
                                        state.status(chat_id, trigger_id, "sent")
                                        reservation_active = False
                                        _log_ordinary(sticker_limit_reason, task_id)
                                        break
                                allowed, reservation_reason = state.start_ordinary_part(
                                    chat_id, trigger_id, part_number,
                                )
                                if not allowed:
                                    finish_skip(reservation_reason)
                                    return
                                active_part = part_number
                                sending_started = True
                                state.status(chat_id, trigger_id, "sending")
                                ordinary_sending.add(key)
                                # No await is allowed between these final guards and send.
                                if (loop.time() >= deadline or not ordinary_generation_is_active(generation)):
                                    state.finish_ordinary_part(
                                        chat_id, trigger_id, part_number, "released",
                                    )
                                    active_part = 0
                                    sending_started = False
                                    ordinary_sending.discard(key)
                                    finish_skip("deadline_expired" if loop.time() >= deadline else "paused")
                                    return
                                if send_not_before > waited_gate:
                                    state.defer_ordinary_part(chat_id, trigger_id, part_number)
                                    active_part = 0
                                    sending_started = False
                                    ordinary_sending.discard(key)
                                    gate_wait = True
                                else:
                                    sticker_sent_at = None
                                    try:
                                        async with asyncio.timeout_at(deadline):
                                            if part_number == 2 and sticker_part is not None:
                                                sticker_sent_at = time.time()
                                                state.reserve_sticker_cooldown(
                                                    chat_id, cooldown_seconds=0,
                                                    now=sticker_sent_at,
                                                )
                                                sent_message = await telegram.send_file(
                                                    chat_id, sticker_part.document,
                                                    reply_to=(trigger_id if part_number == 1 else None),
                                                )
                                                try:
                                                    state.record_sticker_sent(
                                                        chat_id,
                                                        sticker_document_key(
                                                            sticker_part.document,
                                                            sticker_part.emoji,
                                                        ),
                                                    )
                                                except Exception:
                                                    print("ordinary: sticker_state_persist_unavailable")
                                                remember_confirmed_sticker(
                                                    sent_message, chat_id,
                                                    trigger_id if part_number == 1 else None,
                                                    thread_id, sticker_part.emoji,
                                                )
                                                last_sticker_pack_by_chat[chat_id] = (
                                                    sticker_part.pack_key
                                                    or str(sticker_part.pack_priority)
                                                )
                                            else:
                                                sent_message = await telegram.send_message(
                                                    chat_id, part_text,
                                                    reply_to=(trigger_id if part_number == 1 else None),
                                                    parse_mode=None, link_preview=False,
                                                )
                                                remember_confirmed_reply(
                                                    sent_message, chat_id,
                                                    trigger_id if part_number == 1 else None,
                                                    thread_id,
                                                )
                                            overall = state.finish_ordinary_part(
                                                chat_id, trigger_id, part_number, "sent",
                                            )
                                            state.status(chat_id, trigger_id, overall)
                                            if part_number == 1:
                                                try:
                                                    first_outbound_id = int(getattr(sent_message, "id", 0)) or None
                                                except (TypeError, ValueError):
                                                    first_outbound_id = None
                                                if (first_outbound_id is not None
                                                        and result is not None
                                                        and getattr(result, "reason", None) == "ok"):
                                                    open_followup_session(
                                                        chat_id, thread_id,
                                                        first_outbound_id,
                                                    )
                                            reservation_active = overall == "partial"
                                            active_part = 0
                                            sending_started = False
                                    except Exception as exc:
                                        if type(exc).__name__ == "FloodWaitError":
                                            flood_wait = True
                                            if sticker_sent_at is not None:
                                                state.release_sticker_cooldown(
                                                    chat_id, reserved_at=sticker_sent_at,
                                                )
                                            seconds = max(1, int(getattr(exc, "seconds", 1)))
                                            send_not_before = max(
                                                send_not_before, loop.time() + seconds,
                                            )
                                            state.defer_ordinary_part(
                                                chat_id, trigger_id, part_number,
                                            )
                                            state.status(
                                                chat_id, trigger_id,
                                                "partial" if part_number > 1 else "pending",
                                            )
                                            active_part = 0
                                            sending_started = False
                                        else:
                                            state.finish_ordinary_part(
                                                chat_id, trigger_id, part_number, "uncertain",
                                            )
                                            state.status(chat_id, trigger_id, "uncertain")
                                            reservation_active = False
                                            active_part = 0
                                            sending_started = False
                                            _log_ordinary(
                                                "telegram_send_timeout_uncertain"
                                                if isinstance(exc, asyncio.TimeoutError)
                                                else "telegram_send_uncertain", task_id,
                                            )
                                            return
                                    finally:
                                        ordinary_sending.discard(key)
                    finally:
                        send_lock.release()
                    if gate_wait or flood_wait:
                        continue
                    break
            _log_ordinary("sent", task_id)
        except asyncio.CancelledError:
            if sending_started and active_part:
                state.finish_ordinary_part(
                    chat_id, trigger_id, active_part, "uncertain",
                )
                state.status(chat_id, trigger_id, "uncertain")
                reservation_active = False
                _log_ordinary("send_cancelled_uncertain", task_id)
            else:
                finish_skip("cancelled", superseded_task=task_key in superseded)
        except asyncio.TimeoutError:
            finish_skip("reply_timeout")
        except Exception as exc:
            if sending_started and active_part:
                try:
                    state.finish_ordinary_part(
                        chat_id, trigger_id, active_part, "uncertain",
                    )
                    state.status(chat_id, trigger_id, "uncertain")
                    reservation_active = False
                except Exception:
                    pass
                _log_ordinary("telegram_send_uncertain", task_id)
            else:
                finish_skip("ordinary_error_" + type(exc).__name__)
        finally:
            _log_ordinary(
                "timing", task_id, model_ms=model_ms,
                total_ms=(loop.time() - worker_started_at) * 1000,
            )
            if task_key in superseded:
                superseded.discard(task_key)
            current = ordinary_by_thread.get(key)
            if current is not None and current[0] is asyncio.current_task():
                ordinary_by_thread.pop(key, None)
            active_ordinary.discard(asyncio.current_task())

    async def queue_ordinary(ev, generation, reasons, parent_snapshot,
                             participant_summary=None, recipient_vocatives=(), *,
                             deadline, trigger_snapshot=None,
                             conversation_anchor_id=None,
                             task_id=None):
        message = ev.message
        chat_id = int(ev.chat_id)
        msg_id = int(message.id)
        thread_id = thread_key(message)
        key = (chat_id, thread_id)
        sender_id = getattr(message, "sender_id", None)
        task_id = task_id or uuid.uuid4().hex[:10]
        if not state.claim(chat_id, msg_id, "", "pending", kind="ordinary"):
            _log_ordinary("duplicate_message", task_id)
            return True
        if not ordinary_generation_is_active(generation):
            state.status(chat_id, msg_id, "skip")
            _log_ordinary("paused", task_id)
            return True
        if asyncio.get_running_loop().time() >= deadline:
            state.status(chat_id, msg_id, "skip")
            _log_ordinary("deadline_expired", task_id)
            return True
        previous = ordinary_by_thread.get(key)
        if (previous is not None and previous[0] in collecting_ordinary
                and sender_id is not None and previous[3] == sender_id
                and not previous[0].done() and not previous[0].cancelling()):
            old_task, old_msg_id, old_task_id, _old_sender_id = previous
            superseded.add((chat_id, old_msg_id))
            state.status(chat_id, old_msg_id, "superseded")
            _log_ordinary("superseded", old_task_id)
            old_task.cancel()
        task = asyncio.create_task(
            ordinary_worker(
                ev, generation, reasons, parent_snapshot, participant_summary,
                tuple(recipient_vocatives),
                trigger_snapshot,
                task_id, deadline, key, conversation_anchor_id,
            )
        )
        active_ordinary.add(task)
        ordinary_task_triggers[task] = (chat_id, msg_id)
        collecting_ordinary.add(task)
        ordinary_by_thread[key] = (task, msg_id, task_id, sender_id)

        def task_finished(finished):
            # Cancellation before the coroutine's first step never runs its
            # finally block. Always clean up, without deleting a newer task.
            try:
                if finished.cancelled() and finished not in started_ordinary:
                    if (chat_id, msg_id) not in superseded:
                        state.status(chat_id, msg_id, "skip")
                    _log_ordinary("cancelled_before_start", task_id)
            finally:
                active_ordinary.discard(finished)
                ordinary_task_triggers.pop(finished, None)
                collecting_ordinary.discard(finished)
                started_ordinary.discard(finished)
                superseded.discard((chat_id, msg_id))
                current = ordinary_by_thread.get(key)
                if current is not None and current[0] is finished:
                    ordinary_by_thread.pop(key, None)

        task.add_done_callback(task_finished)
        return True

    async def queue_owner_action(ev, generation, deadline, *, requester_sender_id,
                                 recipient_reference=None,
                                 trigger_reason="group_action"):
        message = ev.message
        chat_id = int(ev.chat_id)
        command_id = int(message.id)
        thread_id = thread_key(message)
        key = (chat_id, thread_id)
        task_id = uuid.uuid4().hex[:10]
        try:
            requester_sender_id = int(requester_sender_id)
        except (TypeError, ValueError):
            _log_ordinary("owner_action_requester_invalid", task_id)
            return True
        if not state.claim(chat_id, command_id, "owner_action", "pending",
                           kind="owner_action"):
            _log_ordinary("owner_action_duplicate", task_id)
            return True
        requester_id = requester_sender_id
        state.record_owner_action(
            chat_id, command_id, thread=thread_id, status="resolving",
        )
        if (not ordinary_generation_is_active(generation)
                or asyncio.get_running_loop().time() >= deadline):
            state.status(chat_id, command_id, "skip")
            state.finish_owner_action(chat_id, command_id, "cancelled")
            _log_ordinary("owner_action_inactive", task_id)
            return True

        try:
            resolved = await resolve_owner_action_target(
                ev, deadline, task_id,
                recipient_reference=recipient_reference,
                requester_sender_id=requester_sender_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _log_ordinary("owner_action_resolution_unavailable", task_id)
            resolved = None
        request = None
        trigger_snapshot = None
        summary = None
        recipient_vocatives = ()
        if resolved is None:
            state.finish_owner_action(chat_id, command_id, "ambiguous")
        else:
            (target_id, target_sender_id, _target_snapshot, summary,
             recipient_username) = resolved
            trigger_snapshot = snapshot_from_message(message, chat_id, own_id)
            relationship = (
                summary.get("relationship", {})
                if isinstance(summary, dict) else {}
            )
            forms = (
                relationship.get("forms_of_address", ())
                if isinstance(relationship, dict) else ()
            )
            if isinstance(forms, (list, tuple)):
                recipient_vocatives = tuple(
                    form for form in forms if isinstance(form, str)
                )
            state.resolve_owner_action(
                chat_id, command_id, target=target_id,
                target_sender=target_sender_id, thread=thread_id,
            )
            cancel_ordinary_for_target(chat_id, thread_id, target_id)
            request = ActionRequest(
                command_id, chat_id, thread_id, target_id, target_sender_id,
                requester_sender_id, trigger_reason,
                tuple(generation), float(deadline), recipient_username,
            )

        # A requester or an authorized owner may cancel while the target is
        # being resolved. Check the frozen same-thread buffer before creating
        # a worker so a later message cannot race into an outbound reply.
        if owner_action_cancelled_by_followup(
                chat_id, thread_id, command_id, requester_id):
            state.status(chat_id, command_id, "skip")
            state.finish_owner_action(chat_id, command_id, "cancelled")
            _log_ordinary("owner_action_requester_followup", task_id)
            return True

        task = asyncio.create_task(
            owner_action_worker(
                ev, generation, deadline, command_id, thread_id, task_id,
                request=request, trigger_snapshot=trigger_snapshot,
                participant_summary=summary,
                recipient_vocatives=recipient_vocatives,
                requester_sender_id=requester_sender_id,
            )
        )
        active_owner_actions.add(task)
        owner_actions_by_thread.setdefault(key, {})[command_id] = task
        owner_action_requesters[(chat_id, thread_id, command_id)] = requester_id

        def action_task_finished(finished):
            if finished.cancelled() and finished not in started_owner_actions:
                try:
                    state.status(chat_id, command_id, "skip")
                    state.finish_owner_action(chat_id, command_id, "cancelled")
                except Exception:
                    pass
                _log_ordinary("owner_action_cancelled_before_start", task_id)
            active_owner_actions.discard(finished)
            actions = owner_actions_by_thread.get(key)
            if actions is not None and actions.get(command_id) is finished:
                actions.pop(command_id, None)
                if not actions:
                    owner_actions_by_thread.pop(key, None)
            owner_action_requesters.pop((chat_id, thread_id, command_id), None)
        task.add_done_callback(action_task_finished)
        return True

    async def video_worker():
        nonlocal send_not_before
        while True:
            ev, generation = await video_queue.get()
            claimed = False
            try:
                m = ev.message
                message_text = m.message or ""
                entities = m.entities or []
                urls = extract_urls(message_text, entities)
                if not urls:
                    if has_url_candidate(message_text, entities):
                        _log_skip("unsupported_url", m.id)
                    continue
                if ev.chat_id not in cfg.chats:
                    _log_skip("chat_not_allowed", m.id)
                    continue
                if not ev.is_group or getattr(ev.chat, "broadcast", False):
                    _log_skip("not_group", m.id)
                    continue
                if m.out or m.sender_id == own_id:
                    _log_skip("own_message", m.id)
                    continue
                if not video_generation_is_active(generation):
                    skip_for_inactive_video(ev, urls[0], False, generation)
                    continue
                sender = await ev.get_sender()
                if not video_generation_is_active(generation):
                    skip_for_inactive_video(ev, urls[0], False, generation)
                    continue
                if isinstance(sender, User) and (sender.bot or sender.id == own_id):
                    _log_skip("sender_bot_or_self", m.id)
                    continue
                if not state.claim(ev.chat_id, m.id, urls[0], "pending", kind="video"):
                    _log_skip("duplicate_message", m.id)
                    continue
                claimed = True
                if not video_generation_is_active(generation):
                    skip_for_inactive_video(ev, urls[0], True, generation)
                    continue
                platform_info = supported_url(urls[0])
                if not platform_info:
                    state.status(ev.chat_id, m.id, "skip")
                    _log_skip("unsupported_url", m.id)
                    continue
                metadata_url, _platform_name = platform_info
                try:
                    meta = await asyncio.to_thread(fetch, metadata_url)
                except MetadataError as exc:
                    if str(exc) not in {"metadata_unavailable", "metadata_timeout"}:
                        raise
                    _log_fallback(f"{exc}_laugh_only", m.id)
                    meta = {"id": "", "title": "", "description": "", "url": metadata_url}
                if not video_generation_is_active(generation):
                    skip_for_inactive_video(ev, urls[0], True, generation)
                    continue
                title = str(meta.get("title") or "")[:300]
                description = str(meta.get("description") or "")[:1500]
                if not title.strip() and not description.strip():
                    text = laugh_only()
                    _log_fallback("metadata_empty_laugh_only", m.id)
                else:
                    text = await draft(cfg.gemini_key, cfg.model, title, description)
                if not video_generation_is_active(generation):
                    skip_for_inactive_video(ev, urls[0], True, generation)
                    continue
                if not text:
                    state.status(ev.chat_id, m.id, "skip")
                    _log_skip("unsuitable_content", m.id)
                    continue
                if not send:
                    print(f"Черновик для сообщения {m.id}: {text}")
                    state.status(ev.chat_id, m.id, "dry")
                    continue
                if not ev.is_group or ev.chat_id not in cfg.chats:
                    state.status(ev.chat_id, m.id, "skip")
                    _log_skip("send_guard", m.id)
                    continue
                while True:
                    waited_gate = await wait_for_send_gate()
                    flood_wait = None
                    gate_wait = False
                    loop = asyncio.get_running_loop()
                    async with send_lock:
                        if send_not_before > waited_gate:
                            gate_wait = True
                        elif not video_generation_is_active(generation):
                            skip_for_inactive_video(ev, urls[0], True, generation)
                            break
                        else:
                            state.status(ev.chat_id, m.id, "sending")
                            try:
                                await telegram.send_message(ev.chat_id, text, reply_to=m.id,
                                                            parse_mode=None, link_preview=False)
                                state.status(ev.chat_id, m.id, "sent")
                            except Exception as ex:
                                if type(ex).__name__ == "FloodWaitError":
                                    flood_wait = max(1, int(getattr(ex, "seconds", 1)))
                                    send_not_before = max(
                                        send_not_before, loop.time() + flood_wait,
                                    )
                                else:
                                    state.status(ev.chat_id, m.id, "uncertain")
                                    _log_skip("telegram_send_uncertain", m.id)
                                    break
                    if gate_wait:
                        continue
                    if flood_wait is None:
                        break
                    print(f"Telegram просит паузу {flood_wait} секунд.")
                    print(f"wait: telegram_flood_wait seconds={flood_wait} message={m.id}")
                    # The shared gate also delays the other sender path.
            except Exception as ex:
                if state.seen(ev.chat_id, ev.message.id):
                    state.status(ev.chat_id, ev.message.id, "skip")
                safe = str(ex) if str(ex) in SAFE_ERROR_CODES else "processing_error_" + type(ex).__name__
                _log_skip(safe, ev.message.id)
            finally:
                video_queue.task_done()

    worker_task = None
    context_prune_task = None
    try:
        if ordinary_requested:
            context_prune_task = asyncio.create_task(
                _prune_ordinary_context(state, context, cfg.chats, ordinary_is_enabled),
            )
        await telegram.connect()
        if not await telegram.is_user_authorized():
            raise RuntimeError("telegram_not_logged_in_run_login")
        me = await telegram.get_me()
        own_id = me.id
        own_username = getattr(me, "username", None)
        if control_enabled:
            control_bot = create_control_client(cfg)
            try:
                install_control_handler(
                    control_bot, state, own_id, send_lock,
                    on_pause=cancel_ordinary_for_pause,
                    status_provider=ordinary_status_text,
                    on_ordinary_toggle=on_ordinary_toggle,
                    ordinary_available=ordinary_requested,
                )
                await control_bot.start(bot_token=cfg.control_bot_token)
                if not await register_command_menu(cfg.control_bot_token, own_id):
                    print("control: command_menu_registration_unavailable")
            except Exception:
                raise RuntimeError("control_bot_unavailable") from None
        print(f"Режим: {'ОТПРАВКА' if send else 'dry-run'}; разрешенных групп: {len(cfg.chats)}")
        print("Проверяются только новые входящие ссылки в группах; собственные сообщения аккаунта пропускаются.")
        if ordinary_requested and ordinary_is_enabled():
            state_label = "готов" if profile is not None else "пропускает задачи: профиль или конфигурация недоступны"
            print(f"Обычные ответы: включены, backend Gemini, состояние: {state_label}.")
        elif ordinary_requested:
            print("Обычные ответы выключены (/talk_on в управляющем боте).")
        else:
            print("Обычные ответы выключены (AUTO_REPLY_ENABLED=false).")
        links_enabled, _ = state.link_mode_snapshot(default_enabled=True)
        print(
            "Ответы на ссылки TikTok, Reels и Shorts: "
            f"{'включены' if links_enabled else 'выключены'}."
        )
        if not send:
            print("Dry-run: ответы не отправляются; для отправки нужен --send.")
        worker_task = asyncio.create_task(video_worker())

        async def dispatch(ev):
            # Keep the link generation from arrival time, even when ordinary
            # trigger lookups below suspend before a video task is enqueued.
            video_generation = video_generation_snapshot()
            message = ev.message
            allowed_group = (
                ev.chat_id in cfg.chats and bool(getattr(ev, "is_group", False))
                and not getattr(getattr(ev, "chat", None), "broadcast", False)
            )
            if not allowed_group:
                if has_url_candidate(message.message or "", message.entities or []):
                    queue_video(ev, video_generation)
                return
            # Sender/parent lookups are part of this trigger's lifetime. Never
            # renew its deadline or adopt a post-pause generation after I/O.
            intake_generation = ordinary_generation_snapshot()
            intake_deadline = asyncio.get_running_loop().time() + ORDINARY_DEADLINE_SECONDS
            snapshot = None
            context_added = True

            if not ordinary_is_enabled():
                if not getattr(message, "out", False) and has_url_candidate(
                    message.message or "", message.entities or []
                ):
                    queue_video(ev, video_generation)
                return
            if own_id is not None:
                snapshot = snapshot_from_message(message, ev.chat_id, own_id)
                if snapshot is not None:
                    context_added = remember_snapshot(snapshot)
            try:
                sender_id = int(getattr(message, "sender_id", 0) or 0)
            except (TypeError, ValueError):
                sender_id = 0
            owner_command_ids = set(
                getattr(cfg, "owner_command_sender_ids", ()) or ()
            )
            if sender_id != own_id:
                cancel_owner_actions_after_owner_message(
                    ev.chat_id, thread_key(message), int(message.id), sender_id,
                    message.message or "",
                )
            if getattr(message, "out", False) or getattr(message, "sender_id", None) == own_id:
                return
            if getattr(message, "fwd_from", None) is not None:
                if has_url_candidate(message.message or "", message.entities or []):
                    queue_video(ev, video_generation)
                return

            people_artifact = participant_artifact(ev.chat_id)
            action_aliases = (
                *getattr(cfg, "owner_name_aliases", ()), *voice_aliases,
            )
            owner_action_command = False
            if sender_id in owner_command_ids:
                owner_action_command = _is_owner_reply_command(
                    message.message or "", action_aliases,
                )
            has_action_verb, recipient_reference = _group_action_intent(
                message.message or "", people_artifact, action_aliases,
            )
            reply_id = getattr(message, "reply_to_msg_id", None)
            action_route = bool(
                getattr(message, "action", None) is None
                and str(message.message or "").strip()
                and (owner_action_command or (
                    has_action_verb and (recipient_reference is not None or reply_id)
                ))
            )

            match = explicit_triggers(
                message, own_id, own_username,
                action_aliases,
            )
            reasons = list(match.reasons)
            participant_query = match_participant_question(message, people_artifact)
            if participant_query is not None and participant_query.should_trigger:
                reasons.append("participant_question")
            participant_summary = (
                participant_query.summary
                if participant_query is not None and participant_query.should_trigger
                else None
            )
            recipient_vocatives = ()
            sender = None
            parent = None
            task_id = uuid.uuid4().hex[:10]

            def reject_stale_intake():
                reason = ""
                if not ordinary_generation_is_active(intake_generation):
                    reason = "paused"
                elif asyncio.get_running_loop().time() >= intake_deadline:
                    reason = "deadline_expired"
                if not reason:
                    return False
                if not state.claim(ev.chat_id, message.id, "", "skip", kind="ordinary"):
                    reason = "duplicate_message"
                _log_ordinary(reason, task_id)
                return True

            conversation_anchor_id = None
            active_anchor_id = None
            if (not reasons and str(message.message or "").strip()
                    and getattr(message, "action", None) is None
                    and not has_url_candidate(
                        message.message or "", message.entities or [],
                    )):
                active_anchor_id = active_followup_anchor(
                    ev.chat_id, thread_key(message),
                )
                # Unaddressed replies belong to their actual parent. During a
                # follow-up session, keep the bot out of a participant-to-
                # participant exchange once a known name starts that exchange.
                if active_anchor_id is not None and reply_id is None:
                    followup_key = (int(ev.chat_id), thread_key(message))
                    if begins_with_participant_address(
                            message.message or "", people_artifact):
                        other_participant_conversation_anchors[followup_key] = (
                            active_anchor_id
                        )
                    if (other_participant_conversation_anchors.get(followup_key)
                            == active_anchor_id):
                        if reject_stale_intake():
                            return
                        if not state.claim(
                                ev.chat_id, message.id, "", "skip",
                                kind="ordinary"):
                            _log_ordinary("duplicate_message", task_id)
                        else:
                            _log_ordinary(
                                "conversation_addressed_to_participant", task_id,
                            )
                        return
                    if reply_id is None:
                        conversation_anchor_id = active_anchor_id

            if reasons or reply_id or action_route or conversation_anchor_id is not None:
                if not context_added:
                    _log_ordinary("context_buffer_full", task_id)
                if reject_stale_intake():
                    return
                try:
                    async with asyncio.timeout(lookup_budget(intake_deadline)):
                        sender = await ev.get_sender()
                except asyncio.TimeoutError:
                    _log_ordinary("sender_lookup_timeout", task_id)
                    reject_stale_intake()
                    return
                except Exception:
                    _log_ordinary("sender_lookup_error", task_id)
                    reject_stale_intake()
                    return
                if reject_stale_intake():
                    return
                if sender is None:
                    _log_ordinary("sender_lookup_unavailable", task_id)
                    return
                try:
                    event_sender_id = int(getattr(sender, "id", 0) or 0)
                except (TypeError, ValueError):
                    event_sender_id = 0
                if event_sender_id != sender_id:
                    _log_ordinary("sender_mismatch", task_id)
                    return
                if getattr(sender, "bot", False):
                    _log_ordinary("sender_bot", task_id)
                    return
                if action_route and not isinstance(sender, User):
                    _log_ordinary("action_sender_not_user", task_id)
                    return
                if action_route:
                    await queue_owner_action(
                        ev, intake_generation, intake_deadline,
                        requester_sender_id=sender_id,
                        recipient_reference=(
                            None if owner_action_command else recipient_reference
                        ),
                        trigger_reason=(
                            "owner_action" if owner_action_command else "group_action"
                        ),
                    )
                    return
                sender_summary = participant_summary_for_sender(
                    sender, people_artifact,
                )
                if sender_summary is not None:
                    relationship = sender_summary.get("relationship", {})
                    sender_forms = (
                        tuple(relationship.get("forms_of_address", ()))
                        if isinstance(relationship, dict) else ()
                    )
                    # A participant-question summary may describe someone
                    # other than the person who sent the question. Never let
                    # that person's nickname become a vocative for the sender.
                    queried_relationship = (
                        participant_summary.get("relationship", {})
                        if isinstance(participant_summary, dict) else {}
                    )
                    queried_forms = (
                        {form.casefold() for form in
                         queried_relationship.get("forms_of_address", ())
                         if isinstance(form, str)}
                        if isinstance(queried_relationship, dict) else set()
                    )
                    recipient_vocatives = tuple(
                        form for form in sender_forms
                        if isinstance(form, str) and form.casefold() not in queried_forms
                    )
                    if participant_query is None and participant_summary is None:
                        participant_summary = sender_summary
            if reply_id:
                try:
                    async with asyncio.timeout(lookup_budget(intake_deadline)):
                        parent = await ev.get_reply_message()
                except asyncio.TimeoutError:
                    _log_ordinary("reply_parent_timeout", task_id)
                except Exception:
                    _log_ordinary("reply_parent_lookup_error", task_id)
                else:
                    if parent is None:
                        _log_ordinary("reply_parent_unavailable", task_id)
                if reject_stale_intake():
                    return
                if parent is not None:
                    parent_chat = getattr(parent, "chat_id", ev.chat_id)
                    if parent_chat is not None and int(parent_chat) != int(ev.chat_id):
                        _log_ordinary("reply_parent_wrong_chat", task_id)
                        parent = None
                if ("reply" not in reasons and parent is not None
                        and getattr(parent, "sender_id", None) == own_id):
                    reasons.append("reply")
                    other_participant_conversation_anchors.pop(
                        (int(ev.chat_id), thread_key(message)), None,
                    )
                elif (not reasons and parent is not None
                      and getattr(parent, "sender_id", None) != own_id
                      and active_anchor_id is not None):
                    other_participant_conversation_anchors[
                        (int(ev.chat_id), thread_key(message))
                    ] = active_anchor_id
                    if reject_stale_intake():
                        return
                    if not state.claim(
                            ev.chat_id, message.id, "", "skip", kind="ordinary"):
                        _log_ordinary("duplicate_message", task_id)
                    else:
                        _log_ordinary(
                            "conversation_addressed_to_participant", task_id,
                        )
                    return

            if is_addressed_identity_question(
                reasons, message,
                (*getattr(cfg, "owner_name_aliases", ()), *voice_aliases, own_username),
            ):
                reasons.append("identity_question")

            if not reasons and conversation_anchor_id is not None:
                reasons.append("conversation")

            if reasons:
                parent_snapshot = None
                if parent is not None:
                    parent_snapshot = snapshot_from_message(
                        parent, ev.chat_id, own_id,
                    )
                    if parent_snapshot is not None:
                        # Store the Telegram-reported thread as-is. The adjusted
                        # copy below exists only for this one reply window.
                        remember_snapshot(parent_snapshot)
                        # The exact reply parent is trusted same-chat context.
                        # Normalize only its per-request snapshot into the
                        # trigger's thread; global history stays thread-filtered.
                        parent_snapshot = replace(
                            parent_snapshot, thread_id=thread_key(message),
                        )
                normalized_reasons = []
                if "mention_id" in reasons or "mention_username" in reasons:
                    normalized_reasons.append("mention")
                if "name_alias" in reasons or "ai_address" in reasons:
                    normalized_reasons.append("name")
                if "reply" in reasons:
                    normalized_reasons.append("reply")
                if "participant_question" in reasons:
                    normalized_reasons.append("participant_question")
                if "identity_question" in reasons:
                    normalized_reasons.append("identity_question")
                if "conversation" in reasons:
                    normalized_reasons.append("conversation")
                await queue_ordinary(
                    ev, intake_generation, tuple(normalized_reasons), parent_snapshot,
                    participant_summary, recipient_vocatives,
                    deadline=intake_deadline,
                    trigger_snapshot=snapshot, task_id=task_id,
                    conversation_anchor_id=conversation_anchor_id,
                )
                return

            if has_url_candidate(message.message or "", message.entities or []):
                queue_video(ev, video_generation)

        telegram.add_event_handler(dispatch, events.NewMessage(chats=list(cfg.chats)))
        await telegram.run_until_disconnected()
    finally:
        if context_prune_task:
            context_prune_task.cancel()
            try:
                await context_prune_task
            except asyncio.CancelledError:
                pass
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        for task in tuple(active_ordinary):
            if not task.done():
                task.cancel()
        if active_ordinary:
            await asyncio.gather(*tuple(active_ordinary), return_exceptions=True)
        if control_bot:
            try:
                await control_bot.disconnect()
            except Exception:
                pass
        state.close()
        try:
            await telegram.disconnect()
        finally:
            lock.close()
            os.umask(old_umask)
