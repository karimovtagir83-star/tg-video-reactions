"""Bounded in-memory context collection for optional group auto-replies."""

from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass

from .triggers import _entity_spans, thread_key
from .stickers import sticker_context_marker

MAX_GROUPS = 100
MAX_MESSAGES_PER_GROUP = 50
CONTEXT_TTL_SECONDS = 24 * 60 * 60
MAX_MESSAGE_CHARS = 800
MAX_WINDOW_CHARS = 8_000
MAX_WINDOW_MESSAGES = 10
_ADDRESS_REASONS = frozenset({
    "mention", "reply", "name", "mention_name", "mention_reply",
    "name_reply", "mention_name_reply", "ai_address",
})
_TRIGGER_REASONS = _ADDRESS_REASONS | {
    "participant_question", "conversation", "group_action",
} | {
    f"{reason}_{suffix}"
    for reason in _ADDRESS_REASONS
    for suffix in ("participant_question", "identity_question",
                   "participant_question_identity_question")
} | {"owner_action"}

_URL_RE = re.compile(
    r"(?i)(?:\b(?:https?://|ftp://|www\.)[^\s<>()]+"
    r"|\b(?:[\w-]+\.)+[a-zа-яё]{2,}(?::\d{1,5})?(?:[/?#][^\s<>()]*)?)"
)
# Treat compact ASCII scheme URIs as links, including non-web schemes such as
# tg:// and telegram://. Requiring a URI scheme plus a non-space payload keeps
# ordinary sentence punctuation like "Время: 12:30, всё готово." intact.
_URI_SCHEME_RE = re.compile(r"(?i)(?<![\w.+-])[a-z][a-z0-9+.-]{0,31}:[^\s<>\"']+")
_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
_HANDLE_RE = re.compile(r"(?<![\w])@[A-Za-z0-9_]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{5,}(?!\w)")
_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|token|password|пароль|код|секрет)\s*[:=]?\s*\S+"
)
_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b")
@dataclass(frozen=True)
class MessageSnapshot:
    # Telegram identifiers stay local and are stripped before API serialization.
    chat_id: int
    message_id: int
    sender_id: int | None
    own: bool
    thread_id: int
    reply_to_id: int | None
    timestamp: float
    text: str


def _timestamp(message) -> float:
    value = getattr(message, "date", None)
    if value is None:
        return time.time()
    try:
        return value.timestamp()
    except Exception:
        return time.time()


def _remove_entity_spans(text: str, entities) -> str:
    spans = _entity_spans(text, entities)
    for entity in entities or []:
        name = type(entity).__name__
        if name in {"MessageEntityCode", "MessageEntityPre"}:
            start = getattr(entity, "offset", 0)
            length = getattr(entity, "length", 0)
            if length > 0:
                spans.append((start, start + length))
    if not spans:
        return text
    raw = text.encode("utf-16-le", errors="surrogatepass")
    units = len(raw) // 2
    merged = []
    for start, end in sorted((max(0, s), min(units, e)) for s, e in spans if e > s):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    result = bytearray()
    cursor = 0
    for start, end in merged:
        result.extend(raw[cursor * 2:start * 2])
        result.extend(" ".encode("utf-16-le"))
        cursor = end
    result.extend(raw[cursor * 2:])
    return result.decode("utf-16-le", errors="ignore")


def has_unredacted_private_data(text: str) -> bool:
    """Detect residual handles, links, contacts, and secrets before API use."""
    if any(regex.search(text) for regex in (
        _URI_SCHEME_RE, _EMAIL_RE, _URL_RE, _HANDLE_RE, _PHONE_RE, _LONG_NUMBER_RE,
        _TOKEN_RE, _SECRET_RE,
    )):
        return True
    return "```" in text or "`" in text


def sanitize_text(text: str, entities=()) -> str:
    """Redact Telegram entities and direct identifiers while preserving body text."""
    text = unicodedata.normalize("NFC", _remove_entity_spans(str(text or ""), entities))
    text = _URI_SCHEME_RE.sub(" [ссылка] ", text)
    text = _EMAIL_RE.sub(" [контакт] ", text)
    text = _URL_RE.sub(" [ссылка] ", text)
    text = _HANDLE_RE.sub(" [упоминание] ", text)
    text = _PHONE_RE.sub(" [контакт] ", text)
    text = _LONG_NUMBER_RE.sub(" [идентификатор] ", text)
    text = _TOKEN_RE.sub(" [секрет] ", text)
    text = _SECRET_RE.sub(" [секрет] ", text)
    text = re.sub(r"```[\s\S]*?```|`[^`]*`", " [код] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_MESSAGE_CHARS]


def sanitize_window(window: dict) -> dict:
    """Validate and sanitize the only context shape allowed to leave the process."""
    if not isinstance(window, dict) or set(window) - {"messages", "trigger_index", "trigger_reason"}:
        raise ValueError("context_invalid")
    entries = window.get("messages")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_WINDOW_MESSAGES:
        raise ValueError("context_invalid")
    cleaned = []
    chars = 0
    for item in entries:
        if not isinstance(item, dict) or set(item) - {
                "speaker", "text", "reply_to_previous", "reply_parent"}:
            raise ValueError("context_invalid")
        speaker = item.get("speaker")
        text = item.get("text")
        reply_to_previous = item.get("reply_to_previous", False)
        reply_parent = item.get("reply_parent", False)
        if speaker != "self" and not (
            isinstance(speaker, str) and re.fullmatch(r"p[1-9]", speaker)
        ):
            raise ValueError("context_invalid")
        if not isinstance(text, str) or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError("context_invalid")
        if not isinstance(reply_to_previous, bool) or not isinstance(reply_parent, bool):
            raise ValueError("context_invalid")
        clean_text = sanitize_text(text)
        if has_unredacted_private_data(clean_text):
            raise ValueError("context_private_data")
        chars += len(clean_text)
        if chars > MAX_WINDOW_CHARS:
            raise ValueError("context_invalid")
        clean_item = {
            "speaker": speaker,
            "text": clean_text,
            "reply_to_previous": reply_to_previous,
        }
        # This marker carries no Telegram identifiers and distinguishes the
        # exact selected reply parent from unrelated nearby messages.
        if reply_parent:
            clean_item["reply_parent"] = True
        cleaned.append(clean_item)
    try:
        trigger_index = int(window.get("trigger_index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("context_invalid") from exc
    if not 0 <= trigger_index < len(cleaned):
        raise ValueError("context_invalid")
    trigger_reason = window.get("trigger_reason", "mention")
    if trigger_reason not in _TRIGGER_REASONS:
        raise ValueError("context_invalid")
    return {
        "messages": cleaned,
        "trigger_index": trigger_index,
        "trigger_reason": trigger_reason,
    }


def snapshot_from_message(message, chat_id: int, own_id: int) -> MessageSnapshot | None:
    if message is None or getattr(message, "fwd_from", None) is not None:
        return None
    body = getattr(message, "message", None)
    body = body if isinstance(body, str) else ""
    sticker_marker = sticker_context_marker(message)
    if not body.strip() and sticker_marker is None:
        return None
    sender_id = getattr(message, "sender_id", None)
    try:
        sender_id = int(sender_id) if sender_id is not None else None
    except (TypeError, ValueError):
        sender_id = None
    try:
        message_id = int(getattr(message, "id"))
    except (TypeError, ValueError):
        return None
    try:
        own = bool(getattr(message, "out", False) or sender_id == int(own_id))
    except (TypeError, ValueError):
        own = bool(getattr(message, "out", False))
    thread = thread_key(message)
    parent_id = getattr(message, "reply_to_msg_id", None)
    try:
        parent_id = int(parent_id) if parent_id is not None else None
    except (TypeError, ValueError):
        parent_id = None
    clean = sanitize_text(body, getattr(message, "entities", None) or ())
    if sticker_marker:
        clean = " ".join(part for part in (clean, sticker_marker) if part).strip()
        clean = clean[:MAX_MESSAGE_CHARS]
    if not clean:
        return None
    return MessageSnapshot(
        int(chat_id), message_id, sender_id, own, int(thread), parent_id,
        _timestamp(message), clean,
    )


class ContextBuffer:
    """Keep only the latest 50 sanitized text messages for each allowed group."""

    def __init__(self, max_groups: int = MAX_GROUPS,
                 max_messages_per_group: int = MAX_MESSAGES_PER_GROUP,
                 clock=time.time):
        self.max_groups = max_groups
        self.max_messages_per_group = max_messages_per_group
        self._clock = clock
        self._messages: dict[int, deque[MessageSnapshot]] = {}
        # One notifier per chat bounds event state even when a group has many
        # forum topics. A message in another topic may wake a waiter, which
        # rechecks the thread-filtered buffer and continues waiting.
        self._events: dict[int, asyncio.Event] = {}

    def _is_recent(self, snapshot: MessageSnapshot, now: float) -> bool:
        try:
            timestamp = float(snapshot.timestamp)
        except (TypeError, ValueError):
            return False
        return (math.isfinite(timestamp)
                and now - CONTEXT_TTL_SECONDS <= timestamp <= now + 300)

    def prune_expired(self, now: float | None = None) -> int:
        """Remove expired/future snapshots from every in-memory group."""
        now = self._clock() if now is None else float(now)
        removed = 0
        for chat_id, items in tuple(self._messages.items()):
            retained = [item for item in items if self._is_recent(item, now)]
            removed += len(items) - len(retained)
            if retained:
                self._messages[chat_id] = deque(
                    retained[-self.max_messages_per_group:],
                    maxlen=self.max_messages_per_group,
                )
            else:
                self._messages.pop(chat_id, None)
        return removed

    def clear(self) -> int:
        """Drop every in-memory context snapshot and wake any waiting task."""
        removed = sum(len(items) for items in self._messages.values())
        self._messages.clear()
        for event in self._events.values():
            event.set()
        return removed

    def add(self, snapshot: MessageSnapshot, *, now: float | None = None) -> bool:
        now = self._clock() if now is None else float(now)
        self.prune_expired(now)
        if not self._is_recent(snapshot, now):
            return False
        if snapshot.chat_id not in self._messages:
            if len(self._messages) >= self.max_groups:
                return False
            self._messages[snapshot.chat_id] = deque(maxlen=self.max_messages_per_group)
        items = self._messages[snapshot.chat_id]
        if any(item.message_id == snapshot.message_id for item in items):
            return True
        retained = [item for item in items if self._is_recent(item, now)]
        retained.append(snapshot)
        retained.sort(key=lambda item: (item.timestamp, item.message_id))
        self._messages[snapshot.chat_id] = deque(
            retained[-self.max_messages_per_group:],
            maxlen=self.max_messages_per_group,
        )
        event = self._events.get(snapshot.chat_id)
        if event is not None:
            event.set()
        return True

    def hydrate(self, snapshots, *, now: float | None = None) -> int:
        """Restore only caller-selected, already-sanitized recent snapshots."""
        now = self._clock() if now is None else float(now)
        loaded = 0
        for snapshot in sorted(
            snapshots, key=lambda item: (item.timestamp, item.chat_id, item.message_id),
        ):
            if self.add(snapshot, now=now):
                loaded += 1
        return loaded

    def messages(self, chat_id: int, thread_id: int) -> list[MessageSnapshot]:
        self.prune_expired()
        return sorted(
            (item for item in self._messages.get(chat_id, ()) if item.thread_id == thread_id),
            key=lambda item: (item.timestamp, item.message_id),
        )

    async def wait_for_after(self, chat_id: int, thread_id: int, trigger_id: int,
                             needed: int, timeout: float) -> int:
        if needed <= 0 or timeout <= 0:
            return 0
        event = self._events.setdefault(chat_id, asyncio.Event())
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            count = sum(item.message_id > trigger_id for item in self.messages(chat_id, thread_id))
            if count >= needed:
                return count
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return count
            event.clear()
            # Recheck after clearing so an update between the first count and
            # event.clear cannot be lost.
            count = sum(item.message_id > trigger_id for item in self.messages(chat_id, thread_id))
            if count >= needed:
                return count
            try:
                await asyncio.wait_for(event.wait(), remaining)
            except asyncio.TimeoutError:
                return sum(item.message_id > trigger_id for item in self.messages(chat_id, thread_id))

    def owner_replied_after(self, chat_id: int, thread_id: int, trigger_id: int,
                            own_id: int, *, exclude_message_ids=()) -> bool:
        self.prune_expired()
        excluded = {int(value) for value in exclude_message_ids if value is not None}
        return any(
            item.chat_id == chat_id
            and item.message_id > trigger_id
            and item.thread_id == thread_id
            and item.sender_id == own_id
            and item.reply_to_id == trigger_id
            and item.message_id not in excluded
            for item in self._messages.get(chat_id, ())
        )

    def make_window(self, chat_id: int, thread_id: int, trigger_id: int,
                    own_id: int, before: int, after: int,
                    extra: MessageSnapshot | None = None,
                    history: list[MessageSnapshot] | None = None,
                    trigger_reason: str = "mention",
                    trigger_snapshot: MessageSnapshot | None = None,
                    now: float | None = None,
                    required_context_ids: tuple[int, ...] | list[int] = (),
                    pinned_context_ids: tuple[int, ...] | list[int] = ()) -> dict:
        now = self._clock() if now is None else float(now)
        self.prune_expired(now)
        items = list(self._messages.get(chat_id, ()))
        items.extend(history or ())
        if extra is not None:
            items.append(extra)
        # A queued trigger can leave the bounded RAM tail while context/history
        # is collected. Its task retains just this one sanitized snapshot.
        if trigger_snapshot is not None and trigger_snapshot.message_id == trigger_id:
            items.append(trigger_snapshot)
        unique: dict[int, MessageSnapshot] = {}
        for item in items:
            if (item.chat_id == chat_id and item.thread_id == thread_id
                    and self._is_recent(item, now)):
                unique[item.message_id] = item
        ordered = sorted(unique.values(), key=lambda item: (item.timestamp, item.message_id))
        trigger_pos = next((i for i, item in enumerate(ordered) if item.message_id == trigger_id), None)
        if trigger_pos is None:
            return {"messages": [], "trigger_index": 0, "trigger_reason": trigger_reason}
        trigger = ordered[trigger_pos]
        previous_candidates = [item for item in ordered[:trigger_pos] if item.text]
        previous = previous_candidates[-before:] if before > 0 else []
        following = [item for item in ordered[trigger_pos + 1:] if item.text][:after]
        selected = previous + [trigger] + following

        # The replied-to parent may be useful even when it is not in the
        # bounded RAM tail; keep at most one and make room by trimming the
        # oldest pre-trigger item first.
        if trigger.reply_to_id and trigger.reply_to_id not in {m.message_id for m in selected}:
            parent = unique.get(trigger.reply_to_id)
            if parent is not None and parent.text:
                selected.insert(max(0, len(previous) - 1), parent)
        required_ids = set()
        for value in required_context_ids:
            try:
                message_id = int(value)
            except (TypeError, ValueError):
                continue
            parent = unique.get(message_id)
            if parent is None or not parent.text:
                continue
            required_ids.add(message_id)
            if message_id not in {item.message_id for item in selected}:
                selected.append(parent)
        pinned_ids = set()
        for value in pinned_context_ids:
            try:
                message_id = int(value)
            except (TypeError, ValueError):
                continue
            pinned = unique.get(message_id)
            if pinned is None or not pinned.text:
                continue
            pinned_ids.add(message_id)
            if message_id not in {item.message_id for item in selected}:
                selected.append(pinned)
        selected.sort(key=lambda item: (item.timestamp, item.message_id))
        protected_ids = required_ids | pinned_ids | {trigger_id}
        if trigger.reply_to_id is not None:
            protected_ids.add(trigger.reply_to_id)
        while len(selected) > MAX_WINDOW_MESSAGES:
            removable = next(
                (i for i, item in enumerate(selected)
                 if item.message_id not in protected_ids),
                None,
            )
            if removable is None:
                break
            selected.pop(removable)
        selected = [
            MessageSnapshot(m.chat_id, m.message_id, m.sender_id, m.own, m.thread_id,
                            m.reply_to_id, m.timestamp, m.text[:MAX_MESSAGE_CHARS])
            for m in selected
        ]
        # Preserve the trigger and closest surrounding messages under the
        # aggregate cap. Each message is already bounded to 800 characters.
        while sum(len(m.text) for m in selected) > MAX_WINDOW_CHARS:
            removable = next(
                (i for i, item in enumerate(selected)
                 if item.message_id not in protected_ids),
                None,
            )
            if removable is None:
                break
            selected.pop(removable)
        speaker_ids: dict[int, str] = {}
        messages = []
        selected_ids = {m.message_id for m in selected}
        for item in selected:
            if item.own or item.sender_id == own_id:
                speaker = "self"
            else:
                identity = item.sender_id
                if identity not in speaker_ids:
                    speaker_ids[identity] = f"p{len(speaker_ids) + 1}"
                speaker = speaker_ids[identity]
            message = {
                "speaker": speaker,
                "text": item.text,
                "reply_to_previous": bool(item.reply_to_id in selected_ids),
            }
            if (item.message_id in required_ids
                    or (not required_ids and item.message_id == trigger.reply_to_id)):
                message["reply_parent"] = True
            messages.append(message)
        try:
            trigger_index = next(i for i, m in enumerate(selected) if m.message_id == trigger_id)
        except StopIteration:
            return {"messages": [], "trigger_index": 0, "trigger_reason": trigger_reason}
        return {
            "messages": messages,
            "trigger_index": trigger_index,
            "trigger_reason": trigger_reason if trigger_reason in _TRIGGER_REASONS else "mention",
        }
