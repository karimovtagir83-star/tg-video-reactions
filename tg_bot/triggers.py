"""Exact, local trigger matching for the optional ordinary-reply path."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
_URL = re.compile(
    r"(?i)(?:\b(?:https?://|ftp://|www\.)[^\s<>()]+"
    r"|\b(?:[\w-]+\.)+[a-zа-яё]{2,}(?::\d{1,5})?(?:[/?#][^\s<>()]*)?)"
)
_IDENTITY_PREFIX_RE = re.compile(
    r"(?iu)^\s*(?:(?:ну|а|эй|слушай|братан|бро|это)\s*[,!—–-]?\s*)*"
    r"(?:ты|вы)\b\s*[,!—–-]?\s*"
    r"(?:(?:вообще|реально|правда|точно|по-настоящему|как бы)\s+)*"
    r"(?:не\s+)?(?:реальн\w*|настоящ\w*|жив\w*|человек\w*|бот\w*)\b"
)
_WHO_ARE_YOU_RE = re.compile(
    r"(?iu)^\s*(?:(?:ну|а|эй|слушай|братан|бро)\s*[,!—–-]?\s*)*"
    r"(?:(?:ты|вы)\s+)?кто\s+(?:ты|вы)\b|"
    r"^\s*(?:(?:ну|а|эй|слушай|братан|бро)\s*[,!—–-]?\s*)*"
    r"(?:ты|вы)\s+(?:(?:вообще|реально|сам|сама|тут|так)\s+)*кто\b|"
    r"^\s*(?:(?:ну|а|эй|слушай|братан|бро)\s*[,!—–-]?\s*)*"
    r"это\s+(?:ты|вы)\s*$"
)
_AI_TERM = (
    r"(?:нейронк\w*|нейросет\w*|искусственн\w+\s+интеллект\w*|"
    r"языков\w+\s+модел\w*|машинн\w+\s+интеллект\w*|"
    r"электронн\w+\s+мозг\w*|чат[ -]?(?:бот\w*|гпт\w*)|chat[ -]?gpt|"
    r"нейропомощник\w*|нейробот\w*|ассистент\w*|робот\w*|бот\w*|"
    r"иишк\w*|ии|гпт\w*|gpt|ai)"
)
_AI_QUALIFIER = (
    r"(?:туп\w*|глуп\w*|дебильн\w*|бесполезн\w*|бестолков\w*|жалк\w*|"
    r"ебуч\w*|ебан\w*|ёбан\w*|еблив\w*|уебищн\w*|хуев\w*|долбан\w*|"
    r"говн\w*|дерьм\w*|крут\w*|классн\w*|лучш\w*|умн\w*|хорош\w*|"
    r"охуенн\w*|ахуенн\w*|имб\w*|топов\w*|шикарн\w*|офигенн\w*|"
    r"пиздат\w*|заебат\w*|гениальн\w*|мощн\w*|прикольн\w*|смешн\w*|"
    r"красавчик\w*|молодец|супер|великолепн\w*|замечательн\w*|"
    r"превосходн\w*|легендарн\w*|топ|класс|огонь)"
)
_AI_DIRECT_LEAD_RE = re.compile(
    r"(?iu)^\s*(?:(?:ну|эй|слушай|бля|блядь)\s*[,!—–-]?\s*){0,2}"
    r"(?:(?:реально|просто|вообще|очень|крайне|сам\w*|пиздец\s+как|охуенно)\s+){0,3}"
    r"(?:" + _AI_QUALIFIER + r"\s+){1,3}" + _AI_TERM + r"\b"
)
_AI_DIRECT_SECOND_PERSON_RE = re.compile(
    r"(?iu)\b(?:ты|вы)\s+(?:(?:ну|реально|просто|вообще|очень|"
    r"какой|какая|какое|сам\w*|" + _AI_QUALIFIER + r")\s+){0,4}"
    + _AI_TERM + r"\b"
)
_AI_DIRECT_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"(?iu)^\s*(?:ты|вы)\s+" + _AI_TERM + r"\b\s*[,—–:-]?\s*"
    r"(?:(?:ну|реально|просто|вообще|очень|сам\w*)\s+){0,2}"
    r"(?:" + _AI_QUALIFIER + r"\b)"
)
_AI_DIRECT_INSULT_RE = re.compile(
    _AI_DIRECT_LEAD_RE.pattern
)
_AI_DIRECT_PREDICATE_RE = re.compile(
    r"(?iu)^\s*" + _AI_TERM + r"\b\s*[,—–:-]?\s*"
    r"(?:(?:ты|вы)\s+)?(?:(?:реально|просто|вообще|очень|сам\w*)\s+){0,2}"
    r"(?:" + _AI_QUALIFIER + r"\b|(?:ты|вы)\s+(?:имба|огонь|красавчик)\b)"
)
_AI_DIRECT_VOCATIVE_RE = re.compile(
    r"(?iu)^\s*" + _AI_TERM
    + r"\s*[,!—–-](?:\s*(?:ты|вы|ответь|скажи|расскажи|объясни|"
      r"помоги|глянь|смотри|что|кто|как|почему|можешь)\b|\s*$)"
)
_AI_TERM_ONLY_RE = re.compile(
    r"(?iu)^\s*" + _AI_TERM + r"\s*[.!?,…]*\s*$"
)
_AI_IDENTITY_RE = re.compile(
    r"(?iu)^\s*(?:(?:ну|а|эй|слушай|братан|бро)\s*[,!—–-]?\s*)*"
    r"(?:ты|вы)\s+(?:(?:вообще|реально|правда|точно|сам|сама|"
    r"как бы)\s+)*" + _AI_TERM + r"\s*[?!…]*\s*$"
)
_QUOTED_RE = re.compile(
    r'(?s)(?:«[^»]*»|“[^”]*”|„[^“]*“|"[^\"]*"|\x60[^\x60]*\x60)'
)


def is_direct_ai_address(message) -> bool:
    """Match clear second-person or direct-vocative references to the bot."""
    if getattr(message, "fwd_from", None) is not None:
        return False
    text = str(getattr(message, "message", "") or "")
    text = unicodedata.normalize(
        "NFC", _remove_utf16_spans(
            text, _entity_spans(text, getattr(message, "entities", None) or []),
        ),
    )
    text = _QUOTED_RE.sub(" ", _EMAIL.sub(" ", _URL.sub(" ", text)))
    return bool(
        _AI_DIRECT_SECOND_PERSON_RE.search(text)
        or _AI_DIRECT_SECOND_PERSON_PREDICATE_RE.search(text)
        or _AI_DIRECT_INSULT_RE.search(text)
        or _AI_DIRECT_PREDICATE_RE.search(text)
        or _AI_DIRECT_VOCATIVE_RE.search(text)
        or _AI_TERM_ONLY_RE.fullmatch(text)
    )


@dataclass(frozen=True)
class TriggerMatch:
    reasons: tuple[str, ...]


def utf16_slice(text: str, offset: int, length: int) -> str:
    """Slice Telegram entity offsets, which count UTF-16 code units."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    start = max(0, int(offset)) * 2
    end = min(len(raw), (int(offset) + int(length)) * 2)
    if end < start:
        return ""
    return raw[start:end].decode("utf-16-le", errors="ignore")


def _entity_class_name(entity) -> str:
    return type(entity).__name__


def _entity_spans(text: str, entities, *, remove_mentions=True) -> list[tuple[int, int]]:
    spans = []
    for entity in entities or []:
        name = _entity_class_name(entity)
        if "Blockquote" in name:
            start, length = getattr(entity, "offset", 0), getattr(entity, "length", 0)
            if length > 0:
                spans.append((start, start + length))
        elif remove_mentions and (name in {"MessageEntityMention", "MessageEntityMentionName"}
                                  or name in {"MessageEntityTextUrl", "MessageEntityUrl", "MessageEntityEmail", "MessageEntityPhone"}):
            start, length = getattr(entity, "offset", 0), getattr(entity, "length", 0)
            if length > 0:
                spans.append((start, start + length))
    return spans


def _remove_utf16_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    raw = text.encode("utf-16-le", errors="surrogatepass")
    units = len(raw) // 2
    clipped = sorted((max(0, s), min(units, e)) for s, e in spans if e > s)
    merged: list[list[int]] = []
    for start, end in clipped:
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


def _name_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().replace("ё", "е")


def _matches_alias(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    normalized = _name_normalize(text)
    normalized = _EMAIL.sub(" ", normalized)
    normalized = _URL.sub(" ", normalized)
    for alias in aliases:
        candidate = _name_normalize(alias.strip())
        if not candidate or len(candidate) > 80:
            continue
        pattern = re.compile(rf"(?<![\w]){re.escape(candidate)}(?![\w])", re.UNICODE)
        if pattern.search(normalized):
            return True
    return False


def is_direct_identity_question(message, aliases=()) -> bool:
    """Recognize a second-person identity question without requiring punctuation."""
    if getattr(message, "fwd_from", None) is not None:
        return False
    text = str(getattr(message, "message", "") or "")
    text = unicodedata.normalize(
        "NFC", _remove_utf16_spans(text, _entity_spans(text, getattr(message, "entities", None) or [])),
    )
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    # An exact configured owner name may be used as a leading vocative. Remove
    # only that prefix; never treat a named third party as the bot.
    safe_aliases = [alias for alias in aliases or ()
                    if isinstance(alias, str) and alias.strip()]
    for alias in sorted(safe_aliases, key=len, reverse=True):
        candidate = _name_normalize(alias.strip())
        if not candidate:
            continue
        normalized = _name_normalize(text)
        match = re.match(rf"^\s*{re.escape(candidate)}\s*[,!—–:-]\s*", normalized)
        if match:
            text = text[match.end():]
            break
    if (_IDENTITY_PREFIX_RE.search(text) or _WHO_ARE_YOU_RE.search(text)
            or _AI_IDENTITY_RE.search(text)):
        return True
    normalized = _name_normalize(text)
    for alias in safe_aliases:
        candidate = _name_normalize(alias.strip())
        if not candidate or len(candidate) > 80:
            continue
        alias_question = re.compile(
            r"(?iu)^\s*(?:(?:ну|а|эй|слушай|братан|бро)\s*[,!—–-]?\s*)*"
            r"(?:(?:это)\s+)?(?:ты|вы)\s+"
            r"(?:(?:случайно|реально|правда|вообще|сам|сама)\s+)*"
            + re.escape(candidate)
            + r"(?:\s*[,!—–-]?\s*(?:да|что\s+ли|или\s+бот))?\s*[?!…]*\s*$"
        )
        if alias_question.search(normalized):
            return True
    return False


def is_addressed_identity_question(reasons, message, aliases=()) -> bool:
    """Require verified evidence that an identity question addresses the owner."""
    evidence = {"mention_id", "mention_username", "name_alias", "reply", "ai_address"}
    return bool(set(reasons or ()) & evidence) and is_direct_identity_question(message, aliases)


def explicit_triggers(message, own_id: int, own_username: str | None,
                      aliases: tuple[str, ...] | list[str] = ()) -> TriggerMatch:
    """Match only verified mention entities or exact configured name variants."""
    if getattr(message, "fwd_from", None) is not None:
        return TriggerMatch(())
    text = str(getattr(message, "message", "") or "")
    entities = getattr(message, "entities", None) or []
    reasons: list[str] = []
    clean_text = _remove_utf16_spans(text, _entity_spans(text, entities))

    username = (own_username or "").lstrip("@").casefold()
    for entity in entities:
        name = _entity_class_name(entity)
        if name == "MessageEntityMentionName":
            try:
                if int(getattr(entity, "user_id", 0)) == int(own_id):
                    reasons.append("mention_id")
            except (TypeError, ValueError):
                continue
        elif name == "MessageEntityMention" and username:
            mention = utf16_slice(
                text, getattr(entity, "offset", 0), getattr(entity, "length", 0)
            ).lstrip("@").casefold()
            if mention == username:
                reasons.append("mention_username")

    if aliases and _matches_alias(clean_text, aliases):
        reasons.append("name_alias")
    if is_direct_ai_address(message):
        reasons.append("ai_address")
    return TriggerMatch(tuple(dict.fromkeys(reasons)))


async def reply_is_to_owner(event, own_id: int) -> bool:
    """Fetch exactly one parent from this event's own group and verify its sender."""
    message = getattr(event, "message", None)
    if message is None or not getattr(message, "reply_to_msg_id", None):
        return False
    try:
        parent = await event.get_reply_message()
    except Exception:
        return False
    if parent is None:
        return False
    parent_chat = getattr(parent, "chat_id", None)
    event_chat = getattr(event, "chat_id", None)
    if parent_chat is not None and event_chat is not None and int(parent_chat) != int(event_chat):
        return False
    return getattr(parent, "sender_id", None) == own_id


def thread_key(message) -> int:
    """Return a Telegram forum topic ID, or zero for ordinary group history."""
    reply = getattr(message, "reply_to", None)
    top = getattr(reply, "reply_to_top_id", None) if reply is not None else None
    if not top and reply is not None and getattr(reply, "forum_topic", False):
        # A topic-creation/root message can identify its topic through the
        # reply target even when Telegram omits reply_to_top_id.
        top = getattr(reply, "reply_to_msg_id", None)
    top = top or getattr(message, "message_thread_id", None)
    try:
        return int(top or 0)
    except (TypeError, ValueError):
        return 0
