"""Local-only participant summaries and exact question matching.

This module stores concise observations and local identity aliases keyed by
exact visible export labels. Raw messages and aliases remain local; only one
validated participant summary may be selected for a Gemini request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .style import (
    _EMAIL_RE, _HANDLE_RE, _LONG_NUM_RE, _PHONE_RE, _URL_RE,
    _manifest_labels, _private_path, default_owner_manifest,
    iter_export_messages, normalize_label,
)

MAX_PARTICIPANT_FILE_BYTES = 1024 * 1024
MAX_PARTICIPANT_SUMMARY_BYTES = 1024
MAX_PARTICIPANT_LABEL = 120
MAX_PARTICIPANT_ALIASES = 100

TOPIC_LABELS = {
    "work_projects": "работа и проекты",
    "technology": "технологии и программное обеспечение",
    "learning": "учёба",
    "entertainment": "фильмы, игры и музыка",
    "travel": "поездки и путешествия",
}
COMMUNICATION_LABELS = {
    "brief": "сообщения обычно короткие",
    "detailed": "сообщения часто содержат подробности",
    "asks_questions": "в сообщениях часто встречаются вопросы",
    "uses_emojis": "иногда встречаются эмодзи",
    "informal": "манера письма часто разговорная",
    "laughs": "часто встречается письменный смех",
    "uses_casual_profanity": "в сообщениях встречается разговорный мат",
    "emphatic": "часто встречается усиление фраз растяжением слов или повторной пунктуацией",
}

_INFORMAL_STYLE_RE = re.compile(
    r"(?iu)\b(?:бро|братан|дружище|крч|чето|щас|типа|короче|лол|кек|"
    r"ну|ладно)\b"
)
_LAUGHTER_STYLE_RE = re.compile(r"(?iu)\b(?:ахах\w*|хах\w*|лол|кек)\b|[😂🤣😄😆]")
_CASUAL_PROFANITY_STYLE_RE = re.compile(
    r"(?iu)\b(?:нахуй|пиздец|ебать|заеб\w*|охуел\w*)\b"
)
_EMPHATIC_STYLE_RE = re.compile(
    r"(?:[!?]{2,}|…{2,}|\.{3,}|\b[\w]*([а-яё])\1{2,}\w*\b)", re.IGNORECASE,
)

_TOPIC_PATTERNS = {
    "work_projects": re.compile(
        r"(?iu)\b(?:работ\w*|проект\w*|задач\w*|офис\w*|срок\w*|дедлайн\w*|"
        r"work\w*|project\w*|task\w*|deadline\w*)\b"
    ),
    "technology": re.compile(
        r"(?iu)\b(?:код\w*|программ\w*|компьютер\w*|сервер\w*|модел\w*|"
        r"бот\w*|сайт\w*|приложен\w*|техник\w*|software|code|computer|"
        r"server|model|website|app)\b"
    ),
    "learning": re.compile(
        r"(?iu)\b(?:уч[её]б\w*|экзамен\w*|курс\w*|заняти\w*|лекци\w*|"
        r"learning|study|studying|exam|course|lesson)\b"
    ),
    "entertainment": re.compile(
        r"(?iu)\b(?:фильм\w*|сериал\w*|игр\w*|музык\w*|концерт\w*|аниме|"
        r"movie\w*|film\w*|series|game\w*|music|concert)\b"
    ),
    "travel": re.compile(
        r"(?iu)\b(?:поезд\w*|путешеств\w*|отпуск\w*|поездк\w*|билет\w*|"
        r"travel\w*|trip\w*|holiday|vacation|ticket\w*)\b"
    ),
}
_SENSITIVE_TOPIC_RE = re.compile(
    r"(?iu)\b(?:здоров\w*|болезн\w*|диагноз\w*|лечени\w*|лекарств\w*|"
    r"таблетк\w*|врач\w*|медицин\w*|психолог\w*|терап\w*|депресс\w*|"
    r"тревожност\w*|религи\w*|молитв\w*|верующ\w*|атеист\w*|политик\w*|"
    r"выбор\w*|партия\w*|зарплат\w*|деньг\w*|долг\w*|кредит\w*|банк\w*|"
    r"инвестиц\w*|парол\w*|адрес\w*|телефон\w*|паспорт\w*|семейн\w*|"
    r"отношени\w*|развод\w*|секс\w*|беремен\w*|ориентац\w*|национальност\w*|"
    r"ethnic\w*|religio\w*|politic\w*|health\w*|medical\w*|diagnos\w*|"
    r"treatment|medicine|medication|salary|debt|credit|bank|password|address|"
    r"phone|passport|relationship\w*|sexual\w*|pregnan\w*|orientation\w*)\b"
)
_QUOTE_RE = re.compile(r'(?s)(?:«[^»]*»|“[^”]*”|„[^“]*“|"[^"]*"|\x60[^\x60]*\x60)')
_DIRECT_PARTICIPANT_QUERY_RE = re.compile(
    r"(?iu)^\s*(?:(?:а|ну|слушай)\s+)*(?:кто|что|как|какой|какая|какие|кого|кому|"
    r"расскажи|опиши|охарактеризуй|знаешь|думаешь|считаешь|можешь\s+рассказать|"
    r"who|what|how|describe|tell\s+me|know|think|say)\b"
)
_PARTICIPANT_GENERIC_RE = re.compile(
    r"(?iu)\b(?:участник\w*|собеседник\w*|человек\w*|люд\w*|остальн\w*|"
    r"кажд\w*|everyone|participants?|people|members?)\b"
)
_ABOUT_OBJECT_RE = re.compile(
    r"(?iu)\b(?:о|об|обо|про|насч[её]т|about)\s+"
)
_PRONOUN_ABOUT_OBJECTS = frozenset({
    "я", "мы", "ты", "вы", "он", "она", "оно", "они", "меня", "нас",
    "тебя", "вас", "его", "её", "ее", "их", "него", "неё", "нее",
    "нему", "ней", "нём", "нем", "них", "ему", "ей", "им", "ними",
    "это", "этого", "этой", "этот", "эта", "эти", "того", "тому",
    "этим", "this", "that", "it", "him", "her", "them", "you", "he",
    "she", "they", "me", "us", "these", "those",
})
_LABEL_PRIVATE_RE = re.compile(
    r"(?iu)(?:\b(?:https?://|www\.)|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"(?<!\w)@[A-Za-z0-9_]{2,}|(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)|\b\d{5,}\b)"
)
_PRIVATE_CONTENT_RE = re.compile(
    r"(?iu)\b(?:api[_ -]?key|token|password|парол[ья]|секрет|код\s+"
    r"подтверждения)\s*[:=]?\s*\S+"
)


@dataclass(frozen=True)
class ParticipantQuery:
    """A local query match; label stays process-local and is never serialized."""

    should_trigger: bool
    summary: dict | None = None
    label: str | None = None


def participant_summary_path(chat_id: int) -> Path:
    """Return a per-chat private artifact path without exposing the chat ID."""
    digest = hashlib.sha256(str(int(chat_id)).encode("ascii")).hexdigest()[:20]
    return Path("data/style") / f"participants-{digest}.json"


def participant_profiles_path() -> Path:
    """Return the ignored, owner-only shared profile map for allowlisted chats."""
    return Path("data/style") / "participants-global.json"


def _safe_message_for_summary(text: str) -> str:
    return re.sub(r"\s+", " ", _QUOTE_RE.sub(" ", text)).strip()


def _valid_label(label: str) -> bool:
    return (
        isinstance(label, str) and 1 <= len(label.strip()) <= MAX_PARTICIPANT_LABEL
        and not _LABEL_PRIVATE_RE.search(label)
        and not any(ord(char) < 32 or ord(char) == 127 for char in label)
    )


def _normalize_alias(alias: str) -> str:
    return normalize_label(alias).lstrip("@")


def _valid_alias(alias: object) -> bool:
    if not isinstance(alias, str) or not alias.strip() or len(alias.strip()) > 80:
        return False
    value = alias.strip()
    if value.startswith("@"):
        return re.fullmatch(r"@[A-Za-z0-9_]{2,32}", value) is not None
    return _valid_label(value)


def build_participant_summaries(
    export_dir: str | Path,
    chat_id: int,
    manifest: dict | None = None,
) -> tuple[dict, dict]:
    """Build safe observation codes locally from one group's own messages."""
    try:
        chat_id = int(chat_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("participant_chat_id_invalid") from exc
    if not chat_id:
        raise ValueError("participant_chat_id_invalid")
    manifest = manifest or default_owner_manifest()
    owner_labels = _manifest_labels(manifest)
    by_label: dict[str, dict] = {}
    visible_labels: dict[str, str] = {}
    label_author_keys: dict[str, set[str]] = {}
    unkeyed_labels: set[str] = set()
    unresolved_identity: set[str] = set()
    conflicted: set[str] = set()
    scopes: set[str] = set()
    total_messages = 0
    eligible_messages = 0

    for message in iter_export_messages(
        Path(export_dir).expanduser(), manifest, include_unknown_labels=True,
    ):
        scopes.add(message.chat_scope)
        if len(scopes) > 1:
            raise ValueError("participant_export_must_be_one_chat")
        total_messages += 1
        label = message.author_label
        if (message.service or message.ambiguous or message.author_identity is not None
                or not label or not _valid_label(label)):
            continue
        normalized = normalize_label(label)
        if normalized in owner_labels:
            continue
        previous_label = visible_labels.setdefault(normalized, label)
        if previous_label != label:
            conflicted.add(normalized)
            continue
        author_key = getattr(message, "author_key", None)
        if author_key is None:
            # An exact display label is now an explicitly approved identity
            # basis within this single group. It carries a same-name collision
            # risk, recorded in the docs; mixed keyed/unkeyed records remain
            # ambiguous and are excluded below.
            unkeyed_labels.add(normalized)
        elif isinstance(author_key, str) and re.fullmatch(r"\d{1,20}", author_key):
            label_author_keys.setdefault(normalized, set()).add(author_key)
            if len(label_author_keys[normalized]) > 1:
                conflicted.add(normalized)
                continue
        else:
            unresolved_identity.add(normalized)
            continue
        if unkeyed_labels.intersection(label_author_keys):
            conflicted.update(unkeyed_labels.intersection(label_author_keys))
            continue
        text = _safe_message_for_summary(message.text)
        if (not text or _SENSITIVE_TOPIC_RE.search(text)
                or _PRIVATE_CONTENT_RE.search(text)
                or any(pattern.search(text) for pattern in (
                    _EMAIL_RE, _HANDLE_RE, _LONG_NUM_RE, _PHONE_RE, _URL_RE,
                ))):
            continue
        eligible_messages += 1
        stats = by_label.setdefault(normalized, {
            "label": label,
            "message_count": 0,
            "word_total": 0,
            "question_count": 0,
            "emoji_count": 0,
            "informal_count": 0,
            "laughter_count": 0,
            "profanity_count": 0,
            "emphasis_count": 0,
            "topics": Counter(),
        })
        stats["message_count"] += 1
        words = re.findall(r"[\w]+", text, flags=re.UNICODE)
        stats["word_total"] += len(words)
        stats["question_count"] += "?" in text
        stats["emoji_count"] += bool(re.search("[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", text))
        stats["informal_count"] += bool(_INFORMAL_STYLE_RE.search(text))
        stats["laughter_count"] += bool(_LAUGHTER_STYLE_RE.search(text))
        stats["profanity_count"] += bool(_CASUAL_PROFANITY_STYLE_RE.search(text))
        stats["emphasis_count"] += bool(_EMPHATIC_STYLE_RE.search(text))
        for code, pattern in _TOPIC_PATTERNS.items():
            if pattern.search(text):
                stats["topics"][code] += 1
        del text, words, message

    participants = {}
    for normalized, label in sorted(visible_labels.items()):
        author_keys = label_author_keys.get(normalized, set())
        if (normalized in conflicted or normalized in unresolved_identity
                or (normalized in unkeyed_labels and author_keys)
                or (normalized not in unkeyed_labels and len(author_keys) != 1)):
            continue
        stats = by_label.get(normalized)
        topics: list[str] = []
        communication: list[str] = []
        if stats and stats["message_count"] >= 5:
            topics = [
                topic for topic, count in sorted(
                    stats["topics"].items(), key=lambda item: (-item[1], item[0]),
                ) if count >= 2
            ][:3]
            count = stats["message_count"]
            average_words = stats["word_total"] / count
            if average_words <= 16:
                communication.append("brief")
            elif average_words >= 32:
                communication.append("detailed")
            if stats["question_count"] >= 2 and stats["question_count"] / count >= 0.3:
                communication.append("asks_questions")
            if stats["emoji_count"] >= 2 and stats["emoji_count"] / count >= 0.25:
                communication.append("uses_emojis")
            if stats["informal_count"] >= 2 and stats["informal_count"] / count >= 0.25:
                communication.append("informal")
            if stats["laughter_count"] >= 2 and stats["laughter_count"] / count >= 0.25:
                communication.append("laughs")
            if stats["profanity_count"] >= 2 and stats["profanity_count"] / count >= 0.25:
                communication.append("uses_casual_profanity")
            if stats["emphasis_count"] >= 2 and stats["emphasis_count"] / count >= 0.25:
                communication.append("emphatic")
        participants[label] = {"topics": topics, "communication": communication}

    label_only_summaries = sum(
        normalize_label(label) in unkeyed_labels
        for label in participants
    )

    artifact = {
        "schema_version": 1,
        "chat_id": chat_id,
        "participants": participants,
    }
    encoded = json.dumps(artifact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PARTICIPANT_FILE_BYTES:
        raise ValueError("participant_summary_too_large")
    report = {
        "message_blocks": total_messages,
        "eligible_participant_messages": eligible_messages,
        "participant_labels": len(participants),
        "summaries_ready": sum(bool(item["topics"] or item["communication"])
                                for item in participants.values()),
        "ambiguous_labels_excluded": len(conflicted),
        "unresolved_identity_labels_excluded": len(unresolved_identity),
        "label_only_identity_summaries": label_only_summaries,
    }
    return artifact, report


def write_private_participant_summaries(
    artifact: dict,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically store only allowlisted summary codes beneath ignored data/."""
    artifact = _validate_artifact(artifact)
    target = _private_path(Path(output))
    if target.is_symlink():
        raise ValueError("participant_summary_path_invalid")
    payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if len(payload.encode("utf-8")) > MAX_PARTICIPANT_FILE_BYTES:
        raise ValueError("participant_summary_too_large")
    if target.exists() and not overwrite:
        raise FileExistsError("private_profile_exists")
    fd, temp_name = tempfile.mkstemp(prefix=".participants-", dir=target.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp_path, target)
        else:
            os.link(temp_path, target)
            temp_path.unlink()
        os.chmod(target, 0o600)
        return target
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _validate_summary_record(value: object) -> dict | None:
    base_keys = {"topics", "communication"}
    if (not isinstance(value, dict)
            or frozenset(value) not in {frozenset(base_keys),
                                        frozenset(base_keys | {"relationship"})}):
        raise ValueError("participant_summary_invalid")
    topics = value["topics"]
    communication = value["communication"]
    if (not isinstance(topics, list) or not isinstance(communication, list)
            or any(not isinstance(code, str) for code in topics + communication)
            or len(topics) > 3 or len(communication) > 8
            or len(set(topics)) != len(topics) or len(set(communication)) != len(communication)
            or any(code not in TOPIC_LABELS for code in topics)
            or any(code not in COMMUNICATION_LABELS for code in communication)):
        raise ValueError("participant_summary_invalid")
    safe = {"topics": list(topics), "communication": list(communication)}
    if "relationship" in value:
        relationship = value["relationship"]
        if not isinstance(relationship, dict) or set(relationship) != {
                "overview", "you_to_them", "them_to_you", "forms_of_address"}:
            raise ValueError("participant_summary_invalid")
        narrative_keys = ("overview", "you_to_them", "them_to_you")
        narrative = {}
        for key in narrative_keys:
            text = relationship[key]
            if (not isinstance(text, str) or not text.strip()
                    or len(text.encode("utf-8")) > 400
                    or any(ord(char) < 32 or ord(char) == 127 for char in text)
                    or _LABEL_PRIVATE_RE.search(text)
                    or _PRIVATE_CONTENT_RE.search(text)):
                raise ValueError("participant_summary_invalid")
            narrative[key] = text.strip()
        forms = relationship["forms_of_address"]
        if (not isinstance(forms, list) or len(forms) > 5
                or any(not _valid_label(form) for form in forms)
                or len({normalize_label(form) for form in forms}) != len(forms)):
            raise ValueError("participant_summary_invalid")
        safe["relationship"] = {
            **narrative, "forms_of_address": list(forms),
        }
    return safe


def _validate_artifact(value: object, expected_chat_id: int | None = None) -> dict:
    """Accept local aliases and concise observations, never raw chat content."""
    if not isinstance(value, dict):
        raise ValueError("participant_summary_invalid")
    version = value.get("schema_version")
    expected_keys = ({"schema_version", "chat_id", "participants"}
                     if version == 1 else
                     {"schema_version", "chat_id", "participants", "aliases", "scope"})
    if (set(value) != expected_keys or version not in {1, 2}
            or not isinstance(value.get("participants"), dict)
            or len(value["participants"]) > 5_000):
        raise ValueError("participant_summary_invalid")
    try:
        chat_id = int(value["chat_id"])
        scope = value.get("scope", "group")
        is_global = version == 2 and scope == "allowlisted_groups" and chat_id == 0
        if (version == 2 and not (is_global or (scope == "group" and chat_id != 0))
                or version == 1 and not chat_id
                or expected_chat_id is not None and not is_global
                and chat_id != int(expected_chat_id)):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("participant_summary_invalid") from exc
    participants = {}
    normalized_seen = set()
    for label, record in value["participants"].items():
        if not _valid_label(label):
            raise ValueError("participant_summary_invalid")
        normalized = normalize_label(label)
        if normalized in normalized_seen:
            raise ValueError("participant_summary_invalid")
        normalized_seen.add(normalized)
        participants[label] = _validate_summary_record(record)
    if version == 1:
        return {"schema_version": 1, "chat_id": chat_id,
                "participants": participants}
    aliases = value["aliases"]
    if not isinstance(aliases, dict) or len(aliases) > MAX_PARTICIPANT_ALIASES:
        raise ValueError("participant_summary_invalid")
    safe_aliases = {}
    normalized_aliases = set()
    for alias, label in aliases.items():
        normalized = _normalize_alias(alias) if _valid_alias(alias) else ""
        if (not normalized or not isinstance(label, str) or label not in participants
                or normalized in normalized_aliases):
            raise ValueError("participant_summary_invalid")
        normalized_aliases.add(normalized)
        safe_aliases[alias.strip()] = label
    return {"schema_version": 2, "chat_id": chat_id, "scope": scope,
            "participants": participants, "aliases": safe_aliases}


def load_private_participant_summaries(
    path: str | Path,
    expected_chat_id: int,
) -> dict:
    """Read an owner-only per-group label map; never serialize it to Gemini."""
    root = (_root_path() / "data").resolve()
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = _root_path() / source
    if source.is_symlink():
        raise ValueError("participant_summary_unavailable")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("participant_summary_unavailable") from exc
    mode = resolved.stat().st_mode
    if (not stat.S_ISREG(mode) or mode & 0o077
            or resolved.stat().st_size > MAX_PARTICIPANT_FILE_BYTES
            or resolved.parent.stat().st_mode & 0o077
            or root.stat().st_mode & 0o077):
        raise ValueError("participant_summary_unavailable")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
        return _validate_artifact(artifact, expected_chat_id)
    except Exception as exc:
        raise ValueError("participant_summary_unavailable") from exc


def _root_path() -> Path:
    from . import style
    return style.ROOT


def participant_summary_for_api(record: dict | None) -> str | None:
    """Render one selected record from fixed vocabulary, with a strict byte cap."""
    if record is None:
        return None
    safe = _validate_summary_record(record)
    if safe is None or not (safe["topics"] or safe["communication"]
                            or safe.get("relationship")):
        return None
    fragments = []
    if safe["topics"]:
        topics = ", ".join(TOPIC_LABELS[code] for code in safe["topics"])
        fragments.append(f"В сообщениях этого участника часто встречаются темы: {topics}.")
    relationship = safe.get("relationship")
    if safe["communication"] and not relationship:
        fragments.extend(COMMUNICATION_LABELS[code].capitalize() + "."
                         for code in safe["communication"])
    if relationship:
        fragments.extend((
            f"По переписке: {relationship['overview']}",
            f"Ты с ним: {relationship['you_to_them']}",
            f"Он отвечает: {relationship['them_to_you']}",
        ))
        forms = relationship["forms_of_address"]
        if forms:
            fragments.append(
                "Твои привычные дружеские обращения к нему: "
                + ", ".join(forms) + "."
            )
    result = " ".join(fragments)
    if len(result.encode("utf-8")) > MAX_PARTICIPANT_SUMMARY_BYTES:
        raise ValueError("participant_summary_invalid")
    return result


def participant_label_for_sender(sender, artifact: dict | None) -> str | None:
    """Resolve one sender to a local exact label, skipping ambiguous aliases."""
    if sender is None or not isinstance(artifact, dict):
        return None
    aliases = artifact.get("aliases", {})
    participants = artifact.get("participants", {})
    lookup = {_normalize_alias(alias): label for alias, label in aliases.items()}
    lookup.update({normalize_label(label): label for label in participants})
    candidates = []
    ambiguous = set(artifact.get("_ambiguous_aliases", ()))
    username = getattr(sender, "username", None)
    if isinstance(username, str) and username:
        candidates.append("@" + username.lstrip("@"))
    display = " ".join(
        value for value in (getattr(sender, "first_name", None),
                            getattr(sender, "last_name", None))
        if isinstance(value, str) and value.strip()
    )
    if display:
        candidates.append(display)
    matched = set()
    for candidate in candidates:
        normalized = _normalize_alias(candidate)
        if normalized in ambiguous:
            return None
        label = lookup.get(normalized)
        if label in participants:
            matched.add(label)
    return next(iter(matched)) if len(matched) == 1 else None


def participant_label_for_reference(reference: str, artifact: dict | None) -> str | None:
    """Resolve an exact participant alias or common Russian case form locally."""
    matched = _participant_reference_targets(reference, artifact)
    return next(iter(matched)) if matched is not None and len(matched) == 1 else None


def participant_reference_is_ambiguous(reference: str, artifact: dict | None) -> bool:
    """Keep known conflicting private aliases out of display-name fallback."""
    matched = _participant_reference_targets(reference, artifact)
    return matched is None or len(matched) > 1


def participant_reference_is_known(reference: str, artifact: dict | None) -> bool:
    """Return whether a reference matches at least one local participant alias."""
    matched = _participant_reference_targets(reference, artifact)
    return matched is None or bool(matched)


def _participant_reference_targets(reference: str, artifact: dict | None) -> set[str] | None:
    if not isinstance(reference, str) or not isinstance(artifact, dict):
        return set()
    normalized_reference = _normalize_alias(reference.strip().strip(" ,.!?;:—–-"))
    if not normalized_reference:
        return set()

    aliases = artifact.get("aliases", {})
    participants = artifact.get("participants", {})
    duplicate_aliases = set(artifact.get("_ambiguous_aliases", ()))
    if any(normalized_reference in _participant_reference_forms(alias)
           for alias in duplicate_aliases):
        return None
    targets: dict[str, set[str]] = {}

    for label in participants:
        if (isinstance(label, str)
                and _normalize_alias(label) not in duplicate_aliases):
            for value in _participant_reference_forms(label):
                targets.setdefault(value, set()).add(label)
    for alias, label in aliases.items():
        if not isinstance(alias, str) or not isinstance(label, str):
            continue
        if _normalize_alias(alias) in duplicate_aliases:
            continue
        for value in _participant_reference_forms(alias):
            targets.setdefault(value, set()).add(label)

    return targets.get(normalized_reference, set())


def _participant_reference_forms(alias: str) -> set[str]:
    normalized = _normalize_alias(alias)
    if not normalized:
        return set()
    result = {normalized}
    if alias.strip().startswith("@"):
        return result
    parts = normalized.split()
    prefix = " ".join(parts[:-1])
    final = parts[-1]
    # Short names still have productive Russian case forms (for example,
    # names ending in -а/-я/-ь). Keep the conservative cutoff for other
    # short tokens to avoid turning common words into participant aliases.
    if len(final) < 3 or (len(final) < 4 and not final.endswith(("а", "я", "ь"))):
        return result
    if final.endswith("а"):
        stem = final[:-1]
        inflections = ("ы", "е", "у", "ой", "ою")
    elif final.endswith("я"):
        stem = final[:-1]
        inflections = ("и", "е", "ю", "ей", "ею")
    elif final.endswith("ь"):
        stem = final[:-1]
        inflections = ("я", "ю", "и", "е", "ем", "ём")
    else:
        stem = final
        inflections = ("а", "у", "е", "ом", "ем", "ой", "ей", "и", "ы")
    for ending in inflections:
        result.add(f"{prefix} {stem}{ending}".strip())
    return result


def _participant_question_forms(alias: str) -> set[str]:
    """Build searchable forms, inflecting Russian names but keeping handles exact."""
    if alias.strip().startswith("@"):
        return {alias.strip()}
    normalized = _normalize_alias(alias)
    if not normalized:
        return set()
    final = normalized.split()[-1]
    if not re.fullmatch(r"(?iu)[а-яё]+", final):
        return {normalized}
    return _participant_reference_forms(alias)


def _participant_question_alias_index(artifact: dict | None):
    """Map searchable alias forms to labels while retaining conflicting forms."""
    if not isinstance(artifact, dict):
        return {}, set()
    participants = artifact.get("participants", {})
    aliases = artifact.get("aliases", {})
    duplicate_aliases = set(artifact.get("_ambiguous_aliases", ()))
    form_targets: dict[str, set[str]] = {}
    form_text: dict[str, str] = {}
    ambiguous_forms: set[str] = set()

    def add(alias, label):
        if not isinstance(alias, str) or not isinstance(label, str):
            return
        normalized = _normalize_alias(alias)
        is_ambiguous = normalized in duplicate_aliases
        for form in _participant_question_forms(alias):
            key = " ".join(form.casefold().split())
            form_text.setdefault(key, form)
            form_targets.setdefault(key, set()).add(label)
            if is_ambiguous:
                ambiguous_forms.add(key)

    if isinstance(participants, dict):
        for label in participants:
            add(label, label)
    if isinstance(aliases, dict):
        for alias, label in aliases.items():
            add(alias, label)
    ambiguous_forms.update(
        key for key, targets in form_targets.items() if len(targets) > 1
    )
    return {key: (form_text[key], targets)
            for key, targets in form_targets.items()}, ambiguous_forms


def participant_reference_matches_sender(
    reference: str, sender, artifact: dict | None,
) -> bool:
    """Match a local command reference to one sender's profile or display name."""
    if participant_reference_is_ambiguous(reference, artifact):
        return False
    resolved_label = participant_label_for_reference(reference, artifact)
    if resolved_label is not None:
        return participant_label_for_sender(sender, artifact) == resolved_label

    normalized_reference = _normalize_alias(reference.strip().strip(" ,.!?;:—–-"))
    if not normalized_reference:
        return False
    candidates = []
    username = getattr(sender, "username", None)
    if isinstance(username, str) and username:
        candidates.append("@" + username.lstrip("@"))
    display = " ".join(
        value for value in (getattr(sender, "first_name", None),
                            getattr(sender, "last_name", None))
        if isinstance(value, str) and value.strip()
    )
    if display:
        candidates.append(display)
    first_name = getattr(sender, "first_name", None)
    if isinstance(first_name, str) and first_name.strip():
        candidates.append(first_name.strip())
    return any(
        normalized_reference in _participant_reference_forms(candidate)
        for candidate in candidates
    )


def participant_summary_for_sender(sender, artifact: dict | None) -> dict | None:
    """Select one local relationship profile by the sender's public username."""
    label = participant_label_for_sender(sender, artifact)
    if label is None:
        return None
    return _validate_summary_record(artifact["participants"][label])


def match_participant_question(
    message,
    artifact: dict | None,
) -> ParticipantQuery | None:
    """Recognize exact local display names, inflections, and usernames."""
    if message is None or getattr(message, "fwd_from", None) is not None:
        return None
    raw_text = unicodedata.normalize("NFC", str(getattr(message, "message", "") or ""))
    # Keep the literal handle for exact local alias matching. URLs, quotes, and
    # unrelated Telegram entity text still cannot become participant names.
    text = _URL_RE.sub(" ", _EMAIL_RE.sub(" ", _QUOTE_RE.sub(" ", raw_text)))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    participants = (artifact or {}).get("participants", {})
    alias_index, ambiguous_forms = _participant_question_alias_index(artifact)
    # A leading name followed by punctuation is normally a vocative (who is
    # asking the bot), not the subject of the question. Strip that form before
    # resolving the named participant.
    for form, _targets in sorted(
            alias_index.values(), key=lambda item: len(item[0]), reverse=True):
        if len(form) < 2 or form.startswith("@"):
            continue
        prefix = re.match(rf"(?iu)^\s*{re.escape(form)}(?=\s*[,!—-])", text)
        if prefix is not None:
            text = re.sub(r"^\s*[^,!—-]*[,!—-]\s*", "", text[prefix.end():], count=1)
            break

    matches: list[tuple[int, int, str]] = []
    for form_key, (form, _targets) in alias_index.items():
        if len(form) < 2:
            continue
        boundary = re.compile(rf"(?iu)(?<![\w]){re.escape(form)}(?![\w])")
        matches.extend((match.start(), match.end(), form_key)
                       for match in boundary.finditer(text))
    # Resolve a short label that is a prefix of a longer exact label at the
    # same position, while preserving distinct names elsewhere in the query.
    selected_matches: list[tuple[int, int, str]] = []
    for start, end, label in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < chosen_end and end > chosen_start
               for chosen_start, chosen_end, _chosen_label in selected_matches):
            continue
        selected_matches.append((start, end, label))
    matched_forms = list(dict.fromkeys(
        form_key for _start, _end, form_key in selected_matches
    ))
    matched = list(dict.fromkeys(
        label for form_key in matched_forms
        for label in alias_index[form_key][1]
    ))
    matched_ambiguous = any(form_key in ambiguous_forms for form_key in matched_forms)
    generic_question = bool(_PARTICIPANT_GENERIC_RE.search(text))
    about_match = _ABOUT_OBJECT_RE.search(text)
    name_question = about_match is not None
    question_like = "?" in text or bool(_DIRECT_PARTICIPANT_QUERY_RE.search(text))
    if not question_like:
        return None

    if about_match is not None:
        # The explicit object of "about/o/pro" outranks other names in the
        # sentence. Resolve the longest known alias at the start of the object
        # so trailing verbs do not get swallowed into a greedy name phrase.
        object_text = text[about_match.end():]
        first_token = re.match(r"(?iu)^\s*([\w]+)", object_text)
        first_target_token = normalize_label(first_token.group(1)) if first_token else ""
        if first_target_token in _PRONOUN_ABOUT_OBJECTS:
            # A pronoun is not a local participant identity. If another exact
            # participant is named, keep the trigger but ask instead of using
            # that other participant's summary; without one, leave the query
            # outside the participant trigger path.
            return (
                ParticipantQuery(True, None) if matched or generic_question
                else None
            )
        target_matches = []
        for form_key, (form, _targets) in alias_index.items():
            target_match = re.match(
                rf"(?iu)^\s*{re.escape(form)}(?![\w])", object_text,
            )
            if target_match is not None:
                target_matches.append((target_match.end(), form_key))
        if not target_matches:
            return ParticipantQuery(True, None)
        longest_target = max(end for end, _form_key in target_matches)
        target_forms = list(dict.fromkeys(
            form_key for end, form_key in target_matches if end == longest_target
        ))
        target_labels = list(dict.fromkeys(
            label for form_key in target_forms
            for label in alias_index[form_key][1]
        ))
        if (len(target_labels) == 1
                and not any(form_key in ambiguous_forms for form_key in target_forms)
                and not matched_ambiguous
                and len(matched) == 1
                and matched[0] == target_labels[0]
                and isinstance(participants, dict)
                and target_labels[0] in participants):
            target_label = target_labels[0]
            return ParticipantQuery(
                True, _validate_summary_record(participants[target_label]),
                target_label,
            )
        # Another exact name in the same query makes the target ambiguous even
        # when one name appears in an explicit object phrase.
        return ParticipantQuery(True, None)

    if matched and len(matched) == 1:
        label = matched[0]
        if (matched_ambiguous or artifact is None
                or not isinstance(participants, dict) or label not in participants):
            return ParticipantQuery(True, None)
        # A stripped vocative is never promoted back into the target slot.
        return ParticipantQuery(
            True, _validate_summary_record(participants[label]),
            label,
        )

    if matched or matched_ambiguous or generic_question or name_question:
        return ParticipantQuery(True, None)
    # Pronoun-only questions such as "How do I run it?" and "What do you
    # think of him?" do not establish which group member is being discussed.
    return None


def combine_participant_artifacts(global_artifact: dict | None,
                                  group_artifact: dict | None) -> dict | None:
    """Combine shared and group-local records, with local aliases taking priority."""
    if global_artifact is None:
        return group_artifact
    if group_artifact is None:
        return global_artifact
    participants = dict(global_artifact.get("participants", {}))
    participants.update(group_artifact.get("participants", {}))
    aliases = {}
    ambiguous = set(global_artifact.get("_ambiguous_aliases", ()))
    ambiguous.update(group_artifact.get("_ambiguous_aliases", ()))
    for source in (global_artifact, group_artifact):
        for alias, label in source.get("aliases", {}).items():
            normalized = _normalize_alias(alias)
            existing_key = next((key for key in aliases
                                 if _normalize_alias(key) == normalized), None)
            if source is group_artifact:
                if existing_key is not None:
                    aliases.pop(existing_key, None)
                ambiguous.discard(normalized)
                aliases[alias] = label
            elif existing_key is None and normalized not in ambiguous:
                aliases[alias] = label
    return {
        "participants": participants,
        "aliases": aliases,
        "_ambiguous_aliases": ambiguous,
    }
