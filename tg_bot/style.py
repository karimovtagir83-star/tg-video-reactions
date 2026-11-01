"""Local-only parser and aggregate style-profile builder for Telegram HTML exports.

This module deliberately has no network or model dependency. It never stores
message text: only bounded aggregate statistics and an opaque source digest are
written to the private profile.
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from .voice import compile_vocative_elongations, load_voice_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OWNERS_PATH = ROOT / "style" / "owners.json"
MAX_EXPORT_FILE = 128 * 1024 * 1024
MAX_EXPORT_MESSAGE = 32_000
MAX_AUTHOR_LABEL = 512
MAX_PROFILE_TEXT = 4_000

_FILE_RE = re.compile(r"^messages(\d*)\.html$", re.IGNORECASE)
_URL_RE = re.compile(
    r"(?i)(?:\b(?:https?://|ftp://|www\.)[^\s<>()]+"
    r"|\b(?:[\w-]+\.)+[a-zа-яё]{2,}(?::\d{1,5})?(?:[/?#][^\s<>()]*)?)"
)
_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
_HANDLE_RE = re.compile(r"(?<![\w])@[A-Za-z0-9_]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_LONG_NUM_RE = re.compile(r"(?<!\w)\d{5,}(?!\w)")
_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|token|password|пароль|код|секрет)\s*[:=]?\s*\S+"
)
_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+|$)")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", re.UNICODE
)

# The profile only reports frequencies for this small generic vocabulary.
# It cannot turn names, topics, or one-off expressions into prompt content.
_SAFE_STYLE_WORDS = {
    "ну", "да", "нет", "вот", "это", "как", "что", "просто", "очень", "э",
    "кстати", "короче", "ладно", "ок", "okay", "well", "so", "yeah",
    "no", "yes", "just", "really", "btw",
}
_SAFE_FREQUENT_WORDS = _SAFE_STYLE_WORDS | {
    "наверное", "вообще", "сейчас", "тоже", "спасибо", "пожалуйста",
    "конечно", "кажется", "думаю", "реально", "вроде", "зато", "поэтому",
    "действительно", "главное", "если", "тогда", "потом", "может", "возможно",
    "точно", "супер", "классно", "нормально", "серьезно", "серьезный",
    "черт", "блин", "капец", "офигеть", "крч", "чето", "братан", "бро",
    "типа", "щас",
}
_SAFE_CASUAL_WORDS = {
    "ну", "короче", "типа", "ок", "ладно", "ахах", "лол", "классно", "э",
    "реально", "блин", "черт", "капец", "офигеть", "крч", "чето",
    "братан", "бро", "щас",
}
_SAFE_MILD_INTERJECTION_WORDS = {"блин", "черт", "капец"}
_SAFE_ABBREVIATIONS = {
    "имхо", "кмк", "лол", "лолкек", "ахах", "спс", "пж", "плиз",
    "ок", "окей", "btw", "lol", "thx", "pls", "idk", "omg",
}
_PUNCTUATION = ".,!?…:;()-—–"


def normalize_label(value: str) -> str:
    """Normalize exact author labels without doing substring or fuzzy matching."""
    return unicodedata.normalize("NFC", " ".join(str(value).split())).casefold()


def default_owner_manifest() -> dict:
    try:
        value = json.loads(DEFAULT_OWNERS_PATH.read_text(encoding="utf-8"))
        _manifest_labels(value)
        return value
    except (OSError, ValueError, TypeError):
        return {
            "schema_version": 1,
            "matching": "exact-from_name-nfc-casefold",
            "owners": {"current": [], "deleted_account": [], "named_account": []},
            "unresolved_identities": [],
        }


def _voice_style_sets(config: dict | None = None) -> dict[str, set[str]]:
    lexicon = (config or load_voice_config()).get("lexicon", {})
    return {
        "frequent": _SAFE_FREQUENT_WORDS | set(lexicon.get("frequent_words", [])),
        "priority": set(lexicon.get("priority_words", [])),
        "casual": _SAFE_CASUAL_WORDS | set(lexicon.get("casual_words", [])),
        "mild_expletives": set(lexicon.get("mild_expletives", [])),
        "fillers": set(lexicon.get("fillers", [])),
        "canonical_forms": set(lexicon.get("familiar_vocatives", [])),
    }


def _normalize_style_word(word: str, settings: dict[str, set[str]], patterns) -> str:
    folded = word.casefold().replace("ё", "е")
    for pattern in patterns:
        if pattern.fullmatch(folded):
            if settings["canonical_forms"]:
                return sorted(settings["canonical_forms"])[0].casefold().replace("ё", "е")
    for filler in settings["fillers"]:
        if re.fullmatch(r"(?:" + re.escape(filler) + r")+", folded,
                        flags=re.IGNORECASE):
            return filler.casefold().replace("ё", "е")
    return folded


def _manifest_labels(manifest: dict) -> dict[str, str]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("owners_manifest_invalid")
    owners = manifest.get("owners")
    if not isinstance(owners, dict):
        raise ValueError("owners_manifest_invalid")
    normalized: dict[str, str] = {}
    for identity, labels in owners.items():
        if not isinstance(identity, str) or not isinstance(labels, list):
            raise ValueError("owners_manifest_invalid")
        for label in labels:
            if not isinstance(label, str) or not label.strip() or len(label) > 120:
                raise ValueError("owners_manifest_invalid")
            key = normalize_label(label)
            if key in normalized:
                raise ValueError("owners_manifest_duplicate_label")
            normalized[key] = identity
    return normalized


def _safe_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _LONG_NUM_RE.sub(" ", text)
    text = _TOKEN_RE.sub(" ", text)
    text = _SECRET_RE.sub(" ", text)
    text = re.sub(r"\b[\w-]{28,}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_PROFILE_TEXT]


@dataclass
class _Frame:
    tag: str
    classes: frozenset[str] = frozenset()
    is_from_name: bool = False
    is_text: bool = False
    skip_text: bool = False


@dataclass
class _Message:
    classes: frozenset[str]
    msg_id: str
    frames: list[_Frame] = field(default_factory=list)
    author_parts: list[str] = field(default_factory=list)
    author_keys: set[str] = field(default_factory=set, repr=False)
    text_parts: list[str] = field(default_factory=list)
    from_name_count: int = 0
    clipped: bool = False
    text_length: int = 0
    author_length: int = 0


@dataclass(frozen=True)
class ExportMessage:
    """Ephemeral parsed message; callers must not persist or print its text."""

    chat_scope: str
    message_id: str
    author_identity: str | None
    author_label: str | None
    text: str
    joined: bool
    service: bool
    ambiguous: bool = False
    # Telegram sender identifiers are transient builder input only.  The
    # participant artifact and API serializer deliberately omit this field.
    author_key: str | None = field(default=None, repr=False)


class _MessageHTMLParser(html.parser.HTMLParser):
    _SKIP_CLASSES = frozenset({
        "quote", "blockquote", "forwarded", "forwarded_body", "media_wrap",
        "media", "photo", "video", "audio", "voice", "sticker", "file",
        "attachment", "signature", "reactions", "reaction", "service",
    })

    def __init__(self, chat_scope: str, owner_map: dict[str, str], owner_labels: dict[str, str],
                 *, include_unknown_labels: bool = False):
        super().__init__(convert_charrefs=True)
        self.chat_scope = chat_scope
        self.owner_map = owner_map
        self.owner_labels = owner_labels
        self.include_unknown_labels = include_unknown_labels
        self.current: _Message | None = None
        self.all_messages: list[ExportMessage] = []
        self.last_owner_label: str | None = None
        self.last_author_key: str | None = None
        self._message_depth = 0

    @staticmethod
    def _author_key(attrs) -> str | None:
        """Extract a strict Telegram user id for local collision checks only."""
        values = dict(attrs)
        for name in ("data-user-id", "data-peer-id", "data-from-id"):
            value = values.get(name)
            if isinstance(value, str) and re.fullmatch(r"\d{1,20}", value):
                return value
        href = values.get("href")
        if isinstance(href, str):
            match = re.fullmatch(r"(?i)tg://user\?id=(\d{1,20})", href)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _attrs(attrs) -> tuple[str, frozenset[str]]:
        values = dict(attrs)
        return values.get("id", ""), frozenset((values.get("class") or "").split())

    def handle_starttag(self, tag, attrs):
        msg_id, classes = self._attrs(attrs)
        if self.current is None and tag == "div" and "message" in classes:
            self.current = _Message(classes=classes, msg_id=msg_id)
            if self.include_unknown_labels:
                key = self._author_key(attrs)
                if key is not None:
                    self.current.author_keys.add(key)
            self._message_depth = 1
            return
        if self.current is None:
            return
        self._message_depth += 1
        parent_skip = any(frame.skip_text for frame in self.current.frames)
        is_from = tag == "div" and "from_name" in classes
        is_text = tag == "div" and "text" in classes
        skip = parent_skip or bool(classes & self._SKIP_CLASSES) or tag in {"script", "style"}
        if tag == "a":
            # Link captions can contain names, URLs, and unrelated private text.
            skip = True
        frame = _Frame(tag, classes, is_from, is_text, skip)
        self.current.frames.append(frame)
        if self.include_unknown_labels and (
            is_from or any(parent.is_from_name for parent in self.current.frames[:-1])
        ):
            key = self._author_key(attrs)
            if key is not None:
                self.current.author_keys.add(key)
        if is_from:
            self.current.from_name_count += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.current is None:
            return
        if not self.current.frames:
            self._finish_message()
            return
        # Telegram exports are well formed, but recover from a mismatched close
        # by unwinding to the nearest matching tag.
        frame_index = next(
            (i for i in range(len(self.current.frames) - 1, -1, -1)
             if self.current.frames[i].tag == tag), None
        )
        if frame_index is None:
            return
        unwound = len(self.current.frames) - frame_index
        del self.current.frames[frame_index:]
        self._message_depth -= unwound
        if tag == "div" and self._message_depth <= 0:
            self._finish_message()

    def handle_data(self, data):
        if self.current is None or not data:
            return
        # Telegram adds forwarding timestamps inside author labels as
        # span.date.details. They are metadata, not part of the visible name.
        # Keep every other suffix in the exact label so an arbitrary nested
        # status/name fragment cannot turn an unknown sender into the owner.
        author_metadata = any(
            frame.tag == "span" and {"date", "details"} <= frame.classes
            for frame in self.current.frames
        )
        if (any(frame.is_from_name for frame in self.current.frames)
                and not author_metadata):
            remaining_author = MAX_AUTHOR_LABEL - self.current.author_length
            if remaining_author > 0:
                piece = data[:remaining_author]
                self.current.author_parts.append(piece)
                self.current.author_length += len(piece)
            if len(data) > remaining_author:
                self.current.clipped = True
        if (any(frame.is_text for frame in self.current.frames)
                and not any(frame.skip_text for frame in self.current.frames)):
            remaining = MAX_EXPORT_MESSAGE - self.current.text_length
            if remaining > 0:
                piece = data[:remaining]
                self.current.text_parts.append(piece)
                self.current.text_length += len(piece)
            if len(data) > remaining:
                self.current.clipped = True

    def _finish_message(self):
        item = self.current
        if item is None:
            return
        self.current = None
        is_service = "service" in item.classes
        is_joined = "joined" in item.classes
        raw_label = " ".join("".join(item.author_parts).split()) or None
        label = None
        ambiguous = item.from_name_count > 1 or item.clipped
        identity = None
        author_key = None
        if is_service or ambiguous:
            self.last_owner_label = None
            self.last_author_key = None
        elif raw_label is not None:
            normalized = normalize_label(raw_label)
            identity = self.owner_map.get(normalized)
            if identity is not None:
                label = self.owner_labels.get(normalized, raw_label)
                self.last_owner_label = label
                self.last_author_key = (
                    next(iter(item.author_keys))
                    if self.include_unknown_labels and len(item.author_keys) == 1
                    else None
                )
            elif self.include_unknown_labels:
                # This private local-only mode is used to build summaries keyed
                # by exact visible export labels; ordinary profile scans omit them.
                label = raw_label
                self.last_owner_label = raw_label
                self.last_author_key = (
                    next(iter(item.author_keys)) if len(item.author_keys) == 1 else None
                )
            else:
                self.last_owner_label = None
                self.last_author_key = None
        elif is_joined and self.last_owner_label is not None:
            # Joined blocks inherit ownership for corpus selection, but are not
            # counted as exact author-label evidence in scan summaries.
            identity = self.owner_map.get(normalize_label(self.last_owner_label))
            if self.include_unknown_labels and identity is None:
                label = self.last_owner_label
            if self.include_unknown_labels:
                if len(item.author_keys) == 1:
                    joined_key = next(iter(item.author_keys))
                    if self.last_author_key is None or joined_key == self.last_author_key:
                        self.last_author_key = joined_key
                    else:
                        self.last_author_key = None
                elif len(item.author_keys) > 1:
                    self.last_author_key = None
                author_key = self.last_author_key
            else:
                author_key = None
        else:
            self.last_owner_label = None
            self.last_author_key = None
        if raw_label is not None:
            author_key = (
                next(iter(item.author_keys))
                if self.include_unknown_labels and len(item.author_keys) == 1
                else None
            )
        text = "".join(item.text_parts)
        self.all_messages.append(ExportMessage(
            self.chat_scope, item.msg_id, identity, label,
            _safe_text(text), is_joined, is_service, ambiguous, author_key,
        ))

    def close(self):
        super().close()
        if self.current is not None:
            # An unclosed message subtree or trailing partial block must not
            # be treated as a complete author/text sample.
            if self.current.frames or self._message_depth > 1:
                self.current.clipped = True
            self._finish_message()

    def drain_messages(self):
        messages, self.all_messages = self.all_messages, []
        return messages


def _file_sort_key(path: Path, root: Path):
    match = _FILE_RE.match(path.name)
    assert match
    suffix = int(match.group(1) or "1")
    return (path.parent.relative_to(root).as_posix().casefold(), suffix, path.name.casefold())


def _export_files(export_dir: Path) -> list[Path]:
    root = export_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("export_dir_invalid")
    found = []
    for path in root.rglob("messages*.html"):
        if path.is_symlink():
            continue
        match = _FILE_RE.match(path.name)
        if not match or not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if path.stat().st_size > MAX_EXPORT_FILE:
            raise ValueError("export_file_too_large")
        found.append(path)
    return sorted(found, key=lambda p: _file_sort_key(p, root))


def iter_export_messages(export_dir: Path, manifest: dict | None = None,
                         *, dedupe: bool = True,
                         include_unknown_labels: bool = False) -> Iterator[ExportMessage]:
    """Yield parsed fields; unknown labels are exposed only for private local analysis."""
    manifest = manifest or default_owner_manifest()
    owner_map = _manifest_labels(manifest)
    owner_labels = {
        normalize_label(label): label
        for labels in manifest["owners"].values()
        for label in labels
    }
    root = export_dir.resolve(strict=True)
    files = _export_files(root)
    seen_ids: set[tuple[str, str]] = set()
    for path in files:
        scope = path.parent.relative_to(root).as_posix() or "root"
        parser = _MessageHTMLParser(
            scope, owner_map, owner_labels,
            include_unknown_labels=include_unknown_labels,
        )
        def unique_messages(messages):
            for msg in messages:
                dedupe_key = (scope, msg.message_id)
                if dedupe and msg.message_id and dedupe_key in seen_ids:
                    continue
                if msg.message_id:
                    seen_ids.add(dedupe_key)
                yield msg
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            while chunk := stream.read(64 * 1024):
                parser.feed(chunk)
                yield from unique_messages(parser.drain_messages())
        parser.close()
        yield from unique_messages(parser.drain_messages())


def _manifest_public_counts(manifest: dict) -> dict[str, int]:
    return {label: 0 for labels in manifest["owners"].values() for label in labels}


def _unresolved_identity_count(manifest: dict) -> int:
    values = manifest.get("unresolved_identities", [])
    return len(values) if isinstance(values, (list, tuple, set)) else 1


def _count_label(counts: dict[str, int], label: str | None) -> str | None:
    if label is None:
        return None
    normalized = normalize_label(label)
    return next((candidate for candidate in counts if normalize_label(candidate) == normalized), None)


def scan_export(export_dir: Path, manifest: dict | None = None) -> dict:
    manifest = manifest or default_owner_manifest()
    _manifest_labels(manifest)
    counts = _manifest_public_counts(manifest)
    total = owned = unknown = eligible = excluded = ambiguous = duplicate_blocks = 0
    seen_ids: set[tuple[str, str]] = set()
    for msg in iter_export_messages(export_dir, manifest, dedupe=False):
        total += 1
        scoped_id = (msg.chat_scope, msg.message_id)
        if msg.message_id and scoped_id in seen_ids:
            duplicate_blocks += 1
        elif msg.message_id:
            seen_ids.add(scoped_id)
        count_label = _count_label(counts, msg.author_label)
        if count_label is not None:
            counts[count_label] += 1
        if msg.ambiguous:
            ambiguous += 1
        if msg.author_identity is None:
            unknown += 1
            continue
        owned += 1
        if msg.text:
            eligible += 1
        else:
            excluded += 1
    return {
        "total_blocks": total,
        "unique_message_blocks": len(seen_ids),
        "duplicate_blocks": duplicate_blocks,
        "owner_label_counts": counts,
        "owned_blocks": owned,
        "unknown_author_blocks": unknown,
        "eligible_text_blocks": eligible,
        "excluded_empty_or_media_blocks": excluded,
        "ambiguous_or_truncated_blocks": ambiguous,
        "unresolved_identity_count": _unresolved_identity_count(manifest),
    }


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


def build_profile(export_dir: Path, manifest: dict | None = None) -> tuple[dict, dict]:
    """Build local aggregate-only style data and a safe counts report."""
    manifest = manifest or default_owner_manifest()
    voice_settings = load_voice_config()
    style_sets = _voice_style_sets(voice_settings)
    normalization = compile_vocative_elongations(
        style_sets["canonical_forms"], flags=re.IGNORECASE | re.UNICODE,
    )
    normalization_patterns = [normalization] if normalization is not None else []
    owner_map = _manifest_labels(manifest)
    export_files = _export_files(export_dir)
    if not export_files:
        raise ValueError("export_files_not_found")
    source_digest = hashlib.sha256()
    root = export_dir.resolve(strict=True)
    for path in export_files:
        # Digest the exact source bytes without exposing file names or contents.
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                size += len(chunk)
                file_digest.update(chunk)
        source_digest.update(size.to_bytes(8, "big"))
        source_digest.update(file_digest.digest())

    lengths: list[int] = []
    word_lengths: list[int] = []
    sentence_counts: list[int] = []
    emoji_message_count = 0
    newline_message_count = 0
    uppercase_message_count = 0
    casual_message_count = 0
    mild_interjection_message_count = 0
    mild_expletive_message_count = 0
    punctuation = Counter()
    safe_words = Counter()
    abbreviations = Counter()
    frequent_words = Counter()
    owner_label_tokens = {
        token.casefold().replace("ё", "е")
        for labels in manifest["owners"].values()
        for label in labels
        for token in _WORD_RE.findall(unicodedata.normalize("NFC", label))
    }
    label_counts = _manifest_public_counts(manifest)
    total = owned = eligible = excluded = unknown = ambiguous = 0
    for msg in iter_export_messages(root, manifest):
        total += 1
        count_label = _count_label(label_counts, msg.author_label)
        if count_label is not None:
            label_counts[count_label] += 1
        if msg.ambiguous:
            ambiguous += 1
        if msg.author_identity is None:
            unknown += 1
            continue
        owned += 1
        clean = _safe_text(msg.text)
        if len(clean) < 2:
            excluded += 1
            continue
        eligible += 1
        # Only scalar features and allowlisted generic terms survive iteration.
        lengths.append(min(len(clean), MAX_PROFILE_TEXT))
        words = [
            _normalize_style_word(w, style_sets, normalization_patterns)
            for w in _WORD_RE.findall(clean)
        ]
        message_words = set(words)
        word_lengths.append(min(len(words), 500))
        sentence_counts.append(min(max(1, len([s for s in _SENTENCE_RE.findall(clean) if s.strip()])), 20))
        if _EMOJI_RE.search(clean):
            emoji_message_count += 1
        if "\n" in msg.text:
            newline_message_count += 1
        letters = [c for c in clean if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.6:
            uppercase_message_count += 1
        for char in clean:
            if char in _PUNCTUATION:
                punctuation[char] += 1
        for word in words:
            if word in (_SAFE_STYLE_WORDS | style_sets["fillers"]):
                safe_words[word] += 1
            if word in _SAFE_ABBREVIATIONS:
                abbreviations[word] += 1
        for word in message_words & style_sets["frequent"]:
            if word not in owner_label_tokens:
                frequent_words[word] += 1
        if message_words & style_sets["casual"]:
            casual_message_count += 1
        if message_words & _SAFE_MILD_INTERJECTION_WORDS:
            mild_interjection_message_count += 1
        if message_words & style_sets["mild_expletives"]:
            mild_expletive_message_count += 1
        # Drop message text immediately; only counters and integer metrics remain.
        del clean, words, msg

    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_fingerprint = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    punctuation_total = sum(punctuation.values()) or 1
    profile = {
        "schema_version": 1,
        "source_fingerprint": source_digest.hexdigest(),
        "manifest_fingerprint": manifest_fingerprint,
        "sample_count": eligible,
        "owner_label_counts": label_counts,
        "unresolved_identity_count": _unresolved_identity_count(manifest),
        "aggregate_style": {
            "message_chars": {
                "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "median": _quantile(lengths, 0.5),
                "p90": _quantile(lengths, 0.9),
            },
            "words_per_message_mean": round(sum(word_lengths) / len(word_lengths), 1) if word_lengths else 0,
            "sentences_per_message_mean": round(sum(sentence_counts) / len(sentence_counts), 2) if sentence_counts else 0,
            "emoji_message_rate": round(emoji_message_count / eligible, 3) if eligible else 0,
            "newline_message_rate": round(newline_message_count / eligible, 3) if eligible else 0,
            "uppercase_message_rate": round(uppercase_message_count / eligible, 3) if eligible else 0,
            "casual_register_message_rate": round(casual_message_count / eligible, 3) if eligible else 0,
            "mild_interjection_message_rate": round(mild_interjection_message_count / eligible, 3) if eligible else 0,
            "mild_expletive_message_rate": round(mild_expletive_message_count / eligible, 3) if eligible else 0,
            "frequent_words": [
                # Keep enough precision for the serializer to verify the
                # five-distinct-message threshold even on large exports.
                {"word": word, "message_rate": round(count / eligible, 12)}
                for word, count in sorted(
                    frequent_words.items(),
                    key=lambda item: (
                        item[0] not in style_sets["priority"],
                        -item[1], item[0],
                    ),
                )
                if count >= 5
            ][:20],
            "punctuation_rates": {
                key: round(punctuation[key] / punctuation_total, 3)
                for key in sorted(_PUNCTUATION) if punctuation[key]
            },
            "generic_fillers": [word for word, n in safe_words.most_common(8) if n >= 3],
            "abbreviations": [word for word, n in abbreviations.most_common(8) if n >= 3],
        },
        "style_rules": [],
    }
    report = {
        "total_blocks": total,
        "owner_label_counts": label_counts,
        "owned_blocks": owned,
        "unknown_author_blocks": unknown,
        "eligible_text_blocks": eligible,
        "excluded_empty_or_media_blocks": excluded,
        "ambiguous_or_truncated_blocks": ambiguous,
        "unresolved_identity_count": _unresolved_identity_count(manifest),
    }
    return profile, report


PROFILE_API_FIELDS = frozenset({"sample_count", "aggregate_style", "style_rules"})


def profile_for_api(profile: dict, max_bytes: int = 3_000) -> dict:
    """Strictly serialize only allowlisted aggregate fields for Gemini."""
    style_sets = _voice_style_sets()
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("profile_unavailable")
    sample_count = profile.get("sample_count", 0)
    if not isinstance(sample_count, int) or sample_count < 10:
        raise ValueError("profile_unavailable")
    aggregate = profile.get("aggregate_style")
    if not isinstance(aggregate, dict):
        raise ValueError("profile_unavailable")
    legacy_aggregate = {
        "message_chars", "words_per_message_mean", "sentences_per_message_mean",
        "emoji_message_rate", "newline_message_rate", "uppercase_message_rate",
        "punctuation_rates", "generic_fillers", "abbreviations",
    }
    previous_extended_aggregate = legacy_aggregate | {
        "casual_register_message_rate", "mild_profanity_message_rate", "frequent_words",
    }
    extended_aggregate = legacy_aggregate | {
        "casual_register_message_rate", "mild_interjection_message_rate",
        "mild_expletive_message_rate", "frequent_words",
    }
    aggregate_shape = set(aggregate)
    if aggregate_shape not in (legacy_aggregate, previous_extended_aggregate, extended_aggregate):
        raise ValueError("profile_unavailable")
    lengths = aggregate.get("message_chars")
    if not isinstance(lengths, dict) or set(lengths) != {"mean", "median", "p90"}:
        raise ValueError("profile_unavailable")
    if any(not isinstance(value, (int, float)) or not 0 <= value <= MAX_PROFILE_TEXT
           for value in lengths.values()):
        raise ValueError("profile_unavailable")
    for key in ("words_per_message_mean", "sentences_per_message_mean"):
        if not isinstance(aggregate.get(key), (int, float)) or not 0 <= aggregate[key] <= 500:
            raise ValueError("profile_unavailable")
    for key in ("emoji_message_rate", "newline_message_rate", "uppercase_message_rate"):
        if not isinstance(aggregate.get(key), (int, float)) or not 0 <= aggregate[key] <= 1:
            raise ValueError("profile_unavailable")
    if aggregate_shape == previous_extended_aggregate:
        # Older profiles called euphemistic interjections "profanity". Preserve
        # their aggregate as interjections; never treat it as actual expletives.
        casual_rate = aggregate["casual_register_message_rate"]
        interjection_rate = aggregate["mild_profanity_message_rate"]
        expletive_rate = 0
    else:
        casual_rate = aggregate.get("casual_register_message_rate", 0)
        interjection_rate = aggregate.get("mild_interjection_message_rate", 0)
        expletive_rate = aggregate.get("mild_expletive_message_rate", 0)
    for value in (casual_rate, interjection_rate, expletive_rate):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("profile_unavailable")
    punctuation_rates = aggregate.get("punctuation_rates")
    if not isinstance(punctuation_rates, dict) or len(punctuation_rates) > len(_PUNCTUATION):
        raise ValueError("profile_unavailable")
    if any(key not in _PUNCTUATION or not isinstance(value, (int, float)) or not 0 <= value <= 1
           for key, value in punctuation_rates.items()):
        raise ValueError("profile_unavailable")
    for key, allowed in (
        ("generic_fillers", _SAFE_STYLE_WORDS | style_sets["fillers"]),
        ("abbreviations", _SAFE_ABBREVIATIONS),
    ):
        values = aggregate.get(key)
        if not isinstance(values, list) or len(values) > 8:
            raise ValueError("profile_unavailable")
        if any(not isinstance(value, str) or value not in allowed for value in values):
            raise ValueError("profile_unavailable")
    frequent = aggregate.get("frequent_words", [])
    if not isinstance(frequent, list) or len(frequent) > 20:
        raise ValueError("profile_unavailable")
    safe_frequent = []
    seen_words = set()
    for item in frequent:
        if not isinstance(item, dict) or set(item) != {"word", "message_rate"}:
            raise ValueError("profile_unavailable")
        word, rate = item["word"], item["message_rate"]
        if (not isinstance(word, str) or word not in style_sets["frequent"]
                or word in seen_words or not isinstance(rate, (int, float))
                or not 0 < rate <= 1
                or rate * sample_count < 4.999):
            raise ValueError("profile_unavailable")
        seen_words.add(word)
        safe_frequent.append({"word": word, "message_rate": float(rate)})
    safe_aggregate = {
        **{key: aggregate[key] for key in legacy_aggregate},
        "casual_register_message_rate": float(casual_rate),
        "mild_interjection_message_rate": float(interjection_rate),
        "mild_expletive_message_rate": float(expletive_rate),
        "frequent_words": safe_frequent,
    }
    result = {
        "sample_count": int(profile["sample_count"]),
        "aggregate_style": safe_aggregate,
        "style_rules": [],
    }
    try:
        requested_limit = int(max_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("profile_unavailable") from exc
    if requested_limit <= 0:
        raise ValueError("profile_unavailable")
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > min(requested_limit, 3_000):
        raise ValueError("profile_unavailable")
    return json.loads(encoded.decode("utf-8"))


def load_private_profile(profile_path: str | Path) -> dict:
    """Read a profile only from ignored, owner-only data/ storage."""
    root = (ROOT / "data").resolve()
    path = Path(profile_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink():
        raise ValueError("profile_unavailable")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("profile_unavailable") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError("profile_unavailable")
    parent_mode = resolved.parent.stat().st_mode
    if parent_mode & 0o077 or root.stat().st_mode & 0o077:
        raise ValueError("profile_unavailable")
    try:
        profile = json.loads(resolved.read_text(encoding="utf-8"))
        profile_for_api(profile)
        return profile
    except Exception as exc:
        raise ValueError("profile_unavailable") from exc


def _private_path(path: Path) -> Path:
    root = (ROOT / "data").resolve()
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    parent = path.parent.resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("private_output_must_be_under_data") from exc
    if parent == root and path.name in {"", ".", ".."}:
        raise ValueError("private_output_invalid")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    os.chmod(parent, 0o700)
    return parent / path.name


def write_private_artifacts(profile: dict, manifest: dict, profile_path: Path,
                            manifest_path: Path, overwrite: bool = False) -> None:
    """Atomically write owner-only files beneath ignored data/.

    Without --overwrite, existing targets are never opened or modified.
    """
    profile_target = _private_path(profile_path)
    manifest_target = _private_path(manifest_path)
    if profile_target == manifest_target:
        raise ValueError("private_paths_must_differ")
    payloads = (
        (profile_target, json.dumps(profile, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
        (manifest_target, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
    )
    temporary: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for target, text in payloads:
            if target.exists() and not overwrite:
                raise FileExistsError("private_profile_exists")
            fd, temp_name = tempfile.mkstemp(prefix=".style-", dir=target.parent)
            temp_path = Path(temp_name)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.append((temp_path, target))
        for temp_path, target in temporary:
            if overwrite:
                os.replace(temp_path, target)
            else:
                # Hard-link installation is exclusive, so races cannot replace
                # an existing profile when overwrite was not requested.
                os.link(temp_path, target)
                temp_path.unlink()
            installed.append(target)
        for target in installed:
            os.chmod(target, 0o600)
    except Exception:
        if not overwrite:
            for target in installed:
                try:
                    target.unlink()
                except OSError:
                    pass
        raise
    finally:
        for temp_path, _target in temporary:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
