"""Local-only sticker catalogs for ordinary replies."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from telethon import functions, types

MAX_RECENT_STICKERS = 80
MAX_STICKER_PACK_ITEMS = 400
RECENT_STICKER_CACHE_SECONDS = 5 * 60
GENERIC_STICKER_MARKER = "[стикер]"
PACK_CATALOG_REFRESH_SECONDS = 60 * 60
PACK_CATALOG_LOOKUP_SECONDS = 5.0
PACK_LOOKUP_CONCURRENCY = 4
_KEYCAP_RE = re.compile(r"[#*0-9]\ufe0f?\u20e3\Z")
_STICKER_BLOCKLIST_MAX_BYTES = 32 * 1024
_STICKER_BLOCKLIST_MAX_IDS = 1000
_STICKER_PACK_CONFIG_MAX_BYTES = 16 * 1024
_STICKER_PACK_CONFIG_MAX_TITLES = 32
_STICKER_PACK_CONFIG_MAX_TITLE_LENGTH = 128
_MAX_TELEGRAM_DOCUMENT_ID = (1 << 63) - 1


def _read_private_style_file(filename: str, max_bytes: int) -> tuple[bool, bytes | None]:
    """Read a bounded owner-only file below data/style; None bytes means absent."""
    if Path(filename).name != filename or filename in ("", ".", ".."):
        return False, None
    data_dir = Path(__file__).resolve().parent.parent / "data"
    style_dir = data_dir / "style"
    private_path = style_dir / filename

    try:
        data_stat = data_dir.lstat()
    except FileNotFoundError:
        return True, None
    except OSError:
        return False, None
    if not stat.S_ISDIR(data_stat.st_mode):
        return False, None

    try:
        style_stat = style_dir.lstat()
    except FileNotFoundError:
        return True, None
    except OSError:
        return False, None
    if not stat.S_ISDIR(style_stat.st_mode):
        return False, None

    try:
        private_stat = private_path.lstat()
    except FileNotFoundError:
        return True, None
    except OSError:
        return False, None
    if not stat.S_ISREG(private_stat.st_mode):
        return False, None

    try:
        owner_uid = os.getuid()
        for directory_stat in (data_stat, style_stat):
            mode = stat.S_IMODE(directory_stat.st_mode)
            if (directory_stat.st_uid != owner_uid or mode & 0o077
                    or mode & 0o500 != 0o500):
                return False, None
        file_mode = stat.S_IMODE(private_stat.st_mode)
        if (private_stat.st_uid != owner_uid or file_mode & 0o077
                or file_mode & 0o400 != 0o400
                or private_stat.st_size > max_bytes):
            return False, None

        data_real = data_dir.resolve(strict=True)
        private_real = private_path.resolve(strict=True)
        try:
            private_real.relative_to(data_real)
        except ValueError:
            return False, None

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(private_path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            opened_mode = stat.S_IMODE(opened_stat.st_mode)
            if (not stat.S_ISREG(opened_stat.st_mode)
                    or opened_stat.st_uid != owner_uid
                    or opened_mode & 0o077
                    or opened_mode & 0o400 != 0o400
                    or opened_stat.st_size > max_bytes):
                return False, None
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                raw = source.read(max_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(raw) > max_bytes:
            return False, None
        return True, raw
    except (OSError, ValueError, RuntimeError):
        return False, None



def _json_object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_sticker_blocklist() -> frozenset[int] | None:
    """Load the optional private document-ID list; None means fail closed."""
    valid, raw = _read_private_style_file(
        "sticker_blocklist.json", _STICKER_BLOCKLIST_MAX_BYTES,
    )
    if not valid:
        return None
    if raw is None:
        return frozenset()
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_object_without_duplicate_keys,
        )
        if (not isinstance(parsed, dict)
                or set(parsed) != {"document_ids"}
                or not isinstance(parsed["document_ids"], list)
                or len(parsed["document_ids"]) > _STICKER_BLOCKLIST_MAX_IDS):
            return None
        identifiers = parsed["document_ids"]
        if any(type(identifier) is not int or identifier <= 0
               or identifier > _MAX_TELEGRAM_DOCUMENT_ID
               for identifier in identifiers):
            return None
        if len(set(identifiers)) != len(identifiers):
            return None
        return frozenset(identifiers)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError,
            TypeError, OverflowError, RecursionError):
        return None


def _load_priority_pack_titles() -> tuple[str, ...]:
    """Load configured pack priorities; missing/invalid config means no priority."""
    valid, raw = _read_private_style_file(
        "stickers.json", _STICKER_PACK_CONFIG_MAX_BYTES,
    )
    if not valid or raw is None:
        return ()
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_object_without_duplicate_keys,
        )
        if (not isinstance(parsed, dict)
                or set(parsed) != {"priority_pack_titles"}
                or not isinstance(parsed["priority_pack_titles"], list)
                or len(parsed["priority_pack_titles"]) > _STICKER_PACK_CONFIG_MAX_TITLES):
            return ()
        normalized = []
        seen = set()
        for title in parsed["priority_pack_titles"]:
            if type(title) is not str:
                return ()
            title = " ".join(title.split())
            if (not title or len(title) > _STICKER_PACK_CONFIG_MAX_TITLE_LENGTH
                    or any(ord(char) < 0x20 or ord(char) == 0x7f for char in title)):
                return ()
            key = title.casefold()
            if key in seen:
                return ()
            seen.add(key)
            normalized.append(key)
        return tuple(normalized)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError,
            TypeError, RecursionError):
        return ()


def is_valid_sticker_emoji(value) -> bool:
    """Accept one short emoji sequence, never labels or arbitrary text."""
    if not isinstance(value, str):
        return False
    value = unicodedata.normalize("NFC", value.strip())
    if not value or len(value) > 24 or any(char.isspace() for char in value):
        return False
    # ASCII keycap bases are meaningful only as a complete emoji sequence.
    # Otherwise strings such as "😀123" could pass as one sticker emoji.
    if any(char in "#*0123456789" for char in value):
        return bool(_KEYCAP_RE.fullmatch(value))
    bases = 0
    for char in value:
        code = ord(char)
        if (0x1F000 <= code <= 0x1FAFF
                or 0x2600 <= code <= 0x27BF
                or 0x2300 <= code <= 0x23FF
                or 0x1F1E6 <= code <= 0x1F1FF):
            bases += 1
            continue
        if (0xFE0E <= code <= 0xFE0F
                or code == 0x200D
                or code == 0x20E3
                or 0x1F3FB <= code <= 0x1F3FF
                or 0xE0020 <= code <= 0xE007F):
            continue
        return False
    return bases > 0


def document_sticker_emoji(document) -> str | None:
    """Return the alt of a sticker document when it is a valid emoji."""
    attributes = getattr(document, "attributes", None) or ()
    for attribute in attributes:
        if type(attribute).__name__ != "DocumentAttributeSticker":
            continue
        alt = getattr(attribute, "alt", None)
        if is_valid_sticker_emoji(alt):
            return unicodedata.normalize("NFC", alt.strip())
    return None


def sticker_context_marker(message) -> str | None:
    """Expose only an emoji marker for an incoming sticker, never its media."""
    media = getattr(message, "media", None)
    document = getattr(media, "document", None)
    if document is None:
        document = getattr(message, "document", None)
    attributes = getattr(document, "attributes", None) or ()
    if not any(type(attribute).__name__ == "DocumentAttributeSticker"
               for attribute in attributes):
        return None
    return document_sticker_emoji(document) or GENERIC_STICKER_MARKER


@dataclass(frozen=True)
class RecentSticker:
    emoji: str
    document: object
    pack_priority: int = 0
    pack_favorite: bool = False
    pack_key: str = ""


def sticker_document_key(document, emoji: str = "") -> str:
    """Return a local-only stable key for deduping and repeat limits."""
    identifier = getattr(document, "id", None)
    return f"id:{identifier}" if identifier is not None else f"emoji:{emoji}"


class RecentStickerCatalog:
    """Bounded, short-lived cache of local-account recent sticker documents."""

    def __init__(self, *, max_items: int = MAX_RECENT_STICKERS,
                 ttl_seconds: float = RECENT_STICKER_CACHE_SECONDS,
                 clock=time.monotonic):
        self.max_items = max(1, min(int(max_items), MAX_STICKER_PACK_ITEMS))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._clock = clock
        self._refresh_after = float("-inf")
        self._items: tuple[RecentSticker, ...] = ()
        self._blocked_document_ids: frozenset[int] = frozenset()
        self._lock = asyncio.Lock()
        self.refresh_count = 0

    def _sync_blocklist(self) -> bool:
        blocked_ids = _load_sticker_blocklist()
        if blocked_ids is None:
            self._items = ()
            self._refresh_after = float("-inf")
            if hasattr(self, "catalog_status"):
                self.catalog_status = "blocklist_invalid"
            return False
        self._blocked_document_ids = blocked_ids
        self._items = tuple(
            item for item in self._items
            if not self._document_is_blocked(item.document)
        )
        if getattr(self, "catalog_status", None) == "blocklist_invalid":
            self.catalog_status = "not_loaded"
        return True

    def _document_is_blocked(self, document) -> bool:
        identifier = getattr(document, "id", None)
        return identifier is not None and identifier in self._blocked_document_ids

    @property
    def emojis(self) -> tuple[str, ...]:
        if not self._sync_blocklist():
            return ()
        seen = set()
        ordered = []
        for item in self._items:
            if item.emoji not in seen:
                seen.add(item.emoji)
                ordered.append(item.emoji)
        return tuple(ordered)

    def find(self, emoji: str, *, exclude_pack_priorities=(),
             exclude_pack_keys=()) -> RecentSticker | None:
        if not is_valid_sticker_emoji(emoji) or not self._sync_blocklist():
            return None
        return self._pick_pack_balanced(
            (item for item in self._items if item.emoji == emoji),
            exclude_pack_priorities=exclude_pack_priorities,
            exclude_pack_keys=exclude_pack_keys,
        )

    def find_alternative(self, emoji: str, *, exclude_document_ids=(),
                         exclude_pack_priorities=(), exclude_pack_keys=()) -> RecentSticker | None:
        if not is_valid_sticker_emoji(emoji) or not self._sync_blocklist():
            return None
        excluded = set(exclude_document_ids)
        return self._pick_pack_balanced(
            (item for item in self._items
             if item.emoji == emoji
             and sticker_document_key(item.document, item.emoji) not in excluded),
            exclude_pack_priorities=exclude_pack_priorities,
            exclude_pack_keys=exclude_pack_keys,
        )

    def _pick_pack_balanced(self, candidates, *, exclude_pack_priorities=(),
                            exclude_pack_keys=()) -> RecentSticker | None:
        """Spread matching sticker choices across packs with configured favorites weighted."""
        by_pack: dict[object, list[RecentSticker]] = {}
        for item in candidates:
            key = item.pack_key or item.pack_priority
            by_pack.setdefault(key, []).append(item)
        excluded_packs = {int(value) for value in exclude_pack_priorities}
        excluded_keys = {str(value) for value in exclude_pack_keys}
        alternatives = {
            key: value for key, value in by_pack.items()
            if key not in excluded_packs and str(key) not in excluded_keys
        }
        if alternatives:
            by_pack = alternatives
        if not by_pack:
            return None
        pack_priorities = tuple(by_pack)
        preferred = tuple(
            priority for priority in pack_priorities
            if any(item.pack_favorite for item in by_pack[priority])
        )
        regular = tuple(priority for priority in pack_priorities if priority not in preferred)
        if preferred and regular:
            selected_bucket = random.choices((preferred, regular), weights=(0.7, 0.3), k=1)[0]
        else:
            selected_bucket = preferred or regular
        selected_pack = random.choice(selected_bucket)
        return random.choice(by_pack[selected_pack])

    async def get(self, client) -> tuple[RecentSticker, ...]:
        if not self._sync_blocklist():
            return ()
        now = self._clock()
        if now < self._refresh_after:
            return self._items
        async with self._lock:
            if not self._sync_blocklist():
                return ()
            now = self._clock()
            if now < self._refresh_after:
                return self._items
            try:
                result = await client(functions.messages.GetRecentStickersRequest(
                    hash=0, attached=False,
                ))
                documents = getattr(result, "stickers", None) or ()
                items = []
                seen_documents = set()
                for document in documents:
                    if self._document_is_blocked(document):
                        continue
                    emoji = document_sticker_emoji(document)
                    if emoji is None:
                        continue
                    identifier = getattr(document, "id", None)
                    if identifier is not None and identifier in seen_documents:
                        continue
                    if identifier is not None:
                        seen_documents.add(identifier)
                    items.append(RecentSticker(emoji, document))
                    if len(items) >= self.max_items:
                        break
                self._items = tuple(items)
                self._refresh_after = self._clock() + self.ttl_seconds
            except Exception as exc:
                # A catalog outage must only remove the optional sticker path.
                self._items = ()
                delay = self.ttl_seconds
                if type(exc).__name__ == "FloodWaitError":
                    try:
                        seconds = float(getattr(exc, "seconds", 0))
                        if math.isfinite(seconds) and seconds > 0:
                            delay = max(delay, seconds)
                    except (TypeError, ValueError, OverflowError):
                        pass
                self._refresh_after = self._clock() + delay
            if not self._sync_blocklist():
                return ()
            self._items = tuple(
                item for item in self._items
                if not self._document_is_blocked(item.document)
            )
            return self._items


class StickerPackCatalog(RecentStickerCatalog):
    """Cache stickers from installed and archived sets, with preferred packs first."""

    def __init__(self, *, pack_title: str | None = None,
                 priority_pack_titles=None, **kwargs):
        kwargs.setdefault("max_items", MAX_STICKER_PACK_ITEMS)
        kwargs.setdefault("ttl_seconds", PACK_CATALOG_REFRESH_SECONDS)
        super().__init__(**kwargs)
        configured = list(
            _load_priority_pack_titles()
            if priority_pack_titles is None else priority_pack_titles
        )
        if pack_title:
            configured.insert(0, pack_title)
        self.priority_pack_titles = tuple(
            " ".join(str(title).split()).casefold()
            for title in configured if str(title).strip()
        )
        self.catalog_status = "not_loaded"

    async def get(self, client) -> tuple[RecentSticker, ...]:
        if not self._sync_blocklist():
            return ()
        now = self._clock()
        if now < self._refresh_after:
            return self._items
        async with self._lock:
            if not self._sync_blocklist():
                return ()
            now = self._clock()
            if now < self._refresh_after:
                return self._items
            self.refresh_count += 1
            try:
                installed = await client(functions.messages.GetAllStickersRequest(hash=0))
                packs = []
                seen_pack_names = set()
                for index, pack in enumerate(getattr(installed, "sets", None) or ()):
                    short_name = str(getattr(pack, "short_name", "")).casefold()
                    if short_name and short_name not in seen_pack_names:
                        seen_pack_names.add(short_name)
                        packs.append((index, pack, False))

                archived_incomplete = False
                try:
                    async with asyncio.timeout(PACK_CATALOG_LOOKUP_SECONDS):
                        offset_id = 0
                        archived_count = None
                        archived_index = 0
                        while archived_count is None or archived_index < archived_count:
                            archived = await client(
                                functions.messages.GetArchivedStickersRequest(
                                    offset_id=offset_id, limit=100,
                                    masks=False, emojis=False,
                                )
                            )
                            archived_count = int(getattr(archived, "count", 0) or 0)
                            covered_sets = list(getattr(archived, "sets", None) or ())
                            if not covered_sets:
                                break
                            last_set_id = None
                            for covered in covered_sets:
                                pack = getattr(covered, "set", None)
                                if pack is None:
                                    continue
                                archived_index += 1
                                short_name = str(getattr(pack, "short_name", "")).casefold()
                                if short_name and short_name not in seen_pack_names:
                                    seen_pack_names.add(short_name)
                                    packs.append((archived_index, pack, True))
                                last_set_id = getattr(pack, "id", None)
                            if last_set_id is None or last_set_id == offset_id:
                                break
                            offset_id = last_set_id
                except asyncio.CancelledError:
                    raise
                except Exception:
                    archived_incomplete = True

                packs = sorted(packs, key=self._pack_order)
                grouped_items: list[list[RecentSticker]] = [[] for _ in packs]
                semaphore = asyncio.Semaphore(PACK_LOOKUP_CONCURRENCY)

                async def fetch_pack(position, pack):
                    short_name = getattr(pack, "short_name", None)
                    if not short_name:
                        return position, ()
                    try:
                        async with semaphore:
                            result = await client(functions.messages.GetStickerSetRequest(
                                stickerset=types.InputStickerSetShortName(short_name),
                                hash=0,
                            ))
                        return position, getattr(result, "documents", None) or ()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        return position, ()

                tasks = [
                    asyncio.create_task(fetch_pack(position, pack))
                    for position, (_, pack, _) in enumerate(packs)
                ]
                pending = []
                try:
                    if tasks:
                        done, pending = await asyncio.wait(
                            tasks, timeout=PACK_CATALOG_LOOKUP_SECONDS,
                        )
                    else:
                        done, pending = set(), []
                    for task in done:
                        try:
                            position, documents = task.result()
                        except Exception:
                            continue
                        seen_in_pack = set()
                        pack_rank = position
                        pack_title = " ".join(
                            str(getattr(pack, "title", "")).split(),
                        ).casefold()
                        pack_favorite = any(
                            pack_title == preferred or preferred in pack_title
                            for preferred in self.priority_pack_titles
                        )
                        pack_key = str(getattr(pack, "id", None) or short_name)
                        for document in documents:
                            if self._document_is_blocked(document):
                                continue
                            emoji = document_sticker_emoji(document)
                            if emoji is None:
                                continue
                            identifier = getattr(document, "id", None)
                            if identifier is not None and identifier in seen_in_pack:
                                continue
                            if identifier is not None:
                                seen_in_pack.add(identifier)
                            grouped_items[position].append(
                                RecentSticker(
                                    emoji, document, pack_rank, pack_favorite, pack_key,
                                ),
                            )
                finally:
                    pending = [task for task in tasks if not task.done()]
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                items = []
                seen_documents = set()
                for depth in range(max((len(group) for group in grouped_items), default=0)):
                    for group in grouped_items:
                        if depth >= len(group):
                            continue
                        item = group[depth]
                        identifier = getattr(item.document, "id", None)
                        if identifier is not None and identifier in seen_documents:
                            continue
                        if identifier is not None:
                            seen_documents.add(identifier)
                        items.append(item)
                        if len(items) >= self.max_items:
                            break
                    if len(items) >= self.max_items:
                        break
                if not self._sync_blocklist():
                    return ()
                items = [item for item in items
                         if not self._document_is_blocked(item.document)]
                self._items = tuple(items)
                self.catalog_status = "ready" if items else "pack_empty"
                refresh_seconds = (
                    min(self.ttl_seconds, RECENT_STICKER_CACHE_SECONDS)
                    if pending or archived_incomplete else self.ttl_seconds
                )
                self._refresh_after = self._clock() + refresh_seconds
            except Exception as exc:
                self._items = ()
                self.catalog_status = "unavailable"
                delay = self.ttl_seconds
                if type(exc).__name__ == "FloodWaitError":
                    try:
                        seconds = float(getattr(exc, "seconds", 0))
                        if math.isfinite(seconds) and seconds > 0:
                            delay = max(delay, seconds)
                    except (TypeError, ValueError, OverflowError):
                        pass
                self._refresh_after = self._clock() + delay
            return self._items

    def _pack_order(self, indexed_pack):
        original_index, pack, archived = indexed_pack
        title = " ".join(str(getattr(pack, "title", "")).split()).casefold()
        priority = len(self.priority_pack_titles)
        for index, preferred in enumerate(self.priority_pack_titles):
            if title == preferred or preferred in title:
                priority = index
                break
        return priority, bool(archived), original_index
