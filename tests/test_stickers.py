import asyncio
from types import SimpleNamespace

from tg_bot import context
from tg_bot.stickers import (
    GENERIC_STICKER_MARKER, RecentStickerCatalog, document_sticker_emoji,
    is_valid_sticker_emoji, sticker_context_marker,
)


class DocumentAttributeSticker:
    def __init__(self, alt):
        self.alt = alt


def _document(identifier, alt):
    return SimpleNamespace(
        id=identifier,
        access_hash=identifier + 100,
        file_reference=b"local-only",
        attributes=[DocumentAttributeSticker(alt)],
    )


def test_sticker_alt_is_validated_and_context_contains_only_marker():
    good = _document(1, "😀")
    bad = _document(2, "untrusted label")
    assert is_valid_sticker_emoji("😀")
    assert is_valid_sticker_emoji("1️⃣")
    assert is_valid_sticker_emoji("#⃣")
    assert not is_valid_sticker_emoji("😀123")
    assert not is_valid_sticker_emoji("😀1️⃣")
    assert not is_valid_sticker_emoji("1")
    assert not is_valid_sticker_emoji("untrusted label")
    assert document_sticker_emoji(good) == "😀"
    assert sticker_context_marker(SimpleNamespace(media=SimpleNamespace(document=bad))) == GENERIC_STICKER_MARKER

    message = SimpleNamespace(
        id=7, sender_id=9, out=False, message="", entities=[],
        fwd_from=None, date=None, reply_to_msg_id=None,
        media=SimpleNamespace(document=good),
    )
    snapshot = context.snapshot_from_message(message, -100, 42)
    assert snapshot.text == "😀"
    assert not hasattr(snapshot, "media")


def test_recent_sticker_catalog_uses_telethon_method_and_keeps_documents_local():
    clock = [0.0]
    docs = [_document(11, "😀"), _document(12, "😀"), _document(13, "🤖")]

    class Client:
        calls = 0

        async def __call__(self, request):
            self.calls += 1
            assert type(request).__name__ == "GetRecentStickersRequest"
            assert request.hash == 0
            assert request.attached is False
            return SimpleNamespace(stickers=docs)

    client = Client()
    catalog = RecentStickerCatalog(max_items=3, ttl_seconds=5, clock=lambda: clock[0])

    async def scenario():
        first = await catalog.get(client)
        assert len(first) == 3
        assert catalog.emojis == ("😀", "🤖")
        assert catalog.find("😀").document is docs[0]
        assert not hasattr(first[0], "access_hash")
        clock[0] = 2
        assert await catalog.get(client) == first
        assert client.calls == 1
        clock[0] = 6
        await catalog.get(client)
        assert client.calls == 2

    asyncio.run(scenario())


def test_recent_sticker_catalog_fails_closed_to_empty():
    class BrokenClient:
        async def __call__(self, _request):
            raise RuntimeError("private transport detail")

    catalog = RecentStickerCatalog()
    assert asyncio.run(catalog.get(BrokenClient())) == ()
    assert catalog.emojis == ()


def test_recent_sticker_catalog_honors_flood_wait_before_retry():
    clock = [0.0]

    class FloodWaitError(Exception):
        seconds = 40

    class Client:
        calls = 0

        async def __call__(self, _request):
            self.calls += 1
            if self.calls == 1:
                raise FloodWaitError()
            return SimpleNamespace(stickers=[_document(14, "🤖")])

    client = Client()
    catalog = RecentStickerCatalog(ttl_seconds=5, clock=lambda: clock[0])

    async def scenario():
        assert await catalog.get(client) == ()
        assert catalog.emojis == ()
        clock[0] = 39.9
        assert await catalog.get(client) == ()
        assert client.calls == 1
        clock[0] = 40
        assert catalog.emojis == ()
        result = await catalog.get(client)
        assert len(result) == 1
        assert catalog.emojis == ("🤖",)
        assert client.calls == 2

    asyncio.run(scenario())
