"""Reliability checks using synthetic events, private temporary state and fake I/O."""

import asyncio
import sqlite3
import time
from types import SimpleNamespace

import pytest
from telethon.tl.types import MessageReplyHeader

from tg_bot import reply_ai, runner
from tg_bot.context import ContextBuffer, snapshot_from_message
from tg_bot.reply_ai import OrdinaryReply, ReplyResult
from tg_bot.state import State


class MessageEntityMentionName:
    offset = 0
    length = 4
    user_id = 42


class Event:
    chat_id = -100
    is_group = True
    chat = SimpleNamespace(broadcast=False)

    def __init__(self, message_id, *, mention=True, reply=False, sender="ok",
                 parent="ok", text="Ping synthetic trigger body", chat_id=-100):
        self.chat_id = chat_id
        self.sender_mode = sender
        self.parent_mode = parent
        self.message = SimpleNamespace(
            id=message_id, date=SimpleNamespace(
                timestamp=lambda: time.time() + message_id / 1_000_000,
            ),
            out=False, sender_id=9, message=text,
            entities=[MessageEntityMentionName()] if mention else [], fwd_from=None,
            reply_to_msg_id=17 if reply else None, reply_to=None, message_thread_id=0,
        )

    async def get_sender(self):
        if self.sender_mode == "timeout":
            await asyncio.Event().wait()
        if self.sender_mode == "error":
            raise RuntimeError("PRIVATE-SENDER-ERROR")
        if self.sender_mode == "none":
            return None
        return SimpleNamespace(id=9, bot=self.sender_mode == "bot")

    async def get_reply_message(self):
        if self.parent_mode == "timeout":
            await asyncio.Event().wait()
        if self.parent_mode == "error":
            raise RuntimeError("PRIVATE-PARENT-ERROR")
        if self.parent_mode == "none":
            return None
        parent_id = 18 if self.parent_mode == "wrong_thread" else 17
        parent = Event(parent_id, mention=False, text="synthetic owner parent").message
        parent.sender_id = 42
        parent.chat_id = -999 if self.parent_mode == "wrong_chat" else self.chat_id
        if self.parent_mode == "wrong_thread":
            parent.reply_to = MessageReplyHeader(
                forum_topic=True, reply_to_msg_id=19, reply_to_top_id=99,
            )
        return parent


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_periodic_context_pruner_cleans_db_and_ram_while_running(monkeypatch):
    class FakeState:
        calls = 0

        def prune_context_snapshots(self, chats, *, enabled):
            assert chats == {-100}
            assert enabled is True
            self.calls += 1

    class FakeContext:
        calls = 0

        def prune_expired(self):
            self.calls += 1

    async def scenario():
        state = FakeState()
        context = FakeContext()
        monkeypatch.setattr(runner, "CONTEXT_PRUNE_INTERVAL_SECONDS", 0.001)
        task = asyncio.create_task(runner._prune_ordinary_context(
            state, context, {-100},
        ))
        try:
            await wait_until(lambda: state.calls >= 1 and context.calls >= 1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert state.calls >= 1 and context.calls >= 1

    asyncio.run(scenario())


class Harness:
    def __init__(self, tmp_path, monkeypatch, **settings):
        self.tmp_path = tmp_path
        self.state = None
        self.handler = None
        self.generated = []
        self.sends = []
        self.file_sends = []
        self.reads = []
        self.read = None
        self.send = None
        self.generate = None
        self.scenario = None
        self.cfg = SimpleNamespace(
            chats={-100}, chats_error=False, api_id=1, api_hash="synthetic",
            gemini_key="synthetic", model="synthetic", reply_model="synthetic",
            auto_reply_enabled=True, reply_config_error="", style_profile_path="synthetic",
            owner_name_aliases=(), reply_context_before=0, reply_context_after=0,
            reply_wait_seconds=0, control_bot_token="",
        )
        for key, value in settings.items():
            setattr(self.cfg, key, value)

        def make_state(path):
            self.state = State(path)
            return self.state

        monkeypatch.setattr(runner, "ROOT", tmp_path)
        monkeypatch.setattr(runner, "State", make_state)
        monkeypatch.setattr(runner, "client", lambda _cfg: self)
        monkeypatch.setattr(runner, "load_private_profile", lambda _path: {})
        monkeypatch.setattr(runner, "load_private_participant_summaries",
                            lambda *_args: {"participants": {}})
        monkeypatch.setattr(runner, "generate_reply", self.generate_reply)
        monkeypatch.setattr(runner, "ORDINARY_LOOKUP_TIMEOUT", 0.02)
        monkeypatch.setattr(runner, "ORDINARY_DEADLINE_SECONDS", 0.15)

    async def connect(self): pass
    async def is_user_authorized(self): return True
    async def get_me(self): return SimpleNamespace(id=42, username="synthetic_owner")
    def add_event_handler(self, handler, _builder): self.handler = handler
    async def disconnect(self): pass

    async def send_message(self, *args, **kwargs):
        self.sends.append((args, kwargs))
        if self.send:
            return await self.send(*args, **kwargs)
        return SimpleNamespace(id=10000 + len(self.sends))

    async def send_file(self, *args, **kwargs):
        self.file_sends.append((args, kwargs))
        return SimpleNamespace(id=20000 + len(self.file_sends))

    async def iter_messages(self, *args, **kwargs):
        self.reads.append(kwargs)
        if self.read:
            async for message in self.read(*args, **kwargs):
                yield message

    async def generate_reply(self, _key, _model, _profile, window, **_kwargs):
        self.generated.append(window)
        if self.generate:
            return await self.generate(window)
        return ReplyResult("Короткий ответ.", "ok")

    async def run_until_disconnected(self):
        async with asyncio.timeout(2):
            await self.scenario()

    def status(self, message_id):
        row = self.state.db.execute(
            "SELECT status FROM jobs WHERE msg=?", (message_id,),
        ).fetchone()
        return row[0] if row else None

    def run(self, scenario, *, send=False, show_drafts=False):
        self.scenario = scenario
        asyncio.run(runner.run(self.cfg, send=send, show_drafts=show_drafts))
        with sqlite3.connect(self.tmp_path / "data" / "state.sqlite3") as db:
            return dict(db.execute("SELECT msg,status FROM jobs"))


def assert_private_output(output):
    assert "PRIVATE-" not in output
    assert "synthetic trigger body" not in output
    assert "synthetic owner parent" not in output
    assert "message=" not in output
    assert "sender=" not in output


@pytest.mark.parametrize("mode,reason", [
    ("timeout", "sender_lookup_timeout"), ("error", "sender_lookup_error"),
    ("none", "sender_lookup_unavailable"), ("bot", "sender_bot"),
])
def test_sender_failures_are_bounded_observable_and_never_generate(
        tmp_path, monkeypatch, capsys, mode, reason):
    harness = Harness(tmp_path, monkeypatch)

    async def scenario():
        await harness.handler(Event(101, sender=mode))

    assert harness.run(scenario, send=True) == {}
    assert harness.generated == harness.sends == harness.reads == []
    output = capsys.readouterr().out
    assert f"ordinary: {reason} task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("mode,reason", [
    ("timeout", "reply_parent_timeout"), ("error", "reply_parent_lookup_error"),
    ("none", "reply_parent_unavailable"), ("wrong_chat", "reply_parent_wrong_chat"),
])
def test_reply_only_missing_parent_is_observable_without_guessing_owner(
        tmp_path, monkeypatch, capsys, mode, reason):
    harness = Harness(tmp_path, monkeypatch)

    async def scenario():
        await harness.handler(Event(101, mention=False, reply=True, parent=mode))

    assert harness.run(scenario, send=True) == {}
    assert harness.generated == harness.sends == harness.reads == []
    output = capsys.readouterr().out
    assert f"ordinary: {reason} task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("parent_mode,trigger_parent_id,trigger_topic_id", [
    ("ok", 17, None),       # Topic root has no reply header of its own.
    ("wrong_thread", 18, 17),  # Exact parent is in another topic.
])
def test_same_chat_reply_parent_is_accepted_across_topic_headers(
        tmp_path, monkeypatch, capsys, parent_mode, trigger_parent_id,
        trigger_topic_id):
    harness = Harness(tmp_path, monkeypatch)

    async def scenario():
        event = Event(101, mention=False, reply=True, parent=parent_mode)
        event.message.reply_to_msg_id = trigger_parent_id
        event.message.reply_to = MessageReplyHeader(
            forum_topic=True,
            reply_to_msg_id=trigger_parent_id,
            reply_to_top_id=trigger_topic_id,
        )
        await harness.handler(event)
        await wait_until(lambda: harness.status(101) == "dry")

    assert harness.run(scenario) == {101: "dry"}
    assert len(harness.generated) == 1
    assert harness.generated[0]["trigger_reason"] == "reply"
    assert any(message["text"] == "synthetic owner parent"
               for message in harness.generated[0]["messages"])
    output = capsys.readouterr().out
    assert "reply_parent_wrong_chat" not in output
    assert "reply_parent_wrong_thread" not in output
    assert_private_output(output)


def test_verified_mention_can_proceed_when_optional_parent_is_unavailable(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    async def scenario():
        await harness.handler(Event(101, reply=True, parent="timeout"))
        await wait_until(lambda: harness.status(101) == "dry")

    assert harness.run(scenario) == {101: "dry"}
    assert len(harness.generated) == 1
    assert harness.generated[0]["trigger_reason"] == "mention"
    assert harness.sends == []
    output = capsys.readouterr().out
    assert "ordinary: reply_parent_timeout task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("reason", [
    "context_unclear", "needs_owner", "unsuitable_content",
    "reply_model_unavailable", "reply_invalid_schema", "reply_validation_failed",
    "reply_validation_failed_protected_slur", "reply_validation_failed_threat",
    "profile_or_context_unavailable", "context_too_large", "unknown_model_reason",
])
def test_model_decline_and_validation_failures_send_one_safe_fallback(
        tmp_path, monkeypatch, capsys, reason):
    harness = Harness(tmp_path, monkeypatch)

    async def generate(_window):
        return ReplyResult(None, reason)

    harness.generate = generate

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) in {"sent", "skip"})

    detailed_validation = (
        reason.startswith("reply_validation_failed_")
        and reason != "reply_validation_failed"
    )
    expected_status = "skip" if detailed_validation else "sent"
    assert harness.run(scenario, send=True) == {101: expected_status}
    if detailed_validation:
        assert harness.sends == []
    else:
        assert len(harness.sends) == 1
        expected_text, failure = runner.format_safe_fallback(
            runner._fallback_for_reason(reason)[1],
        )
        assert failure is None
        assert harness.sends[0][0][1] == expected_text
        assert len(harness.reads) == 1  # only the final owner-tail check under send lock
    output = capsys.readouterr().out
    if detailed_validation:
        assert f"ordinary: {reason} task=" in output
        assert "reply_fallback_" not in output
        assert_private_output(output)
        return
    expected_code = runner._fallback_for_reason(reason)[0]
    assert f"ordinary: reply_fallback_{expected_code} task=" in output
    assert "model_ms=" in output and "total_ms=" in output
    assert_private_output(output)


@pytest.mark.parametrize("failure,expected_reason", [
    ("exception", "reply_generation_error"),
    ("timeout", "reply_generation_timeout"),
])
def test_generation_exception_or_timeout_sends_fallback_once(
        tmp_path, monkeypatch, capsys, failure, expected_reason):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ORDINARY_GENERATION_TIMEOUT", 0.02)

    async def generate(_window):
        if failure == "timeout":
            await asyncio.Event().wait()
        raise RuntimeError("PRIVATE-MODEL-ERROR")

    harness.generate = generate

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "sent")

    assert harness.run(scenario, send=True) == {101: "sent"}
    assert len(harness.sends) == 1
    expected_text, failure = runner.format_safe_fallback(
        runner._fallback_for_reason(expected_reason)[1],
    )
    assert failure is None
    assert harness.sends[0][0][1] == expected_text
    output = capsys.readouterr().out
    assert f"ordinary: reply_fallback_{expected_reason} task=" in output
    assert_private_output(output)


def test_runner_skips_timed_out_repair_without_generic_fallback(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "generate_reply", reply_ai.generate_reply)
    monkeypatch.setattr(runner, "load_private_profile", lambda _path: {"synthetic": True})
    monkeypatch.setattr(reply_ai, "profile_for_api", lambda _profile: {})
    voice = {
        "schema_version": 1,
        "identity": {"aliases": [], "response": ""},
        "lexicon": {
            "frequent_words": [], "priority_words": [], "casual_words": [],
            "mild_expletives": [], "strong_words": [], "friendly_nicknames": [],
            "familiar_vocatives": [], "forbidden_vocatives": [],
            "friendly_phrases": [], "attack_words": [], "attack_phrases": [],
            "fillers": [],
        },
        "typography": {"replacements": [], "space_out": []},
    }
    monkeypatch.setattr(reply_ai, "load_voice_config", lambda: voice)
    monkeypatch.setattr(runner, "ORDINARY_GENERATION_TIMEOUT", 0.02)
    calls = []

    async def generate_once(_key, _model, payload):
        calls.append(payload)
        if len(calls) == 1:
            return OrdinaryReply(
                can_reply=True, text="https://example.invalid", reason="ok",
            )
        await asyncio.Event().wait()

    monkeypatch.setattr(reply_ai, "_generate_once", generate_once)

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "skip")

    assert harness.run(scenario, send=True) == {101: "skip"}
    assert len(calls) == 2
    assert harness.sends == []
    output = capsys.readouterr().out
    assert "ordinary: reply_validation_retry_timeout task=" in output
    assert "reply_fallback_" not in output
    assert_private_output(output)


def test_missing_style_profile_uses_fixed_fallback_without_leaking_trigger(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    def missing_profile(_path):
        raise OSError("PRIVATE-PROFILE-PATH")

    monkeypatch.setattr(runner, "load_private_profile", missing_profile)

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "sent")

    assert harness.run(scenario, send=True) == {101: "sent"}
    assert harness.generated == []
    assert len(harness.sends) == 1
    expected_text, failure = runner.format_safe_fallback(runner.ORDINARY_FALLBACK)
    assert failure is None
    assert harness.sends[0][0][1] == expected_text
    assert runner.ORDINARY_FALLBACK == "Не понял, о чём речь. Уточни?"
    assert_private_output(capsys.readouterr().out)


def test_pause_during_generation_blocks_fallback_send(tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    async def decline(_window):
        harness.state.set_paused(True)
        return ReplyResult(None, "reply_validation_failed")

    harness.generate = decline

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "skip")

    assert harness.run(scenario, send=True) == {101: "skip"}
    assert harness.sends == []
    assert "ordinary: paused task=" in capsys.readouterr().out


def test_duplicate_trigger_sends_fallback_only_once(tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    async def decline(_window):
        return ReplyResult(None, "reply_validation_failed")

    harness.generate = decline

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "sent")
        await harness.handler(Event(101))

    assert harness.run(scenario, send=True) == {101: "sent"}
    assert len(harness.sends) == 1
    assert "ordinary: duplicate_message task=" in capsys.readouterr().out


def test_validator_retry_refusal_does_not_send_generic_fallback(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    async def decline(_window):
        return ReplyResult(None, "reply_validation_retry_failed_command_link_or_contact")

    harness.generate = decline

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "skip")

    assert harness.run(scenario, send=True) == {101: "skip"}
    assert harness.sends == []
    output = capsys.readouterr().out
    assert "ordinary: reply_validation_retry_failed_command_link_or_contact task=" in output
    assert "reply_fallback_" not in output


def test_context_window_assembly_error_sends_safe_fallback(
        tmp_path, monkeypatch, capsys):
    class BrokenContext(ContextBuffer):
        def make_window(self, *_args, **_kwargs):
            raise RuntimeError("PRIVATE-CONTEXT-ERROR")

    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ContextBuffer", BrokenContext)

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "dry")

    assert harness.run(scenario) == {101: "dry"}
    assert harness.generated == []
    assert "ordinary: reply_fallback_context_unavailable task=" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize("mode", ["full", "evicted"])
def test_task_retains_sanitized_trigger_when_ram_buffer_cannot_keep_it(
        tmp_path, monkeypatch, capsys, mode):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ContextBuffer", lambda **_kwargs: ContextBuffer(
        max_groups=0 if mode == "full" else 100, max_messages_per_group=1,
    ))

    async def scenario():
        await harness.handler(Event(101))
        await harness.handler(Event(102, mention=False, text="ordinary filler"))
        await wait_until(lambda: harness.status(101) == "dry")

    assert harness.run(scenario) == {101: "dry"}
    window = harness.generated[0]
    assert window["messages"][window["trigger_index"]]["text"] == "synthetic trigger body"
    assert harness.sends == []
    output = capsys.readouterr().out
    assert ("ordinary: context_buffer_full task=" in output) == (mode == "full")
    assert_private_output(output)


def test_empty_trigger_window_sends_safe_clarification_instead_of_skipping(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)

    async def scenario():
        await harness.handler(Event(101, text="Ping"))
        await wait_until(lambda: harness.status(101) == "dry")

    assert harness.run(scenario) == {101: "dry"}
    assert harness.generated == []
    output = capsys.readouterr().out
    assert "ordinary: reply_fallback_context_unavailable task=" in output
    assert "context_private_data" not in output


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_failed_history_lookup_can_retry_for_a_later_trigger(
        tmp_path, monkeypatch, capsys, failure):
    harness = Harness(tmp_path, monkeypatch, reply_context_before=5)

    async def read(*_args, **_kwargs):
        if len(harness.reads) == 1:
            if failure == "timeout":
                await asyncio.Event().wait()
            raise RuntimeError("PRIVATE-HISTORY-ERROR")
        earlier = Event(50, mention=False, text="earlier synthetic context").message
        earlier.date = SimpleNamespace(timestamp=lambda: time.time() - 1)
        yield earlier

    harness.read = read

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "dry")
        await harness.handler(Event(102))
        await wait_until(lambda: harness.status(102) == "dry")

    assert harness.run(scenario) == {101: "dry", 102: "dry"}
    assert len(harness.reads) == 2
    assert len(harness.generated) == 2
    assert any(message["text"] == "earlier synthetic context"
               for message in harness.generated[1]["messages"])
    output = capsys.readouterr().out
    reason = "history_timeout" if failure == "timeout" else "history_unavailable"
    assert f"ordinary: {reason} task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("blocked_read", [1])
def test_owner_tail_timeout_fails_closed_and_releases_send_lock(
        tmp_path, monkeypatch, capsys, blocked_read):
    harness = Harness(tmp_path, monkeypatch)

    async def read(*_args, **_kwargs):
        if len(harness.reads) == blocked_read:
            await asyncio.Event().wait()
        if False:
            yield None

    harness.read = read

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "skip")
        assert harness.sends == []
        await harness.handler(Event(102))
        await wait_until(lambda: harness.status(102) == "sent")

    assert harness.run(scenario, send=True) == {101: "skip", 102: "sent"}
    assert len(harness.sends) == 1
    output = capsys.readouterr().out
    assert "ordinary: owner_tail_timeout task=" in output
    assert "ordinary: owner_tail_unavailable task=" in output
    assert_private_output(output)


def test_send_lock_contention_expires_without_sending_then_releases_capacity(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch, control_bot_token="synthetic")
    captured = {}
    install_handler = runner.install_control_handler

    class Control:
        def add_event_handler(self, _handler, _builder): pass
        async def start(self, **_kwargs): pass
        async def disconnect(self): pass

    def capture_lock(bot, state, owner_id, send_lock, **kwargs):
        captured["lock"] = send_lock
        return install_handler(bot, state, owner_id, send_lock, **kwargs)

    monkeypatch.setattr(runner, "create_control_client", lambda _cfg: Control())
    monkeypatch.setattr(runner, "install_control_handler", capture_lock)

    async def scenario():
        await captured["lock"].acquire()
        try:
            await harness.handler(Event(101))
            await wait_until(lambda: harness.status(101) == "skip")
            assert harness.sends == []
        finally:
            captured["lock"].release()
        await harness.handler(Event(102))
        await wait_until(lambda: harness.status(102) == "sent")

    assert harness.run(scenario, send=True) == {101: "skip", 102: "sent"}
    assert len(harness.sends) == 1
    output = capsys.readouterr().out
    assert "ordinary: send_lock_timeout task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("stage", ["history", "owner_tail"])
def test_lookup_uses_remaining_task_deadline(
        tmp_path, monkeypatch, capsys, stage):
    harness = Harness(tmp_path, monkeypatch,
                      reply_context_before=5 if stage == "history" else 0)
    monkeypatch.setattr(runner, "ORDINARY_LOOKUP_TIMEOUT", 1)
    monkeypatch.setattr(runner, "ORDINARY_DEADLINE_SECONDS", 0.025)

    async def read(*_args, **_kwargs):
        await asyncio.Event().wait()
        if False:
            yield None

    harness.read = read

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "skip")

    assert harness.run(scenario, send=stage == "owner_tail") == {101: "skip"}
    assert harness.sends == []
    output = capsys.readouterr().out
    assert f"ordinary: {stage}_timeout task=" in output
    assert "ordinary: deadline_expired task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("cancellation", ["deadline", "shutdown"])
def test_hanging_send_is_uncertain_without_resend(
        tmp_path, monkeypatch, capsys, cancellation):
    harness = Harness(tmp_path, monkeypatch)
    send_started = asyncio.Event()

    async def send(*_args, **_kwargs):
        send_started.set()
        await asyncio.Event().wait()

    harness.send = send

    async def scenario():
        await harness.handler(Event(101))
        await asyncio.wait_for(send_started.wait(), 1)
        if cancellation == "deadline":
            await wait_until(lambda: harness.status(101) == "uncertain")

    assert harness.run(scenario, send=True) == {101: "uncertain"}
    assert len(harness.sends) == 1
    with sqlite3.connect(tmp_path / "data" / "state.sqlite3") as db:
        assert db.execute("SELECT status FROM ordinary_quota_parts").fetchall() == [("uncertain",)]
    output = capsys.readouterr().out
    reason = ("telegram_send_timeout_uncertain" if cancellation == "deadline"
              else "send_cancelled_uncertain")
    assert f"ordinary: {reason} task=" in output
    assert_private_output(output)


def test_retained_trigger_cannot_cross_group_or_topic_boundary():
    snapshot = snapshot_from_message(Event(101).message, -100, 42)
    buffer = ContextBuffer()
    assert buffer.make_window(-999, 0, 101, 42, 0, 0,
                              trigger_snapshot=snapshot)["messages"] == []
    assert buffer.make_window(-100, 999, 101, 42, 0, 0,
                              trigger_snapshot=snapshot)["messages"] == []


def test_window_trimming_preserves_retained_trigger_under_message_cap():
    snapshot = snapshot_from_message(Event(101).message, -100, 42)
    buffer = ContextBuffer()
    for message_id in range(102, 122):
        buffer.add(snapshot_from_message(Event(message_id).message, -100, 42))
    window = buffer.make_window(-100, 0, 101, 42, 0, 20,
                                trigger_snapshot=snapshot)
    assert len(window["messages"]) == 10
    assert window["trigger_index"] == 0
    assert window["messages"][0]["text"] == snapshot.text


def test_outside_allowlist_does_not_lookup_sender_or_parent(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)

    class ForbiddenEvent(Event):
        async def get_sender(self): pytest.fail("No sender lookup outside allowlist")
        async def get_reply_message(self): pytest.fail("No parent lookup outside allowlist")

    async def scenario():
        await harness.handler(ForbiddenEvent(101, reply=True, chat_id=-999))

    assert harness.run(scenario, send=True) == {}
    assert harness.sends == harness.generated == harness.reads == []


@pytest.mark.parametrize("tail_count", [19, 20])
def test_excluded_own_part_still_counts_toward_owner_tail_scan_limit(
        tmp_path, monkeypatch, capsys, tail_count):
    harness = Harness(tmp_path, monkeypatch)

    async def generate(_window):
        return ReplyResult("Короткий ответ.", "ok", "Проверим конкретный пример.")

    async def read(*_args, **_kwargs):
        # The second locked scan protects part two and ends with our own
        # already-confirmed outbound message.
        if len(harness.reads) == 2:
            for message_id in range(201, 201 + tail_count - 1):
                yield Event(message_id).message
            own_part = Event(10001).message
            own_part.sender_id = 42
            yield own_part

    harness.generate = generate
    harness.read = read
    expected = "partial" if tail_count == 20 else "sent"

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == expected)
        # A partial job is recorded before the second part's final guard, too.
        if tail_count == 20:
            await wait_until(lambda: harness.state.ordinary_progress(-100, 101)["status"]
                             == "partial" and len(harness.reads) == 2)
            await asyncio.sleep(0)

    assert harness.run(scenario, send=True) == {101: expected}
    assert len(harness.sends) == (1 if tail_count == 20 else 2)
    output = capsys.readouterr().out
    assert ("ordinary: owner_tail_truncated task=" in output) == (tail_count == 20)
    assert ("ordinary: owner_tail_unavailable task=" in output) == (tail_count == 20)
    assert_private_output(output)


@pytest.mark.parametrize("lookup", ["sender", "parent"])
def test_pause_resume_during_intake_lookup_never_renews_trigger_generation(
        tmp_path, monkeypatch, capsys, lookup):
    harness = Harness(tmp_path, monkeypatch)
    lookup_started = asyncio.Event()
    release = asyncio.Event()

    class DelayedEvent(Event):
        async def get_sender(self):
            if lookup == "sender":
                lookup_started.set()
                await release.wait()
            return await super().get_sender()

        async def get_reply_message(self):
            lookup_started.set()
            await release.wait()
            return await super().get_reply_message()

    async def scenario():
        pending = asyncio.create_task(harness.handler(
            DelayedEvent(101, mention=lookup == "sender", reply=lookup == "parent"),
        ))
        await asyncio.wait_for(lookup_started.wait(), 1)
        harness.state.set_paused(True)
        harness.state.set_paused(False)
        release.set()
        await pending
        assert harness.status(101) == "skip"
        assert harness.generated == harness.sends == []
        await harness.handler(Event(102))
        await wait_until(lambda: harness.status(102) == "sent")

    assert harness.run(scenario, send=True) == {101: "skip", 102: "sent"}
    assert len(harness.generated) == len(harness.sends) == 1
    output = capsys.readouterr().out
    assert "ordinary: paused task=" in output
    assert_private_output(output)


@pytest.mark.parametrize("lookup", ["sender", "parent"])
def test_intake_lookup_timeout_cannot_exceed_whole_task_deadline(
        tmp_path, monkeypatch, capsys, lookup):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ORDINARY_LOOKUP_TIMEOUT", 1)
    monkeypatch.setattr(runner, "ORDINARY_DEADLINE_SECONDS", 0.025)

    async def scenario():
        await harness.handler(Event(
            101, mention=lookup == "sender", reply=lookup == "parent",
            sender="timeout" if lookup == "sender" else "ok",
            parent="timeout" if lookup == "parent" else "ok",
        ))

    assert harness.run(scenario, send=True) == {101: "skip"}
    assert harness.generated == harness.sends == harness.reads == []
    output = capsys.readouterr().out
    reason = "sender_lookup_timeout" if lookup == "sender" else "reply_parent_timeout"
    assert f"ordinary: {reason} task=" in output
    assert "ordinary: deadline_expired task=" in output
    assert_private_output(output)


def test_sender_and_parent_lookups_share_one_deadline(tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ORDINARY_LOOKUP_TIMEOUT", 1)
    monkeypatch.setattr(runner, "ORDINARY_DEADLINE_SECONDS", 0.035)

    class SlowEvent(Event):
        async def get_sender(self):
            await asyncio.sleep(0.02)
            return await super().get_sender()

        async def get_reply_message(self):
            await asyncio.sleep(0.02)
            return await super().get_reply_message()

    async def scenario():
        await harness.handler(SlowEvent(101, mention=False, reply=True))

    assert harness.run(scenario, send=True) == {101: "skip"}
    assert harness.generated == harness.sends == []
    output = capsys.readouterr().out
    assert "ordinary: reply_parent_timeout task=" in output
    assert "ordinary: deadline_expired task=" in output
    assert_private_output(output)


def test_worker_inherits_time_already_spent_on_successful_intake_lookups(
        tmp_path, monkeypatch, capsys):
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "ORDINARY_LOOKUP_TIMEOUT", 1)
    monkeypatch.setattr(runner, "ORDINARY_DEADLINE_SECONDS", 0.15)
    monkeypatch.setattr(runner, "ORDINARY_GENERATION_TIMEOUT", 0.02)

    class SlowEvent(Event):
        async def get_sender(self):
            await asyncio.sleep(0.025)
            return await super().get_sender()

        async def get_reply_message(self):
            await asyncio.sleep(0.025)
            return await super().get_reply_message()

    async def generate(_window):
        await asyncio.sleep(0.05)
        return ReplyResult("Короткий ответ.", "ok")

    harness.generate = generate

    async def scenario():
        await harness.handler(SlowEvent(101, reply=True))
        await wait_until(lambda: harness.status(101) == "sent")

    assert harness.run(scenario, send=True) == {101: "sent"}
    assert len(harness.generated) == 1
    assert len(harness.sends) == 1
    output = capsys.readouterr().out
    assert "ordinary: reply_fallback_reply_generation_timeout task=" in output
    assert_private_output(output)


def test_sticker_cooldown_is_persisted_and_can_be_released(tmp_path):
    state = State(tmp_path / "data" / "state.sqlite3")
    assert state.sticker_cooldown_available(-100, cooldown_seconds=900, now=1000)
    assert state.reserve_sticker_cooldown(-100, cooldown_seconds=900, now=1000)
    assert not state.sticker_cooldown_available(-100, cooldown_seconds=900, now=1899)
    assert state.sticker_cooldown_available(-100, cooldown_seconds=900, now=1900)
    assert state.reserve_sticker_cooldown(-100, cooldown_seconds=900, now=1900)
    state.release_sticker_cooldown(-100, reserved_at=1900.0)
    assert state.sticker_cooldown_available(-100, cooldown_seconds=900, now=1901)


class _FakeRecentStickerCatalog:
    def __init__(self):
        self.item = SimpleNamespace(emoji="😀", document=object())
        self.emojis = (self.item.emoji,)

    async def get(self, _client):
        return (self.item,)

    def find(self, emoji):
        return self.item if emoji == self.item.emoji else None


def test_runner_sends_sticker_as_durable_second_part_and_remembers_emoji(
        tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, auto_reply_stickers_enabled=True)
    catalog = _FakeRecentStickerCatalog()
    monkeypatch.setattr(runner, "RecentStickerCatalog", lambda: catalog)
    monkeypatch.setattr(runner.random, "random", lambda: 0.0)

    async def generate(_window):
        return ReplyResult("Synthetic primary", "ok", "Synthetic follow-up", "😀")

    harness.generate = generate

    async def scenario():
        await harness.handler(Event(101))
        await wait_until(lambda: harness.status(101) == "sent")

    assert harness.run(scenario, send=True) == {101: "sent"}
    assert len(harness.sends) == 1
    assert len(harness.file_sends) == 1
    assert harness.file_sends[0][0] == (-100, catalog.item.document)
    assert harness.file_sends[0][1]["reply_to"] == 101
    check_db = sqlite3.connect(tmp_path / "data" / "state.sqlite3")
    rows = check_db.execute(
        "SELECT part,status FROM ordinary_quota_parts "
        "WHERE chat=-100 AND msg=101 ORDER BY part",
    ).fetchall()
    assert rows == [(1, "sent"), (2, "sent")]
    saved = check_db.execute(
        "SELECT text FROM context_snapshots WHERE chat=-100 AND message=20001",
    ).fetchone()
    assert saved == ("[стикер 😀]",)
    cooldown = check_db.execute(
        "SELECT value FROM settings WHERE key='ordinary_sticker_cooldown:-100'",
    ).fetchone()
    assert cooldown is not None
    assert time.time() - float(cooldown[0]) < runner.STICKER_COOLDOWN_SECONDS
    check_db.close()


def test_dry_run_shows_selected_sticker_and_never_sends(tmp_path, monkeypatch):
    import io
    import sys

    class TTY(io.StringIO):
        def isatty(self):
            return True

    harness = Harness(tmp_path, monkeypatch, auto_reply_stickers_enabled=True)
    catalog = _FakeRecentStickerCatalog()
    monkeypatch.setattr(runner, "RecentStickerCatalog", lambda: catalog)
    monkeypatch.setattr(runner.random, "random", lambda: 0.0)
    monkeypatch.setattr(sys, "stdin", TTY())
    output = TTY()
    monkeypatch.setattr(sys, "stdout", output)

    async def generate(_window):
        return ReplyResult("Synthetic primary", "ok", "Synthetic follow-up", "😀")

    harness.generate = generate

    async def scenario():
        await harness.handler(Event(102))
        await wait_until(lambda: harness.status(102) == "dry")

    assert harness.run(scenario, send=False, show_drafts=True) == {102: "dry"}
    assert harness.sends == []
    assert harness.file_sends == []
    assert "Черновик стикера" in output.getvalue()
    assert "😀" in output.getvalue()
