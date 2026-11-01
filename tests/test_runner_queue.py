"""Synthetic queue regressions; no credentials, model API, or Telegram calls."""

import asyncio
import sqlite3
import time
from types import SimpleNamespace

import pytest

from tg_bot import runner
from tg_bot.reply_ai import ReplyResult
from tg_bot.state import State


class MessageEntityMentionName:
    offset = 0
    length = 4
    user_id = 42


class Event:
    chat_id = -100
    is_group = True
    chat = SimpleNamespace(broadcast=False)

    def __init__(self, message_id, *, topic=0, sender=9):
        self.message = SimpleNamespace(
            id=message_id, date=SimpleNamespace(
                timestamp=lambda: time.time() + message_id / 1_000_000,
            ),
            out=False, sender_id=sender, message="Ping synthetic question?",
            entities=[MessageEntityMentionName()], fwd_from=None,
            reply_to_msg_id=None, reply_to=None, message_thread_id=topic,
        )

    async def get_sender(self):
        # Intentionally no suspension: bursts can cancel a task before it starts.
        return SimpleNamespace(bot=False, id=self.message.sender_id)


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def exercise(tmp_path, monkeypatch, scenario, generate=None, control=None):
    states = []

    def make_state(path):
        state = State(path)
        states.append(state)
        return state

    class Telegram:
        handler = None
        sent = []

        async def connect(self): pass
        async def is_user_authorized(self): return True
        async def get_me(self): return SimpleNamespace(id=42, username="synthetic_owner")
        def add_event_handler(self, handler, _builder): self.handler = handler
        async def disconnect(self): pass

        async def send_message(self, *args, **kwargs):
            self.sent.append((args, kwargs))
            raise AssertionError("Dry-run must never send")

        async def run_until_disconnected(self):
            await scenario(self, states[0])

    async def reply(*_args, **_kwargs):
        return ReplyResult("Короткий ответ.", "ok")

    telegram = Telegram()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "State", make_state)
    monkeypatch.setattr(runner, "client", lambda _cfg: telegram)
    monkeypatch.setattr(runner, "load_private_profile", lambda _path: {})
    monkeypatch.setattr(runner, "load_private_participant_summaries",
                        lambda *_args: {"participants": {}})
    monkeypatch.setattr(runner, "generate_reply", generate or reply)
    if control is not None:
        monkeypatch.setattr(runner, "create_control_client", lambda _cfg: control)

    cfg = SimpleNamespace(
        chats={-100}, chats_error=False, api_id=1, api_hash="synthetic",
        gemini_key="synthetic", model="synthetic", reply_model="synthetic",
        auto_reply_enabled=True, reply_config_error="", style_profile_path="synthetic",
        owner_name_aliases=(), reply_context_before=0, reply_context_after=0,
        reply_wait_seconds=0, control_bot_token="synthetic" if control else "",
    )
    asyncio.run(runner.run(cfg, send=False))
    assert telegram.sent == []
    with sqlite3.connect(tmp_path / "data" / "state.sqlite3") as db:
        return dict(db.execute("SELECT msg,status FROM jobs"))


def job_status(state, message_id):
    row = state.db.execute("SELECT status FROM jobs WHERE msg=?", (message_id,)).fetchone()
    return row[0] if row else None


def test_burst_supersession_releases_capacity_and_keeps_latest_mapping(
        tmp_path, monkeypatch, capsys):
    async def scenario(telegram, state):
        # Same-sender triggers in the same thread supersede before model work starts.
        for message_id in range(1, 81):
            await telegram.handler(Event(message_id))
        await wait_until(lambda: job_status(state, 80) == "dry")

        # Cancellation callbacks must not remove a later task's thread mapping.
        for message_id in range(101, 151):
            await telegram.handler(Event(message_id, topic=message_id))
        await wait_until(lambda: all(job_status(state, value) == "dry"
                                     for value in range(101, 151)))

    statuses = exercise(tmp_path, monkeypatch, scenario)
    assert all(statuses[value] == "superseded" for value in range(1, 80))
    assert statuses[80] == "dry"
    output = capsys.readouterr().out
    assert "ordinary: superseded task=" in output
    assert "ordinary: cancelled_before_start task=" in output
    assert "pending_limit" not in output
    assert "synthetic question" not in output
    assert "message=" not in output


def test_pause_before_first_step_then_resume_restores_full_capacity(
        tmp_path, monkeypatch, capsys):
    class Control:
        handler = None
        responses = []

        def add_event_handler(self, handler, _builder): self.handler = handler
        async def start(self, **_kwargs): pass
        async def disconnect(self): pass

        async def command(self, command):
            async def respond(text):
                self.responses.append(text)
            await self.handler(SimpleNamespace(
                is_private=True, sender_id=42, raw_text=command, respond=respond,
            ))

    control = Control()

    async def scenario(telegram, state):
        for message_id in range(1, 51):
            await telegram.handler(Event(message_id, topic=message_id))
        await control.command("/pause")
        await control.command("/resume")
        # No yield yet: even cancellation callbacks have not run.
        for message_id in range(101, 151):
            await telegram.handler(Event(message_id, topic=message_id))
        await wait_until(lambda: all(job_status(state, value) == "dry"
                                     for value in range(101, 151)))

    statuses = exercise(tmp_path, monkeypatch, scenario, control=control)
    assert all(statuses[value] == "skip" for value in range(1, 51))
    output = capsys.readouterr().out
    assert "ordinary: cancelled_before_start task=" in output
    assert "pending_limit" not in output
    assert "synthetic question" not in output


def test_cancelled_predecessor_cleanup_keeps_new_context_wait_task(
        tmp_path, monkeypatch, capsys):
    collecting = asyncio.Event()
    release = asyncio.Event()
    original_context = runner.ContextBuffer

    class WaitingContext(original_context):
        async def wait_for_after(self, *_args):
            collecting.set()
            await release.wait()

    monkeypatch.setattr(runner, "ContextBuffer", WaitingContext)

    async def scenario(telegram, state):
        await telegram.handler(Event(1))
        await telegram.handler(Event(2))
        await asyncio.wait_for(collecting.wait(), 2)
        # Let the first task's done callback run while task two is collecting.
        await asyncio.sleep(0)
        await telegram.handler(Event(3))
        release.set()
        await wait_until(lambda: job_status(state, 3) == "dry")

    statuses = exercise(tmp_path, monkeypatch, scenario)
    assert statuses == {1: "superseded", 2: "superseded", 3: "dry"}
    assert "ordinary: superseded task=" in capsys.readouterr().out


@pytest.mark.parametrize("during_collection", [False, True])
@pytest.mark.parametrize("same_sender", [False, True])
def test_collecting_supersession_is_limited_to_same_sender(
        tmp_path, monkeypatch, capsys, during_collection, same_sender):
    collecting = asyncio.Event()
    release = asyncio.Event()
    generated = []
    original_context = runner.ContextBuffer

    class WaitingContext(original_context):
        async def wait_for_after(self, *_args):
            collecting.set()
            await release.wait()

    monkeypatch.setattr(runner, "ContextBuffer", WaitingContext)

    async def generate(*_args, **_kwargs):
        generated.append(True)
        return ReplyResult("Короткий ответ.", "ok")

    async def scenario(telegram, state):
        await telegram.handler(Event(1, sender=9))
        if during_collection:
            await asyncio.wait_for(collecting.wait(), 2)
        await telegram.handler(Event(2, sender=9 if same_sender else 10))
        release.set()
        await wait_until(lambda: job_status(state, 2) == "dry"
                         and job_status(state, 1) == (
                             "superseded" if same_sender else "dry"))

    statuses = exercise(tmp_path, monkeypatch, scenario, generate=generate)
    assert statuses == {1: "superseded" if same_sender else "dry", 2: "dry"}
    assert len(generated) == (1 if same_sender else 2)
    output = capsys.readouterr().out
    assert ("ordinary: superseded task=" in output) == same_sender
    assert "synthetic question" not in output


def test_new_author_during_generation_does_not_cancel_started_reply(
        tmp_path, monkeypatch, capsys):
    started = asyncio.Event()
    release = asyncio.Event()
    generated = []
    cancelled = []

    async def generate(*_args, **_kwargs):
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return ReplyResult("Короткий ответ.", "ok")

    async def scenario(telegram, state):
        await telegram.handler(Event(1, sender=9))
        await asyncio.wait_for(started.wait(), 2)
        await telegram.handler(Event(2, sender=10))
        release.set()
        await wait_until(lambda: job_status(state, 1) == job_status(state, 2) == "dry")

    statuses = exercise(tmp_path, monkeypatch, scenario, generate=generate)
    assert statuses == {1: "dry", 2: "dry"}
    assert generated == [1, 2]
    assert cancelled == []
    assert "superseded" not in capsys.readouterr().out


def test_more_than_fifty_triggers_are_queued_without_pending_skip(
        tmp_path, monkeypatch, capsys):
    started = asyncio.Event()
    release = asyncio.Event()

    async def generate(*_args, **_kwargs):
        started.set()
        await release.wait()
        return ReplyResult("Короткий ответ.", "ok")

    async def scenario(telegram, state):
        for message_id in range(1, 52):
            await telegram.handler(Event(message_id, topic=message_id))
        await asyncio.wait_for(started.wait(), 2)
        await wait_until(lambda: all(job_status(state, message_id) == "pending"
                                     for message_id in range(1, 52)))
        release.set()
        await wait_until(lambda: all(job_status(state, message_id) == "dry"
                                     for message_id in range(1, 52)))

    statuses = exercise(tmp_path, monkeypatch, scenario, generate=generate)
    assert len(statuses) == 51
    assert set(statuses.values()) == {"dry"}
    output = capsys.readouterr().out
    assert "ordinary: pending_limit task=" not in output
    assert "synthetic question" not in output
