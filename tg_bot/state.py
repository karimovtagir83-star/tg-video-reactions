import math
import os
import sqlite3
import time
from pathlib import Path


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        previous_umask = os.umask(0o077)
        try:
            self.db = sqlite3.connect(path, timeout=2)
            self.db.execute("PRAGMA journal_mode=WAL")
        finally:
            os.umask(previous_umask)
        self.path = Path(path)
        self._secure_files()
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(
                chat INTEGER, msg INTEGER, video TEXT, status TEXT, ts INTEGER,
                PRIMARY KEY(chat,msg));
            CREATE TABLE IF NOT EXISTS settings(
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS owner_actions(
                chat INTEGER NOT NULL,
                command INTEGER NOT NULL,
                target INTEGER,
                target_sender INTEGER,
                thread INTEGER NOT NULL,
                status TEXT NOT NULL,
                outbound INTEGER,
                updated REAL NOT NULL,
                PRIMARY KEY(chat,command));
            CREATE TABLE IF NOT EXISTS context_snapshots(
                chat INTEGER NOT NULL,
                message INTEGER NOT NULL,
                sender INTEGER,
                own INTEGER NOT NULL,
                thread INTEGER NOT NULL,
                reply_to INTEGER,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY(chat,message));
            CREATE INDEX IF NOT EXISTS context_snapshots_thread
                ON context_snapshots(chat,thread,ts,message);
            INSERT OR IGNORE INTO settings(key, value) VALUES ('paused', '0');
            INSERT OR IGNORE INTO settings(key, value) VALUES ('generation', '0');
        ''')
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}
        if "kind" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'video'")
        # Keep the historical table names so existing state databases remain
        # readable. These rows track reply/send state, not quotas or cooldowns.
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS ordinary_quota(
                chat INTEGER NOT NULL, msg INTEGER NOT NULL, ts INTEGER NOT NULL,
                mode TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(chat,msg));
        ''')
        quota_columns = {row[1] for row in self.db.execute("PRAGMA table_info(ordinary_quota)")}
        if "active_part" not in quota_columns:
            self.db.execute("ALTER TABLE ordinary_quota ADD COLUMN active_part INTEGER NOT NULL DEFAULT 0")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS ordinary_quota_parts(
                chat INTEGER NOT NULL, msg INTEGER NOT NULL, part INTEGER NOT NULL,
                ts REAL NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(chat,msg,part));
            CREATE INDEX IF NOT EXISTS ordinary_quota_parts_window
                ON ordinary_quota_parts(mode,status,chat,ts);
        ''')
        # Upgrade legacy one-message rows without opening or exposing any user data.
        self.db.execute('''
            INSERT OR IGNORE INTO ordinary_quota_parts(chat,msg,part,ts,mode,status)
            SELECT chat,msg,1,ts,mode,
                CASE status
                    WHEN 'sent' THEN 'sent'
                    WHEN 'uncertain' THEN 'uncertain'
                    WHEN 'sending' THEN 'uncertain'
                    WHEN 'reserved' THEN 'reserved'
                    WHEN 'dry' THEN 'dry'
                    ELSE 'released'
                END
            FROM ordinary_quota
        ''')
        # A process restart during a Telegram request is ambiguous. Preserve the
        # send and dedupe state, and never retry that physical message.
        self.db.execute("UPDATE ordinary_quota_parts SET status='uncertain' WHERE status='sending'")
        self.db.execute("""
            UPDATE ordinary_quota SET status='uncertain',active_part=0
            WHERE status='sending'
        """)
        self.db.execute("""
            UPDATE ordinary_quota_parts SET status='released'
            WHERE status='reserved' AND EXISTS (
                SELECT 1 FROM ordinary_quota q
                WHERE q.chat=ordinary_quota_parts.chat AND q.msg=ordinary_quota_parts.msg
                  AND q.status='uncertain'
            )
        """)
        # A process may stop after part one was confirmed but before the
        # follow-up was attempted. Keep the confirmed part counted, and release
        # the unattempted sibling so a restart can never send it later.
        self.db.execute("""
            UPDATE ordinary_quota_parts SET status='released'
            WHERE status='reserved' AND EXISTS (
                SELECT 1 FROM ordinary_quota q
                WHERE q.chat=ordinary_quota_parts.chat AND q.msg=ordinary_quota_parts.msg
                  AND q.status='partial'
            )
        """)
        # Reservations that never reached a send boundary are not live work
        # after restart. Release their rows but keep jobs as dedupe keys.
        self.db.execute(
            "UPDATE ordinary_quota_parts SET status='released' WHERE status='reserved'"
        )
        self.db.execute("""
            UPDATE ordinary_quota SET status='released',active_part=0
            WHERE status='reserved'
        """)
        self.db.execute("""
            UPDATE jobs SET status='uncertain'
            WHERE kind='ordinary' AND status='sending'
        """)
        self.db.commit()
        self._secure_files()

    def _secure_files(self):
        """Keep the SQLite database and its transient WAL files owner-only."""
        for candidate in (
            self.path,
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
        ):
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                pass

    def prune_context_snapshots(self, allowed_chats, *, enabled: bool,
                                now=None, ttl_seconds=24 * 60 * 60):
        """Drop expired snapshots and all rows outside the active group allowlist."""
        now = time.time() if now is None else float(now)
        allowed = {int(chat) for chat in allowed_chats} if enabled else set()
        cutoff = now - int(ttl_seconds)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "DELETE FROM context_snapshots WHERE ts<? OR ts>?",
                (cutoff, now + 300),
            )
            if allowed:
                present = self.db.execute(
                    "SELECT DISTINCT chat FROM context_snapshots",
                ).fetchall()
                for (chat,) in present:
                    if int(chat) not in allowed:
                        self.db.execute(
                            "DELETE FROM context_snapshots WHERE chat=?", (chat,),
                        )
            else:
                self.db.execute("DELETE FROM context_snapshots")
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self._secure_files()

    def save_context_snapshot(self, snapshot, allowed_chats, *, now=None,
                              ttl_seconds=24 * 60 * 60, max_messages=50):
        """Persist one already-sanitized snapshot for an allowlisted group."""
        allowed = {int(chat) for chat in allowed_chats}
        try:
            chat = int(snapshot.chat_id)
            message = int(snapshot.message_id)
            sender = int(snapshot.sender_id) if snapshot.sender_id is not None else None
            own = bool(snapshot.own)
            thread = int(snapshot.thread_id)
            reply_to = int(snapshot.reply_to_id) if snapshot.reply_to_id is not None else None
            ts = float(snapshot.timestamp)
            from .context import has_unredacted_private_data, sanitize_text
            text = sanitize_text(snapshot.text)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        now = time.time() if now is None else float(now)
        if (chat not in allowed or not text or len(text) > 800
                or not math.isfinite(ts) or not math.isfinite(now)
                or ts < now - int(ttl_seconds) or ts > now + 300
                or has_unredacted_private_data(text)):
            return False
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("DELETE FROM context_snapshots WHERE ts<?", (now - int(ttl_seconds),))
            self.db.execute(
                "INSERT OR REPLACE INTO context_snapshots"
                "(chat,message,sender,own,thread,reply_to,ts,text) VALUES(?,?,?,?,?,?,?,?)",
                (chat, message, sender, int(own), thread, reply_to, ts, text),
            )
            self.db.execute(
                "DELETE FROM context_snapshots WHERE chat=? AND message NOT IN ("
                "SELECT message FROM context_snapshots WHERE chat=? "
                "ORDER BY ts DESC,message DESC LIMIT ?)",
                (chat, chat, max(1, int(max_messages))),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self._secure_files()
        return True

    def load_context_snapshots(self, allowed_chats, *, now=None,
                               ttl_seconds=24 * 60 * 60, max_messages=50):
        """Read only recent snapshots for explicitly allowed group ids."""
        now = time.time() if now is None else float(now)
        cutoff = now - int(ttl_seconds)
        rows = []
        for chat in sorted({int(value) for value in allowed_chats}):
            rows.extend(self.db.execute(
                "SELECT chat,message,sender,own,thread,reply_to,ts,text "
                "FROM context_snapshots WHERE chat=? AND ts>=? AND ts<=? "
                "ORDER BY ts DESC,message DESC LIMIT ?",
                (chat, cutoff, now + 300, max(1, int(max_messages))),
            ).fetchall())
        return sorted(rows, key=lambda row: (row[0], row[6], row[1]))

    def pause_snapshot(self):
        """Return (paused, generation); the generation invalidates queued work."""
        values = dict(self.db.execute(
            "SELECT key, value FROM settings WHERE key IN ('paused', 'generation')"
        ))
        return values.get("paused", "0") == "1", int(values.get("generation", "0"))

    def set_paused(self, paused):
        """Persist pause state and advance generation only when it changes."""
        paused = bool(paused)
        current, generation = self.pause_snapshot()
        if current != paused:
            generation += 1
            self.db.execute("UPDATE settings SET value=? WHERE key='paused'", ("1" if paused else "0",))
            self.db.execute("UPDATE settings SET value=? WHERE key='generation'", (str(generation),))
            self.db.commit()
        return paused, generation

    def ordinary_mode_snapshot(self, default_enabled=False):
        """Return the persisted ordinary-reply mode and its independent generation."""
        values = dict(self.db.execute(
            "SELECT key, value FROM settings "
            "WHERE key IN ('ordinary_enabled', 'ordinary_generation')"
        ))
        if "ordinary_enabled" not in values:
            self.db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('ordinary_enabled', ?)",
                ("1" if default_enabled else "0",),
            )
        if "ordinary_generation" not in values:
            self.db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('ordinary_generation', '0')"
            )
        if "ordinary_enabled" not in values or "ordinary_generation" not in values:
            self.db.commit()
            values = dict(self.db.execute(
                "SELECT key, value FROM settings "
                "WHERE key IN ('ordinary_enabled', 'ordinary_generation')"
            ))
        return values.get("ordinary_enabled", "0") == "1", int(
            values.get("ordinary_generation", "0"),
        )

    def set_ordinary_enabled(self, enabled, *, default_enabled=False):
        """Persist the ordinary-reply mode independently from the global pause."""
        enabled = bool(enabled)
        current, generation = self.ordinary_mode_snapshot(default_enabled)
        if current != enabled:
            generation += 1
            self.db.execute(
                "UPDATE settings SET value=? WHERE key='ordinary_enabled'",
                ("1" if enabled else "0",),
            )
            self.db.execute(
                "UPDATE settings SET value=? WHERE key='ordinary_generation'",
                (str(generation),),
            )
            self.db.commit()
        return enabled, generation

    def link_mode_snapshot(self, default_enabled=True):
        """Return the independently persisted video-link mode and generation."""
        values = dict(self.db.execute(
            "SELECT key, value FROM settings "
            "WHERE key IN ('links_enabled', 'links_generation')"
        ))
        if "links_enabled" not in values:
            self.db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('links_enabled', ?)",
                ("1" if default_enabled else "0",),
            )
        if "links_generation" not in values:
            self.db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('links_generation', '0')"
            )
        if "links_enabled" not in values or "links_generation" not in values:
            self.db.commit()
            values = dict(self.db.execute(
                "SELECT key, value FROM settings "
                "WHERE key IN ('links_enabled', 'links_generation')"
            ))
        return values.get("links_enabled", "1") == "1", int(
            values.get("links_generation", "0"),
        )

    def set_links_enabled(self, enabled, *, default_enabled=True):
        """Persist video-link replies independently from ordinary replies and pause."""
        enabled = bool(enabled)
        current, generation = self.link_mode_snapshot(default_enabled)
        if current != enabled:
            generation += 1
            self.db.execute(
                "UPDATE settings SET value=? WHERE key='links_enabled'",
                ("1" if enabled else "0",),
            )
            self.db.execute(
                "UPDATE settings SET value=? WHERE key='links_generation'",
                (str(generation),),
            )
            self.db.commit()
        return enabled, generation

    def sticker_cooldown_available(self, chat, *, cooldown_seconds=900, now=None):
        """Check a local per-group sticker cooldown stored in the existing settings table."""
        chat = int(chat)
        now = time.time() if now is None else float(now)
        row = self.db.execute(
            "SELECT value FROM settings WHERE key=?",
            (f"ordinary_sticker_cooldown:{chat}",),
        ).fetchone()
        if row is None:
            return True
        try:
            last = float(row[0])
        except (TypeError, ValueError):
            return True
        return now - last >= max(0, float(cooldown_seconds))

    def _setting_value(self, key, default=None):
        row = self.db.execute(
            "SELECT value FROM settings WHERE key=?", (key,),
        ).fetchone()
        return default if row is None else row[0]

    def last_sticker_sent(self, chat):
        """Return the last sticker's private local key and consecutive-send count."""
        key = f"ordinary_sticker_last_document:{int(chat)}"
        sticker_id = self._setting_value(key)
        try:
            count = int(self._setting_value(f"ordinary_sticker_same_run:{int(chat)}", "0"))
        except (TypeError, ValueError):
            count = 0
        return sticker_id, max(0, count)

    def sticker_send_allowed(
        self, chat, document_key, *, min_reply_gap=2, max_same_streak=2,
        reply_offset=0,
    ):
        """Enforce sticker spacing and repeat limits across service restarts."""
        chat = int(chat)
        if not isinstance(document_key, str) or not document_key:
            return False, "sticker_identity_unavailable"
        try:
            reply_seq = int(self._setting_value(
                f"ordinary_sticker_reply_seq:{chat}", "0",
            )) + max(0, int(reply_offset))
            last_seq = int(self._setting_value(
                f"ordinary_sticker_last_reply_seq:{chat}", "0",
            ))
        except (TypeError, ValueError):
            reply_seq, last_seq = 0, 0
        last_id, same_run = self.last_sticker_sent(chat)
        if last_id is not None and reply_seq - last_seq < max(1, int(min_reply_gap)):
            return False, "sticker_reply_spacing"
        if last_id == document_key and same_run >= max(1, int(max_same_streak)):
            return False, "sticker_repeat_streak"
        return True, "ok"

    def record_sticker_sent(self, chat, document_key):
        """Persist a confirmed sticker send and its consecutive-sticker streak."""
        chat = int(chat)
        if not isinstance(document_key, str) or not document_key:
            return False
        self.db.execute("BEGIN IMMEDIATE")
        try:
            last_id_key = f"ordinary_sticker_last_document:{chat}"
            run_key = f"ordinary_sticker_same_run:{chat}"
            seq_key = f"ordinary_sticker_reply_seq:{chat}"
            last_seq_key = f"ordinary_sticker_last_reply_seq:{chat}"
            last_id = self._setting_value(last_id_key)
            try:
                previous_run = int(self._setting_value(run_key, "0"))
                reply_seq = int(self._setting_value(seq_key, "0"))
            except (TypeError, ValueError):
                previous_run, reply_seq = 0, 0
            same_run = previous_run + 1 if last_id == document_key else 1
            values = (
                (last_id_key, document_key),
                (run_key, str(same_run)),
                (last_seq_key, str(reply_seq)),
            )
            self.db.executemany(
                "INSERT INTO settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                values,
            )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def skip_ordinary_followup(self, chat, msg, *, now=None):
        """Complete a text reply after its optional sticker part is frequency-limited."""
        now = time.time() if now is None else float(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            reservation = self.db.execute(
                "SELECT mode,status,active_part FROM ordinary_quota WHERE chat=? AND msg=?",
                (chat, msg),
            ).fetchone()
            first = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=1",
                (chat, msg),
            ).fetchone()
            second = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=2",
                (chat, msg),
            ).fetchone()
            if (reservation is None or reservation[0] != "live"
                    or reservation[1] not in {"reserved", "partial"}
                    or reservation[2] != 0 or first is None or first[0] != "sent"
                    or second is None or second[0] != "reserved"):
                self.db.rollback()
                return False
            self.db.execute(
                "UPDATE ordinary_quota_parts SET status='released',ts=? "
                "WHERE chat=? AND msg=? AND part=2 AND status='reserved'",
                (now, chat, msg),
            )
            self.db.execute(
                "UPDATE ordinary_quota SET status='sent',active_part=0,ts=? "
                "WHERE chat=? AND msg=?",
                (now, chat, msg),
            )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def reserve_sticker_cooldown(self, chat, *, cooldown_seconds=900, now=None):
        """Atomically claim the per-group sticker cooldown before physical send."""
        chat = int(chat)
        now = time.time() if now is None else float(now)
        key = f"ordinary_sticker_cooldown:{chat}"
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT value FROM settings WHERE key=?", (key,),
            ).fetchone()
            if row is not None:
                try:
                    last = float(row[0])
                except (TypeError, ValueError):
                    last = float("-inf")
                if now - last < max(0, float(cooldown_seconds)):
                    self.db.rollback()
                    return False
            self.db.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(now)),
            )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def release_sticker_cooldown(self, chat, *, reserved_at):
        """Undo a reservation only when the same attempt still owns it."""
        key = f"ordinary_sticker_cooldown:{int(chat)}"
        self.db.execute(
            "DELETE FROM settings WHERE key=? AND value=?",
            (key, str(float(reserved_at))),
        )
        self.db.commit()

    def seen(self, chat, msg):
        return self.db.execute("SELECT 1 FROM jobs WHERE chat=? AND msg=?", (chat, msg)).fetchone() is not None

    def claim(self, chat, msg, video, status="dry", now=None, kind="video"):
        now = now or int(time.time())
        self.db.execute("DELETE FROM jobs WHERE ts<?", (now - 7 * 86400,))
        if self.seen(chat, msg):
            return False
        self.db.execute(
            "INSERT INTO jobs(chat,msg,video,status,ts,kind) VALUES(?,?,?,?,?,?)",
            (chat, msg, video, status, now, kind),
        )
        self.db.commit()
        return True

    def record_owner_action(self, chat, command, *, target=None,
                            target_sender=None, thread=0, status="resolving",
                            now=None):
        """Persist only local Telegram IDs and outcome for an owner command."""
        now = time.time() if now is None else float(now)
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO owner_actions"
            "(chat,command,target,target_sender,thread,status,outbound,updated)"
            " VALUES(?,?,?,?,?,?,NULL,?)",
            (int(chat), int(command),
             int(target) if target is not None else None,
             int(target_sender) if target_sender is not None else None,
             int(thread), str(status), now),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def resolve_owner_action(self, chat, command, *, target, target_sender,
                             thread, now=None):
        now = time.time() if now is None else float(now)
        self.db.execute(
            "UPDATE owner_actions SET target=?,target_sender=?,thread=?,"
            "status='pending',updated=? WHERE chat=? AND command=?",
            (int(target), int(target_sender), int(thread), now,
             int(chat), int(command)),
        )
        self.db.commit()

    def finish_owner_action(self, chat, command, status, *, outbound=None, now=None):
        """Record the final action outcome without retaining message text."""
        allowed = {
            "resolving", "ambiguous", "pending", "sending", "sent", "dry",
            "cancelled", "uncertain", "skip", "clarification_sent",
        }
        if status not in allowed:
            raise ValueError("owner_action_status_invalid")
        now = time.time() if now is None else float(now)
        self.db.execute(
            "UPDATE owner_actions SET status=?,outbound=?,updated=?"
            " WHERE chat=? AND command=?",
            (status, int(outbound) if outbound is not None else None, now,
             int(chat), int(command)),
        )
        self.db.commit()

    def status(self, chat, msg, status, now=None):
        ts = int(now or time.time()) if status == "sent" else None
        if ts is None:
            self.db.execute("UPDATE jobs SET status=? WHERE chat=? AND msg=?", (status, chat, msg))
        else:
            self.db.execute("UPDATE jobs SET status=?,ts=? WHERE chat=? AND msg=?", (status, ts, chat, msg))
        self.db.commit()

    def reserve_ordinary_reply(self, chat, msg, *, dry_run: bool, now=None):
        """Atomically reserve ordinary-reply state before model generation."""
        now = time.time() if now is None else float(now)
        mode = "dry" if dry_run else "live"
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("DELETE FROM ordinary_quota_parts WHERE ts<?", (now - 8 * 86400,))
            self.db.execute("DELETE FROM ordinary_quota WHERE ts<?", (now - 8 * 86400,))
            if self.db.execute(
                "SELECT 1 FROM ordinary_quota WHERE chat=? AND msg=?", (chat, msg)
            ).fetchone():
                self.db.rollback()
                return False, "duplicate_message"
            self.db.execute(
                "INSERT INTO ordinary_quota(chat,msg,ts,mode,status) VALUES(?,?,?,?,?)",
                (chat, msg, now, mode, "reserved"),
            )
            self.db.execute(
                "INSERT INTO ordinary_quota_parts(chat,msg,part,ts,mode,status) "
                "VALUES(?,?,?,?,?,'reserved')",
                (chat, msg, 1, now, mode),
            )
            self.db.commit()
            return True, "ok"
        except Exception:
            self.db.rollback()
            raise

    def reserve_ordinary_followup(self, chat, msg, *, dry_run: bool, now=None):
        """Atomically reserve one optional second physical Telegram message."""
        now = time.time() if now is None else float(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            reservation = self.db.execute(
                "SELECT mode,status FROM ordinary_quota WHERE chat=? AND msg=?",
                (chat, msg),
            ).fetchone()
            mode = "dry" if dry_run else "live"
            if reservation is None or reservation != (mode, "reserved"):
                self.db.rollback()
                return False, "ordinary_reservation_unavailable"
            if self.db.execute(
                "SELECT 1 FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=2",
                (chat, msg),
            ).fetchone():
                self.db.rollback()
                return False, "duplicate_follow_up"
            first = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=1",
                (chat, msg),
            ).fetchone()
            if first is None or first[0] != "reserved":
                self.db.rollback()
                return False, "ordinary_reservation_unavailable"
            self.db.execute(
                "INSERT INTO ordinary_quota_parts(chat,msg,part,ts,mode,status) "
                "VALUES(?,?,2,?,?,'reserved')",
                (chat, msg, now, mode),
            )
            self.db.commit()
            return True, "ok"
        except Exception:
            self.db.rollback()
            raise

    def start_ordinary_part(self, chat, msg, part, *, now=None):
        """Mark one reserved reply part in-flight at the send boundary."""
        if part not in {1, 2}:
            return False, "ordinary_part_invalid"
        now = time.time() if now is None else float(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            reservation = self.db.execute(
                "SELECT mode,status,active_part FROM ordinary_quota WHERE chat=? AND msg=?",
                (chat, msg),
            ).fetchone()
            part_row = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=?",
                (chat, msg, part),
            ).fetchone()
            if (reservation is None or reservation[0] != "live"
                    or reservation[2] != 0 or part_row is None or part_row[0] != "reserved"):
                self.db.rollback()
                return False, "ordinary_reservation_unavailable"
            if part == 2:
                first = self.db.execute(
                    "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=1",
                    (chat, msg),
                ).fetchone()
                if first is None or first[0] != "sent":
                    self.db.rollback()
                    return False, "first_part_not_confirmed"
            self.db.execute(
                "UPDATE ordinary_quota_parts SET status='sending',ts=? "
                "WHERE chat=? AND msg=? AND part=?",
                (now, chat, msg, part),
            )
            self.db.execute(
                "UPDATE ordinary_quota SET status='sending',active_part=?,ts=? WHERE chat=? AND msg=?",
                (part, now, chat, msg),
            )
            self.db.commit()
            return True, "ok"
        except Exception:
            self.db.rollback()
            raise

    def start_ordinary_send(self, chat, msg, *, now=None):
        """Compatibility wrapper for the required first message."""
        return self.start_ordinary_part(chat, msg, 1, now=now)

    def defer_ordinary_part(self, chat, msg, part, *, now=None):
        """Return a FloodWait part to reserved state so Telegram can be retried."""
        now = time.time() if now is None else float(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE ordinary_quota_parts SET status='reserved',ts=? "
                "WHERE chat=? AND msg=? AND part=? AND status='sending'",
                (now, chat, msg, part),
            )
            sent = self.db.execute(
                "SELECT 1 FROM ordinary_quota_parts WHERE chat=? AND msg=? AND status='sent' LIMIT 1",
                (chat, msg),
            ).fetchone()
            self.db.execute(
                "UPDATE ordinary_quota SET status=?,active_part=0,ts=? WHERE chat=? AND msg=?",
                ("partial" if sent else "reserved", now, chat, msg),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def defer_ordinary_send(self, chat, msg, *, now=None):
        """Compatibility wrapper for a first-part FloodWait."""
        self.defer_ordinary_part(chat, msg, 1, now=now)

    def finish_ordinary_part(self, chat, msg, part, status, *, now=None):
        """Persist one physical send result and return the overall dedupe state."""
        if status not in {"sent", "uncertain", "released"} or part not in {1, 2}:
            raise ValueError("ordinary_part_status_invalid")
        now = time.time() if now is None else float(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            active = self.db.execute(
                "SELECT active_part FROM ordinary_quota WHERE chat=? AND msg=?",
                (chat, msg),
            ).fetchone()
            part_row = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=?",
                (chat, msg, part),
            ).fetchone()
            if active is None or part_row is None or active[0] != part or part_row[0] != "sending":
                self.db.rollback()
                raise ValueError("ordinary_part_not_active")
            self.db.execute(
                "UPDATE ordinary_quota_parts SET status=?,ts=? WHERE chat=? AND msg=? AND part=?",
                (status, now, chat, msg, part),
            )
            if part == 1 and status in {"sent", "uncertain"}:
                seq_key = f"ordinary_sticker_reply_seq:{int(chat)}"
                try:
                    reply_seq = int(self._setting_value(seq_key, "0")) + 1
                except (TypeError, ValueError):
                    reply_seq = 1
                self.db.execute(
                    "INSERT INTO settings(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (seq_key, str(reply_seq)),
                )
            if status in {"uncertain", "released"}:
                self.db.execute(
                    "UPDATE ordinary_quota_parts SET status='released',ts=? "
                    "WHERE chat=? AND msg=? AND status='reserved'",
                    (now, chat, msg),
                )
            sent = self.db.execute(
                "SELECT count(*) FROM ordinary_quota_parts WHERE chat=? AND msg=? AND status='sent'",
                (chat, msg),
            ).fetchone()[0]
            uncertain = self.db.execute(
                "SELECT count(*) FROM ordinary_quota_parts WHERE chat=? AND msg=? AND status='uncertain'",
                (chat, msg),
            ).fetchone()[0]
            pending = self.db.execute(
                "SELECT 1 FROM ordinary_quota_parts WHERE chat=? AND msg=? "
                "AND status IN ('reserved','sending') LIMIT 1",
                (chat, msg),
            ).fetchone()
            if status == "uncertain" or uncertain:
                overall = "uncertain"
            elif pending:
                overall = "partial"
            elif sent:
                overall = "sent" if status == "sent" else "partial"
            else:
                overall = "released"
            self.db.execute(
                "UPDATE ordinary_quota SET status=?,active_part=0,ts=? WHERE chat=? AND msg=?",
                (overall, now, chat, msg),
            )
            self.db.commit()
            return overall
        except Exception:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def finish_ordinary_reply(self, chat, msg, status, *, now=None):
        if status not in {"sent", "dry", "released", "uncertain"}:
            raise ValueError("ordinary_reply_status_invalid")
        now = time.time() if now is None else float(now)
        row = self.db.execute(
            "SELECT active_part FROM ordinary_quota WHERE chat=? AND msg=?",
            (chat, msg),
        ).fetchone()
        if status in {"sent", "uncertain"}:
            part = row[0] if row and row[0] else 1
            part_row = self.db.execute(
                "SELECT status FROM ordinary_quota_parts WHERE chat=? AND msg=? AND part=?",
                (chat, msg, part),
            ).fetchone()
            if part_row and part_row[0] == "reserved":
                # Kept for callers upgrading from the one-message interface.
                self.db.execute(
                    "UPDATE ordinary_quota_parts SET status='sending',ts=? WHERE chat=? AND msg=? AND part=?",
                    (now, chat, msg, part),
                )
                self.db.execute(
                    "UPDATE ordinary_quota SET active_part=? WHERE chat=? AND msg=?",
                    (part, chat, msg),
                )
                self.db.commit()
            self.finish_ordinary_part(chat, msg, part, status, now=now)
            return
        if status == "dry":
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute(
                    "UPDATE ordinary_quota_parts SET status='dry',ts=? "
                    "WHERE chat=? AND msg=? AND mode='dry' AND status IN ('reserved','sending')",
                    (now, chat, msg),
                )
                self.db.execute(
                    "UPDATE ordinary_quota SET status='dry',active_part=0,ts=? WHERE chat=? AND msg=?",
                    (now, chat, msg),
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE ordinary_quota_parts SET status='released',ts=? "
                "WHERE chat=? AND msg=? AND status='reserved'",
                (now, chat, msg),
            )
            sent = self.db.execute(
                "SELECT 1 FROM ordinary_quota_parts WHERE chat=? AND msg=? AND status='sent' LIMIT 1",
                (chat, msg),
            ).fetchone()
            uncertain = self.db.execute(
                "SELECT 1 FROM ordinary_quota_parts WHERE chat=? AND msg=? AND status='uncertain' LIMIT 1",
                (chat, msg),
            ).fetchone()
            overall = "uncertain" if uncertain else "partial" if sent else "released"
            self.db.execute(
                "UPDATE ordinary_quota SET status=?,active_part=0,ts=? WHERE chat=? AND msg=?",
                (overall, now, chat, msg),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def ordinary_counts(self, now=None):
        now = time.time() if now is None else float(now)
        row = self.db.execute(
            "SELECT count(*) FROM ordinary_quota_parts WHERE mode='live' AND status='sent' AND ts>=?",
            (now - 86400,),
        ).fetchone()
        return {"sent_last_24h": int(row[0] or 0)}

    def ordinary_progress(self, chat, msg):
        row = self.db.execute(
            "SELECT status,active_part FROM ordinary_quota WHERE chat=? AND msg=?",
            (chat, msg),
        ).fetchone()
        counts = dict(self.db.execute(
            "SELECT status,count(*) FROM ordinary_quota_parts WHERE chat=? AND msg=? GROUP BY status",
            (chat, msg),
        ))
        return {
            "status": row[0] if row else "unavailable",
            "active_part": int(row[1] or 0) if row else 0,
            "sent_parts": int(counts.get("sent", 0)),
            "uncertain_parts": int(counts.get("uncertain", 0)),
            "reserved_parts": int(counts.get("reserved", 0)),
            "sending_parts": int(counts.get("sending", 0)),
        }

    def close(self):
        self._secure_files()
        self.db.close()
        self._secure_files()
