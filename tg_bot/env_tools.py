"""Local, atomic edits to one explicitly selected .env setting."""

import os
import re
import tempfile
from pathlib import Path


_TOKEN_LINE = re.compile(r"^\s*(?:export\s+)?CONTROL_BOT_TOKEN\s*=")
_BOT_TOKEN = re.compile(r"[0-9]+:[A-Za-z0-9_-]+\Z")


def set_control_token(env_path: Path, prompt):
    """Prompt privately and atomically update only CONTROL_BOT_TOKEN.

    The caller owns the path; this function never prints the value or returns it.
    """
    env_path = Path(env_path)
    if env_path.is_symlink() or not env_path.is_file():
        raise RuntimeError("env_missing_or_unsafe")

    token = prompt("Вставьте токен управляющего бота (ввод скрыт): ").strip()
    if not token:
        raise RuntimeError("control_token_empty")
    if not _BOT_TOKEN.fullmatch(token):
        raise RuntimeError("control_token_invalid")

    with env_path.open("r", encoding="utf-8", newline="") as source:
        original = source.read()
    lines = original.splitlines(keepends=True)
    updated = []
    replaced = False
    for line in lines:
        if _TOKEN_LINE.match(line):
            if not replaced:
                ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                updated.append(f"CONTROL_BOT_TOKEN={token}{ending}")
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        if updated and not updated[-1].endswith(("\n", "\r")):
            updated[-1] += "\n"
        updated.append(f"CONTROL_BOT_TOKEN={token}\n")
    contents = "".join(updated)

    fd, temporary_name = tempfile.mkstemp(prefix=".env.", dir=env_path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
            target.write(contents)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, env_path)
        os.chmod(env_path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise RuntimeError("env_update_failed") from None
