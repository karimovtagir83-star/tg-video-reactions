import argparse, asyncio, json, os, sys
from getpass import getpass
from pathlib import Path
from .config import Config, ROOT
from .metadata import fetch, MetadataError
from .links import supported_url
from .ai import draft, laugh_only
from .telegram import client, list_groups
from .env_tools import set_control_token
from .style import (
    build_profile, default_owner_manifest, load_private_profile, scan_export,
    write_private_artifacts,
)
from .reply_ai import generate_reply
from .context import sanitize_window
from .participants import (
    build_participant_summaries, participant_summary_path,
    write_private_participant_summaries,
)

DEFAULT_EXPORT_DIR = Path.home() / "Downloads" / "Telegram Desktop" / "ChatExport"


def _read_owner_manifest(path):
    if path is None:
        return default_owner_manifest()
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError
        return value
    except Exception as exc:
        raise RuntimeError("owners_manifest_invalid") from exc


def style_scan_command(export_dir, owners_file=None):
    result = scan_export(Path(export_dir).expanduser(), _read_owner_manifest(owners_file))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


def style_build_command(export_dir, output, owners_file=None,
                        manifest_output="data/style/owners.json", overwrite=False):
    manifest = _read_owner_manifest(owners_file)
    profile, report = build_profile(Path(export_dir).expanduser(), manifest)
    write_private_artifacts(
        profile, manifest, Path(output), Path(manifest_output), overwrite=overwrite,
    )
    aggregate = profile["aggregate_style"]
    summary = {
        **report,
        "message_chars": aggregate["message_chars"],
        "words_per_message_mean": aggregate["words_per_message_mean"],
        "sentences_per_message_mean": aggregate["sentences_per_message_mean"],
        "emoji_message_rate": aggregate["emoji_message_rate"],
        "newline_message_rate": aggregate["newline_message_rate"],
        "uppercase_message_rate": aggregate["uppercase_message_rate"],
        "punctuation_rates": aggregate["punctuation_rates"],
        "generic_fillers": aggregate["generic_fillers"],
        "abbreviations": aggregate["abbreviations"],
        "profile_path": str(output),
        "manifest_path": str(manifest_output),
        "file_permissions": "0600; parent directory 0700",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


def participant_build_command(export_dir, chat_id, owners_file=None,
                              output=None, overwrite=False):
    """Build one group's participant observations locally, without AI/network."""
    artifact, report = build_participant_summaries(
        Path(export_dir).expanduser(), chat_id, _read_owner_manifest(owners_file),
    )
    target = Path(output).expanduser() if output else participant_summary_path(chat_id)
    installed = write_private_participant_summaries(
        artifact, target, overwrite=overwrite,
    )
    print(json.dumps({
        **report,
        "summary_path": str(installed),
        "file_permissions": "0600; parent directory 0700",
        "ai_or_network_used": False,
    }, ensure_ascii=False, sort_keys=True, indent=2))


def _synthetic_profile():
    return {
        "schema_version": 1,
        "sample_count": 20,
        "aggregate_style": {
            "message_chars": {"mean": 42, "median": 36, "p90": 90},
            "words_per_message_mean": 8,
            "sentences_per_message_mean": 1.1,
            "emoji_message_rate": 0.12,
            "newline_message_rate": 0.03,
            "uppercase_message_rate": 0.02,
            "punctuation_rates": {".": 0.5, "!": 0.1, "?": 0.1},
            "generic_fillers": ["ну", "да"],
            "abbreviations": ["ахах"],
        },
        "style_rules": [],
    }


def reply_check_command(cfg):
    synthetic_window = {
        "messages": [
            {"speaker": "p1", "text": "Как тебе такой план?", "reply_to_previous": False},
            {"speaker": "self", "text": "", "reply_to_previous": False},
        ],
        "trigger_index": 1,
        "trigger_reason": "mention",
    }
    result = asyncio.run(generate_reply(
        cfg.gemini_key, getattr(cfg, "reply_model", "") or cfg.model,
        _synthetic_profile(), synthetic_window,
    ))
    print(f"Gemini ordinary-check: {result.reason}.")
    return 0 if result.text else 2


def reply_preview_command(cfg, input_file):
    try:
        value = json.loads(Path(input_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"conversation_window"}:
            raise ValueError
        window = sanitize_window(value["conversation_window"])
        profile = load_private_profile(cfg.style_profile_path)
    except Exception:
        raise RuntimeError("reply_preview_input_invalid") from None
    result = asyncio.run(generate_reply(
        cfg.gemini_key, getattr(cfg, "reply_model", "") or cfg.model,
        profile, window,
    ))
    if result.text:
        parts = [result.text]
        if result.follow_up:
            parts.append(result.follow_up)
        for index, part in enumerate(parts, start=1):
            print(f"Черновик обычного ответа {index}/{len(parts)}: " + part)
        return 0
    print(f"Обычный ответ пропущен: {result.reason}.")
    return 2


def setup():
    env=ROOT/".env"; example=ROOT/".env.example"
    if not env.exists():
        env.write_text(example.read_text(),encoding="utf-8"); os.chmod(env,0o600)
        print("Создан .env. Заполните его локально.")
    else: print(".env уже есть, оставил без изменений.")
    data=ROOT/"data"; data.mkdir(mode=0o700,exist_ok=True); os.chmod(data,0o700)

def check(cfg):
    missing=[]
    if not cfg.api_id: missing.append("TELEGRAM_API_ID")
    if not cfg.api_hash: missing.append("TELEGRAM_API_HASH")
    if not cfg.gemini_key: missing.append("GEMINI_API_KEY (бесплатный лимит зависит от аккаунта и модели)")
    if cfg.chats_error: missing.append("ALLOWED_CHAT_IDS содержит нечисловое значение")
    if not cfg.chats: missing.append("ALLOWED_CHAT_IDS (числовые ID групп)")
    for mod in ("telethon","google.genai","pydantic","dotenv","yt_dlp"):
        try: __import__(mod)
        except ImportError: missing.append("зависимость: "+mod)
    import shutil
    if not shutil.which("deno"): missing.append("Deno нужен для части YouTube ссылок")
    print("Локальная проверка завершена.")
    print("Настроено разрешенных ID: "+str(len(cfg.chats)))
    if not (ROOT/"data"/"user.session").exists(): missing.append("session не найдена (выполните login)")
    elif (ROOT/"data"/"user.session").stat().st_mode & 0o077: missing.append("права session должны быть только у владельца; выполните chmod 600 data/user.session")
    if (ROOT/"data").exists() and (ROOT/"data").stat().st_mode & 0o077: missing.append("права папки data должны быть 700")
    if (ROOT/".env").exists() and (ROOT/".env").stat().st_mode & 0o077: missing.append("права .env должны быть 600")
    if getattr(cfg, "reply_config_error", ""):
        missing.append("проверьте настройки обычных ответов в .env")
    if getattr(cfg, "auto_reply_enabled", False):
        try:
            load_private_profile(cfg.style_profile_path)
        except Exception:
            missing.append("обычные ответы включены, но приватный профиль недоступен или имеет небезопасные права")
    if missing: print("Нужно настроить: "+", ".join(missing)); return 1
    else: print("Локальная конфигурация готова. Сетевых проверок не выполнялось.")

async def do_login(cfg):
    if not cfg.api_id or not cfg.api_hash: raise RuntimeError("telegram_config_missing")
    if not sys.stdin.isatty(): raise RuntimeError("login_requires_local_tty")
    data=ROOT/"data"; data.mkdir(mode=0o700, exist_ok=True); os.chmod(data,0o700)
    old_umask=os.umask(0o077)
    c=client(cfg)
    try:
        await c.start() # Telethon prompts locally for phone/code/2FA.
        me=await c.get_me()
        session=ROOT/"data"/"user.session"
        if session.exists(): os.chmod(session,0o600)
        print(f"Вход выполнен: {me.id}. Session хранится локально в data/.")
    except Exception: raise RuntimeError("telegram_login_failed") from None
    finally:
        await c.disconnect()
        os.umask(old_umask)

async def do_chats(cfg):
    if not cfg.api_id or not cfg.api_hash: raise RuntimeError("telegram_config_missing")
    c=client(cfg)
    try:
        await c.connect()
        if not await c.is_user_authorized(): raise RuntimeError("telegram_not_logged_in_run_login")
        groups=await list_groups(c)
        if not groups: print("Групп не найдено.")
        for name,cid in groups: print(f"{name}\t{cid}")
    except RuntimeError: raise
    except Exception: raise RuntimeError("telegram_unavailable") from None
    finally: await c.disconnect()

async def preview(cfg,url):
    valid=supported_url(url)
    if not valid: raise RuntimeError("unsupported_url")
    try: meta=await asyncio.to_thread(fetch,valid[0])
    except MetadataError as e:
        if str(e) in {"metadata_unavailable", "metadata_timeout"}:
            print("Черновик: " + laugh_only())
            return
        raise RuntimeError(str(e)) from None
    title = str(meta.get("title") or "")[:300]
    description = str(meta.get("description") or "")[:1500]
    result=await draft(cfg.gemini_key,cfg.model,title,description)
    if result: print("Черновик: "+result)
    else: print("Пропуск: содержание явно не подходит для реакции.")

def main():
    p=argparse.ArgumentParser(prog="bot.sh",description="Ответы в разрешенных Telegram-группах.")
    sub=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("setup","check","login","chats","set-control-token"): sub.add_parser(cmd)
    pr=sub.add_parser("preview"); pr.add_argument("--url",required=True)
    scan=sub.add_parser("style-scan",help="безопасные агрегаты экспорта; работает локально")
    scan.add_argument("--export-dir",default=str(DEFAULT_EXPORT_DIR))
    scan.add_argument("--owners-file")
    build=sub.add_parser("style-build",help="собрать приватный агрегированный профиль без AI и сети")
    build.add_argument("--export-dir",default=str(DEFAULT_EXPORT_DIR))
    build.add_argument("--output",default="data/style/profile.json")
    build.add_argument("--owners-file")
    build.add_argument("--manifest-output",default="data/style/owners.json")
    build.add_argument("--overwrite",action="store_true")
    participants=sub.add_parser(
        "participants-build",
        help="локально собрать безопасные наблюдения по одному экспорту группы",
    )
    participants.add_argument("--export-dir",default=str(DEFAULT_EXPORT_DIR))
    participants.add_argument("--chat-id",required=True,type=int)
    participants.add_argument("--owners-file")
    participants.add_argument("--output")
    participants.add_argument("--overwrite",action="store_true")
    sub.add_parser("reply-check",help="проверить Gemini только на синтетическом окне")
    reply_preview=sub.add_parser("reply-preview",help="черновик по выбранному очищенному примеру")
    reply_preview.add_argument("--input-file",required=True)
    rr=sub.add_parser("run")
    rr.add_argument("--send",action="store_true",help="разрешить отправку только после всех проверок")
    rr.add_argument("--show-drafts",action="store_true",help="показать обычные черновики только в локальном TTY dry-run")
    a=p.parse_args()
    try:
        if a.cmd=="setup": setup(); return 0
        if a.cmd=="style-scan":
            style_scan_command(a.export_dir,a.owners_file); return 0
        if a.cmd=="style-build":
            style_build_command(a.export_dir,a.output,a.owners_file,a.manifest_output,a.overwrite); return 0
        if a.cmd=="participants-build":
            participant_build_command(
                a.export_dir, a.chat_id, a.owners_file, a.output, a.overwrite,
            ); return 0
        if a.cmd=="set-control-token":
            set_control_token(ROOT/".env", getpass)
            print("Токен управляющего бота сохранён в .env; значение не показывается.")
            return 0
        cfg=Config.load()
        if a.cmd=="check": return check(cfg) or 0
        if a.cmd=="login": asyncio.run(do_login(cfg))
        elif a.cmd=="chats": asyncio.run(do_chats(cfg))
        elif a.cmd=="preview": asyncio.run(preview(cfg,a.url))
        elif a.cmd=="reply-check": return reply_check_command(cfg)
        elif a.cmd=="reply-preview": return reply_preview_command(cfg,a.input_file)
        elif a.cmd=="run":
            from .runner import run
            asyncio.run(run(cfg,a.send,a.show_drafts))
        return 0
    except KeyboardInterrupt: print("Остановлено."); return 130
    except RuntimeError as e:
        messages = {"telegram_config_missing": "Заполните TELEGRAM_API_ID и TELEGRAM_API_HASH в .env.", "gemini_key_missing": "Заполните GEMINI_API_KEY в .env для AI-черновиков.", "allowlist_empty": "Добавьте числовые ID разрешенных групп в ALLOWED_CHAT_IDS.", "allowlist_invalid": "Исправьте ALLOWED_CHAT_IDS: нужны только целые числовые ID групп.", "telegram_not_logged_in_run_login": "Сначала выполните ./bot.sh login.", "login_requires_local_tty": "Для входа запустите команду в локальном терминале.", "unsupported_url": "Поддерживаются только ссылки TikTok, Instagram Reels и YouTube Shorts.", "metadata_unavailable": "Не удалось получить метаданные ролика; ссылка пропущена.", "metadata_timeout": "Получение метаданных заняло слишком долго; ссылка пропущена.", "metadata_too_large": "Ответ сайта слишком велик; ссылка пропущена.", "redirect_unsupported": "Ссылка перенаправляет на неподдерживаемый адрес.", "ai_unavailable": "AI-запрос не выполнен. Проверьте ключ, модель и доступность API.", "telegram_login_failed": "Не удалось войти в Telegram. Проверьте настройки и попробуйте снова.", "telegram_unavailable": "Не удалось подключиться к Telegram.", "control_bot_unavailable": "Не удалось запустить управляющего бота. Проверьте токен и доступность Telegram.", "env_missing_or_unsafe": "Не найден обычный файл .env. Сначала выполните ./bot.sh setup.", "control_token_empty": "Пустой ввод, настройки не изменены.", "control_token_invalid": "Формат токена не распознан, настройки не изменены.", "env_update_failed": "Не удалось безопасно обновить .env.", "already_running": "Бот уже запущен в другой сессии.", "unsupported_media": "Этот тип ролика не поддерживается.", "show_drafts_not_allowed_with_send": "--show-drafts доступен только в dry-run.", "show_drafts_requires_local_tty": "--show-drafts требует локальный интерактивный терминал.", "owners_manifest_invalid": "Файл сопоставления авторов некорректен.", "export_files_not_found": "В указанной папке не найдены подходящие HTML-файлы экспорта.", "export_file_too_large": "Один файл экспорта превышает локальный лимит размера.", "private_output_must_be_under_data": "Профили и сводки можно сохранить только внутри data/.", "private_profile_exists": "Целевой приватный файл уже существует; для замены нужен явный --overwrite.", "participant_export_must_be_one_chat": "Для сводки укажите экспорт ровно одной группы.", "reply_preview_input_invalid": "Не удалось проверить приватный профиль и выбранный пример.", "reply_config_invalid": "Исправьте настройки обычных ответов в .env."}
        print("Ошибка: "+messages.get(str(e), "Операция не выполнена."),file=sys.stderr); return 2
    except Exception:
        print("Ошибка: operation_failed",file=sys.stderr); return 2

if __name__=="__main__": raise SystemExit(main())
