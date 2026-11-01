#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Подготовка Google Compute Engine VM под tg_video_bot.

Использование: deploy/bootstrap_vm.sh [--dry-run | --apply] [--help]

По умолчанию ничего не меняет: показывает план. --apply устанавливает
локальные системные зависимости, Python 3.13-окружение проекта, создаёт
отсутствующий .env и ставит systemd unit, но НЕ запускает бота.
Команду systemctl enable --now выполняйте только после check, login,
preview и проверки run без --send.

Запускайте от обычного SSH-пользователя, не через sudo. Скрипт использует
sudo только для apt и установки systemd unit. Ключи из .env он не читает,
не печатает и не копирует в unit.
USAGE
}

MODE=dry-run
while (($#)); do
    case "$1" in
        --dry-run) MODE=dry-run ;;
        --apply) MODE=apply ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Неизвестный параметр: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

BOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BOT_USER="$(id -un)"
BOT_HOME="$(getent passwd "$BOT_USER" | cut -d: -f6)"
TEMPLATE="$BOT_DIR/deploy/tg-video-reactions.service.in"

if [[ "$BOT_USER" == root || "$EUID" == 0 ]]; then
    echo "Запустите скрипт от обычного SSH-пользователя, без sudo." >&2
    exit 2
fi
if [[ -z "$BOT_HOME" || ! -f "$BOT_DIR/requirements.lock" || ! -f "$BOT_DIR/.env.example" || ! -f "$TEMPLATE" ]]; then
    echo "Не найдены файлы проекта или домашний каталог пользователя." >&2
    exit 2
fi
if ! python3 - "$BOT_DIR" "$BOT_HOME" "$BOT_USER" <<'PY'
import re
import sys

bot_dir, bot_home, bot_user = sys.argv[1:]
unsafe = set("\t\r\n #=%$\\\"'")
if any(char in unsafe for char in bot_dir + bot_home):
    raise SystemExit("Путь содержит символ, требующий ручной настройки systemd.")
if not re.fullmatch(r"[A-Za-z0-9_-]+", bot_user):
    raise SystemExit("Имя пользователя содержит неподдерживаемые символы.")
PY
then
    echo "Используйте каталог без пробелов и специальных символов, например ~/tg_video_bot." >&2
    exit 2
fi

if [[ "$MODE" == dry-run ]]; then
    cat <<PLAN
План для пользователя $BOT_USER, каталог $BOT_DIR:
  1. Если не хватает curl: установить curl и ca-certificates через apt.
  2. Установить uv и Deno в домашний каталог, если их ещё нет.
  3. Создать/обновить .venv по requirements.lock (Python 3.13).
  4. Выполнить setup: создать .env только если его ещё нет и закрыть права .env/data.
  5. Установить systemd unit в /etc/systemd/system/tg-video-reactions.service.
  6. Перечитать конфигурацию systemd; бота НЕ запускать и автозапуск НЕ включать.

Dry-run: изменений, загрузок, чтения .env и обращений к Google Cloud не было.
Для выполнения этих шагов на уже созданной VM запустите: $0 --apply
PLAN
    exit 0
fi

if [[ ! -t 0 ]]; then
    echo "--apply запускайте из SSH-терминала; для просмотра используйте --dry-run." >&2
    exit 2
fi

export PATH="$BOT_HOME/.local/bin:$BOT_HOME/.deno/bin:$PATH"
if ! command -v curl >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y curl ca-certificates
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$BOT_HOME/.local/bin:$PATH"
fi
if ! command -v deno >/dev/null 2>&1; then
    curl -fsSL https://deno.land/install.sh | sh
    export PATH="$BOT_HOME/.deno/bin:$PATH"
fi

cd "$BOT_DIR"
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python requirements.lock
./bot.sh setup
chmod 600 .env
chmod 700 data

unit_tmp="$(mktemp)"
trap 'rm -f "$unit_tmp"' EXIT
python3 - "$TEMPLATE" "$unit_tmp" "$BOT_USER" "$BOT_DIR" "$BOT_HOME" <<'PY'
from pathlib import Path
import sys

template_path, output_path, user, bot_dir, home = sys.argv[1:]
text = Path(template_path).read_text(encoding="utf-8")
for marker, value in {
    "@@BOT_USER@@": user,
    "@@BOT_DIR@@": bot_dir,
    "@@BOT_HOME@@": home,
}.items():
    text = text.replace(marker, value)
if "@@" in text:
    raise SystemExit("В шаблоне systemd остались незаполненные значения.")
Path(output_path).write_text(text, encoding="utf-8")
PY
sudo install -o root -g root -m 0644 "$unit_tmp" /etc/systemd/system/tg-video-reactions.service
sudo systemctl daemon-reload

cat <<'DONE'
Подготовка завершена. Служба установлена, но бот не запущен.
Дальше вручную:
  1. Заполните .env и проверьте: ./bot.sh check
  2. Выполните ./bot.sh login и ./bot.sh chats.
  3. Проверьте preview и ./bot.sh run без --send.
  4. Только после проверки включите отправку:
     sudo systemctl enable --now tg-video-reactions
DONE
