#!/usr/bin/env bash
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec .venv/bin/python -m tg_bot "$@"
