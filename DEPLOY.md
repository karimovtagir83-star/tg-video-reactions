# Google Cloud: запуск на VM 24/7

Здесь локально подготовлены bootstrap-скрипт и systemd unit. Они не создают аккаунт, проект или VM. Эти шаги, подтверждение способа оплаты, SSH-подключение и ввод API-ключей остаются за вами; скрипт запускается уже внутри созданной VM.

## 1. Создайте VM в Google Cloud

1. В [Google Cloud Console](https://console.cloud.google.com/) создайте проект и подключите Cloud Billing. Google может попросить карту. Включите Compute Engine API.
2. Создайте Ubuntu 24.04 VM типа `e2-micro` в регионе `us-central1`, `us-west1` или `us-east1`; стандартный диск — до 30 GB. Не включайте HTTP/HTTPS firewall rules: бот не принимает входящие веб-запросы.
3. Free Tier ограничен регионом, типом машины, диском и примерно 1 GB исходящего трафика в месяц. Пробный период длится 90 дней; дальше бесплатный лимит требует активного платёжного аккаунта. Перерасход может списываться с карты. Уведомление бюджета не ограничивает списания. Проверьте [актуальные лимиты Google](https://docs.cloud.google.com/free/docs/free-cloud-features) и настройте уведомление до запуска.

## 2. Подключитесь и скопируйте проект

На домашнем Linux-компьютере создайте SSH-ключ, если его ещё нет: `ssh-keygen -t ed25519` (на вопросы можно отвечать Enter). Добавьте файл `~/.ssh/id_ed25519.pub` в SSH keys созданной VM через консоль Google Cloud. Имя пользователя и внешний IP возьмите в карточке VM, затем подключитесь:

```bash
ssh ИМЯ_ПОЛЬЗОВАТЕЛЯ@ВНЕШНИЙ_IP
```

В другом окне домашнего терминала соберите архив и сначала проверьте его список. В команду включён явный список нужных файлов — архив не обходит каталог проекта и не захватывает неизвестные файлы, `.env.local`, другие `.env.*`, `.venv`, `data/`, session или базу данных. При добавлении новых файлов проекта обновите список вручную.

```bash
cd "$HOME/Проекты/tg_bot"
rm -f /tmp/tg_video_bot.tar.gz && \
tar --create --gzip --file=/tmp/tg_video_bot.tar.gz -- \
  .env.example .gitignore README.md PLAN.md DEPLOY.md bot.sh pyproject.toml uv.lock \
  requirements.in requirements-dev.in requirements.lock requirements-dev.lock \
  deploy/bootstrap_vm.sh deploy/tg-video-reactions.service.in \
  tests/test_offline.py \
  tg_bot/__init__.py tg_bot/__main__.py tg_bot/ai.py tg_bot/config.py \
  tg_bot/links.py tg_bot/metadata.py tg_bot/runner.py tg_bot/state.py tg_bot/telegram.py && \
tar --list --gzip --file=/tmp/tg_video_bot.tar.gz
```

Перед `scp` убедитесь, что вывод `tar --list` содержит только перечисленные файлы и `.env.example`. Если архивация или проверка списка завершилась ошибкой, не переносите архив; исправьте список и повторите.

После проверки списка перенесите архив и распакуйте его на VM:

```bash
scp /tmp/tg_video_bot.tar.gz ИМЯ_ПОЛЬЗОВАТЕЛЯ@ВНЕШНИЙ_IP:~/tg_video_bot.tar.gz && \
ssh ИМЯ_ПОЛЬЗОВАТЕЛЯ@ВНЕШНИЙ_IP \
  'mkdir -p "$HOME/tg_video_bot" && tar --extract --gzip --file="$HOME/tg_video_bot.tar.gz" -C "$HOME/tg_video_bot" && rm "$HOME/tg_video_bot.tar.gz"' && \
rm /tmp/tg_video_bot.tar.gz
```

## 3. Установите проект на VM

В SSH-терминале VM выполните:

```bash
cd ~/tg_video_bot
./deploy/bootstrap_vm.sh --dry-run
./deploy/bootstrap_vm.sh --apply
```

Первый запуск только показывает план. `--apply` установит `uv`, Python 3.13, Deno и зависимости, создаст отсутствующие `.env` и `data/`, закроет права на `.env` и поставит службу systemd. Он не читает значения `.env`, не запускает бота и не включает автозапуск. Запускайте его от обычного SSH-пользователя, без `sudo`.

## 4. Настройте Telegram и Gemini вручную

Откройте `.env` на VM командой `nano .env`. Введите `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.1-flash-lite`; пока оставьте один ID разрешённой группы в `ALLOWED_CHAT_IDS`. API ID/hash получите на [my.telegram.org](https://my.telegram.org), отдельный Gemini-ключ — в [Google AI Studio](https://aistudio.google.com/apikey). Gemini free tier имеет динамические квоты и может использовать отправленные title/description для улучшения продуктов; не передавайте конфиденциальные данные.

Сохраните `.env`, затем выполните команды на VM:

```bash
chmod 600 .env
./bot.sh login
./bot.sh chats
```

В `.env` укажите числовой ID одной нужной группы, затем выполните `./bot.sh check`. Код Telegram и пароль 2FA вводите только в SSH-терминале. Не копируйте session из домашнего компьютера и никому не отправляйте `.env` или файлы из `data/`.

Сначала проверьте одну ссылку через `./bot.sh preview --url "https://www.youtube.com/shorts/..."`, затем запустите `./bot.sh run` и убедитесь, что черновик подходит. Эти команды обращаются к сайтам и Gemini; dry-run не отправляет сообщение в Telegram.

## 5. Включите службу после проверки

Bootstrap поставил unit `tg-video-reactions`, но оставил его выключенным. После проверки черновиков явно включите постоянный режим отправки:

```bash
sudo systemctl enable --now tg-video-reactions
sudo systemctl status tg-video-reactions
sudo journalctl -u tg-video-reactions -f
```

Служба запускает `run --send`, хранит ключи только в защищённом `.env` через `EnvironmentFile` systemd и перезапускается после сбоя. После изменения настроек выполните `sudo systemctl restart tg-video-reactions`. Остановить бота и отключить автозапуск: `sudo systemctl disable --now tg-video-reactions`.

Даже VM Free Tier не гарантирует неизменные лимиты Gemini, отсутствие простоя облака или нулевую оплату при выходе за лимиты Google.
