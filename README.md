# finn-cards

**Kielikaveri** - a Telegram bot for learning Finnish: FST-verified
flashcards generated from your own mistakes, FSRS scheduling, voice
practice. "Kielikaveri" is the project's working name; this repo,
`finn-cards`, is its only home.

## Why

An LLM will confidently produce a plausible-looking but wrong Finnish word
form, and at B1 level that's not always obvious to the ear. So forms are
never invented by an LLM here - they come from `uralicNLP` (the omorfi
FST for Finnish), which by construction can only return forms that exist.
See `cards/instructions.md` for the exact algorithm and its rationale.

## Layout

- `cards/schema.json` - the note JSON Schema. Single source of truth;
  nothing else redefines the card format.
- `cards/instructions.md` - how to fill in a card without inventing forms.
- `cards/examples/` - three reference notes (verb, noun with consonant
  gradation, verb with non-trivial government/rektio).
- `src/finn_cards/` - phase 0: `morphology.py` (generate_forms,
  lemmatize, pos_set_for_lemma),
  `strict_schema.py` (converts `cards/schema.json` into an OpenAI
  `strict: true` schema).
- `src/kielikaveri/` - phase 1+: the bot.
  - `config.py` - settings loaded from `.env` (see `.env.example`).
  - `db/models.py` - SQLAlchemy models (`users`, `notes`, `cards`,
    `reviews`), mirroring `cards/schema.json`.
  - `db/engine.py` - async engine/session helpers.
  - `bot/main.py` - entrypoint: long polling, whitelist middleware,
    `/start` `/help`.
  - `bot/middleware.py` - `WhitelistMiddleware`, registered on
    `dp.update.outer_middleware`.
  - `bot/text.py` - `split_message`, for replies over Telegram's 4096
    character limit.
  - `import_cards.py` - imports note JSON files (schema-validated) into
    the database.
- `alembic/` - migrations for the schema above.
- `deploy/kielikaveri.service` - systemd unit template for the bot.
- `scripts/deploy_entry.sh`, `scripts/deploy.sh` - server side of the
  auto-deploy (see below).
- `scripts/backup_db.sh` - nightly SQLite backup on the server;
  `scripts/pull_backup.sh` - pulls the newest one to your own machine
  (see "Backups" below).
- `scripts/measure_coverage.py` - measures hypothesis G3 (does the FST
  cover the forms we need without LLM help) on a curated word list.

## Setup

Requires Python 3.12. First install downloads the Finnish FST models via
`uralicNLP`, so it can take a minute.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # fill in BOT_TOKEN, WHITELIST_USER_IDS at minimum
alembic upgrade head
```

## Running the bot

```bash
python -m kielikaveri.bot.main
```

## Auto-deploy

Started by hand: Actions -> deploy -> Run workflow, on `main` (other
branches skip the deploy job). `.github/workflows/deploy.yml` runs the test
suite first, then an ssh call to the server. The deploy key there can only run
`scripts/deploy_entry.sh`, which checks that the sent sha is on
`origin/main`, takes `scripts/deploy.sh` from that commit and runs it
detached (an ssh drop doesn't kill it mid-migration; the log is in
`logs/deploy-*.log`). `deploy.sh`: check that no chapter is being
translated, service stop, `git reset --hard`, `uv sync --frozen`,
`backup_db.sh`, `alembic upgrade head`, service start, then 20 seconds to
check it stayed up without restarts. If any step after the stop fails the
bot stays stopped - see the job log.

A stop in the middle of a chapter throws its paid translation away (the
chapter goes back to `queued` and starts over), so `deploy.sh` first counts
`chapters` with status `translating` in the database (`sqlite3`, read-only)
and, if there are any, exits with an error before touching anything: the bot
keeps running on the old code. Wait until the bot reports the chapter, then
run the deploy again. The check also refuses when `sqlite3` or the database
file is missing - `backup_db.sh` needs both a few steps later, after the
stop. To deploy anyway (the workflow can't pass it; the deploy key's forced
command ignores the environment), on the server:

```bash
sudo -iu kielikaveri
cd /opt/kielikaveri/finn_cards
git fetch origin main && git rev-parse origin/main    # the sha to deploy
FORCE_DEPLOY=1 SSH_ORIGINAL_COMMAND=<full sha> scripts/deploy_entry.sh
```

It runs the same steps as the workflow (minus the tests) and logs to
`logs/deploy-*.log`. The check only exists in commits that have it:
`deploy.sh` comes from the commit being deployed.

Rolling back: push a revert commit and run the deploy. Re-running an old deploy puts old code
on a database already migrated past it: alembic fails on the unknown
revision and the bot stays stopped.

One-time setup:

1. On your machine, a dedicated key without a passphrase:
   `ssh-keygen -t ed25519 -N "" -C kielikaveri-deploy -f deploy_key`.
2. On the server, as root:
   - add `deploy_key.pub` to `~kielikaveri/.ssh/authorized_keys` as
     ```
     restrict,command="/opt/kielikaveri/finn_cards/scripts/deploy_entry.sh" ssh-ed25519 AAAA... kielikaveri-deploy
     ```
     (`kielikaveri` needs a real shell, not `nologin`; the script must
     already be in the checkout);
   - let `kielikaveri` stop/start only the bot's service, via `visudo -f
     /etc/sudoers.d/kielikaveri-deploy`:
     ```
     kielikaveri ALL=(root) NOPASSWD: /usr/bin/systemctl stop kielikaveri.service, /usr/bin/systemctl start kielikaveri.service
     ```
   - `usermod -aG systemd-journal kielikaveri`, so a failed deploy shows
     the bot's journal lines;
   - `kielikaveri` must own `/opt/kielikaveri/finn_cards`: `chown -R
     kielikaveri:kielikaveri /opt/kielikaveri/finn_cards`. The venv has no
     pip; `deploy.sh` uses uv from `~kielikaveri/.local/bin/uv` (override
     with `UV=`);
   - the Finnish FST model must live in `~kielikaveri/.uralicnlp/fin`, not
     in `.venv/.../site-packages/uralicNLP/models/fin` - `uv sync`
     reinstalls uralicNLP whenever its version changes in `uv.lock`, which
     would wipe a model inside the package.
3. On GitHub, Settings -> Environments -> new environment `production`,
   Deployment branches limited to `main`, with secrets `DEPLOY_HOST`,
   `DEPLOY_SSH_KEY` (contents of `deploy_key`) and `DEPLOY_KNOWN_HOSTS`
   (output of `ssh-keyscan <host>`). Then delete the local `deploy_key`.

## Backups

**The backups contain the book text** (and all cards and review history).
Keep them on the server and on your own machine only: never in git, never
in a cloud folder or chat. `*.db` is in `.gitignore`, and `pull_backup.sh`
refuses a destination inside a git repo.

Two pieces:

- `scripts/backup_db.sh`, on the server: `sqlite3 .backup` into
  `backups/kielikaveri-<UTC timestamp>.db`, deletes copies older than 14
  days (`KEEP_DAYS`). `deploy.sh` runs it before every migration; a nightly
  run comes from cron.
- `scripts/pull_backup.sh`, on your machine (WSL): fetches the newest of
  those files over your normal ssh login (not the deploy key - it can only
  run `deploy_entry.sh`), checks it with `PRAGMA integrity_check`, keeps
  the last 14 copies (`-k`) in `~/kielikaveri-backups` (`-d`). Host and
  user come from arguments or `BACKUP_SSH_HOST`/`BACKUP_SSH_USER`; see
  `scripts/pull_backup.sh -h`. A failed fetch or check exits nonzero and
  leaves the existing copies alone.

Server, as root, once:

```bash
command -v sqlite3 || apt install sqlite3
crontab -e -u kielikaveri
# add (server local time, see `timedatectl`):
# 30 3 * * * cd /opt/kielikaveri/finn_cards && scripts/backup_db.sh >> logs/backup.log 2>&1
crontab -l -u kielikaveri
```

The next morning `ls -l /opt/kielikaveri/finn_cards/backups/` shows a fresh
file and `logs/backup.log` ends with `Backed up ...`.

Your machine, once:

```bash
sudo apt install sqlite3
scripts/pull_backup.sh <user>@<host>   # by hand first: accepts the host key, shows errors
```

It runs with ssh `BatchMode=yes`, so a scheduled run needs a key that works
without typing anything (no passphrase, or one already in a running agent).

Schedule, one of:

- **cron in WSL.** WSL doesn't start services by itself: either enable
  systemd (`[boot]` / `systemd=true` in `/etc/wsl.conf`, then `wsl
  --shutdown` from Windows and `sudo systemctl enable --now cron`) or run
  `sudo service cron start` after every WSL start. Cron only fires while
  the WSL VM is running, and a missed run is not repeated. `crontab -e`:
  ```
  0 9 * * * BACKUP_SSH_HOST=<host> BACKUP_SSH_USER=<user> $HOME/development/finn_cards/scripts/pull_backup.sh >> $HOME/kielikaveri-backups/pull.log 2>&1
  ```
- **Windows Task Scheduler** (also starts WSL itself). New task, daily
  trigger, "Run task as soon as possible after a scheduled start is
  missed" on, action `wsl.exe` with arguments
  ```
  -d <distro> -- bash -lc "BACKUP_SSH_HOST=<host> BACKUP_SSH_USER=<user> ~/development/finn_cards/scripts/pull_backup.sh >> ~/kielikaveri-backups/pull.log 2>&1"
  ```
  (`wsl -l` lists distro names).

Check now and then: `tail ~/kielikaveri-backups/pull.log` and `ls -l
~/kielikaveri-backups` - the newest file should be at most a day old.

Restoring: stop the service, copy the backup over the database file (the
one `DATABASE_URL` points to, `kielikaveri.db` in the checkout by default),
`chown kielikaveri:kielikaveri` it, start the service.

## Reader Mini App

The bot process also serves the reader (`kielikaveri/reader/`) on
`READER_HOST:READER_PORT`, `127.0.0.1:8080` by default. Telegram opens Mini
Apps over HTTPS only, so Caddy runs in front of it on the server (set up by
hand, deploys don't touch it):

```
# /etc/caddy/Caddyfile
46-62-147-103.sslip.io {
    reverse_proxy 127.0.0.1:8080
}
```

Ports 80 and 443 must be open (Let's Encrypt). Set `READER_URL=https://46-62-147-103.sslip.io/`
in `.env`.

The reader opens via `/read` ("📖 Читать" in the commands menu): the bot
replies with an inline "📖 Открыть читалку" button that launches the Mini App.
Empty `READER_URL` makes `/read` answer "Читалка не настроена.". Why not a
WebApp menu button or a keyboard button: a WebApp menu button would replace
the commands menu, and a Mini App opened from a reply-keyboard button gets no
signed initData (only `sendData`), which `reader/auth.py` requires - inline
buttons do get it.

On every start the bot re-registers its commands and sets the menu button
to the commands list, so after a deploy the old "📖 Читать" WebApp button
switches back automatically (Telegram may need the chat reopened to show it),
and `/read` appears in the menu after the restart.

## Logs

Logs go to stdout (captured by `journalctl -u kielikaveri` under systemd)
and to a daily-rotated file at `LOG_FILE` (default `logs/kielikaveri.log`,
14 days kept):

```bash
tail -f logs/kielikaveri.log
```

Set `LOG_LEVEL=DEBUG` in `.env` to also see the per-request lines `httpx`/
`openai` log for every OpenAI call.

## Server hardening

One-time steps by hand, on the server and in the Hetzner console; deploys
don't touch any of this. `ufw` is off on the server, so the only network
filter is the Hetzner Cloud Firewall below.

### Hetzner Cloud Firewall

Without it, the reader port 8080 is kept off the internet only by
`READER_HOST=127.0.0.1`: a `0.0.0.0` typo in `.env` would expose it without
TLS. The Cloud Firewall filters in front of the server and is edited in the
web console, so a wrong rule can't lock you out for good - you just fix the
rule.

1. Keep one ssh session to the server open the whole time.
2. https://console.hetzner.cloud -> your project -> **Firewalls** (left
   sidebar) -> **Create Firewall**. Name it, e.g. `kielikaveri`.
3. **Inbound rules** - keep exactly these, source "Any IPv4" and "Any
   IPv6":
   - TCP `22` (ssh; the new firewall comes with this rule - keep it);
   - TCP `80` (Let's Encrypt HTTP challenge, Caddy's redirect to HTTPS);
   - TCP `443` (the reader over HTTPS);
   - optional: UDP `443` (HTTP/3, Caddy speaks it; without it browsers
     fall back to TCP) and ICMP (ping; handy for checking the server is up).

   **Outbound rules**: leave empty - no outbound rules means all outbound
   traffic is allowed (the bot needs Telegram, OpenAI, GitHub).
4. **Apply to** -> **Select resources** -> **Servers** -> tick the server ->
   **Create Firewall**. It takes effect within seconds.
5. Check, from your machine, in a **new** terminal (the open session stays
   as a way back):
   ```bash
   ssh <user>@46.62.147.103 true && echo ssh ok         # ssh still works
   curl -sI https://46-62-147-103.sslip.io/ | head -n 1  # reader answers (HTTP/2 200 or similar)
   nc -vz -w 5 46.62.147.103 8080                        # must time out, not "succeeded"
   ```
   (`sudo apt install netcat-openbsd` if `nc` is missing.) If the new ssh
   login fails: in the console open the firewall -> **Rules** and check the
   TCP 22 rule, or **Resources** -> remove the server to drop the firewall
   altogether. The server's own web console (Servers -> the server ->
   **Console**, top right `>_`) works whatever the firewall says.

### Listening ports

On the server, once (and after changing Caddy or `READER_HOST`):

```bash
sudo ss -tlnp
```

Lists TCP sockets that accept connections, with the process. Expected:

- `0.0.0.0:22` and `[::]:22` - `sshd`;
- `*:80` and `*:443` - `caddy`;
- `127.0.0.1:2019` - Caddy's admin API, localhost only;
- `127.0.0.1:8080` - `python` (the bot's reader). If this shows `0.0.0.0`
  or `*`, fix `READER_HOST` in `.env` and restart the service;
- `127.0.0.53:53` / `127.0.0.54:53` - `systemd-resolved`, localhost only.

Anything else bound to `0.0.0.0`, `*` or `[::]` is worth a look. A quick
filter for what is reachable from outside:

```bash
sudo ss -tlnp | grep -vE '127\.0\.0\.(1|53|54):|\[::1\]:'
```

should print only the header, 22, 80 and 443. `sudo ss -ulnp` does the
same for UDP (Caddy's 443 for HTTP/3, the resolver's 53, maybe a DHCP
client on 68).

### journald size limit

The bot logs to stdout, and journald keeps it. By default journald may use
up to 10% of the filesystem (capped at 4G); set an explicit limit:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=500M\n' | sudo tee /etc/systemd/journald.conf.d/size.conf
sudo systemctl restart systemd-journald
```

A drop-in rather than an edit of `/etc/systemd/journald.conf`, so a package
upgrade never asks about a changed config file. 500M is months of this
bot's INFO logs. Check:

```bash
systemd-analyze cat-config systemd/journald.conf | grep SystemMaxUse  # the drop-in is read
journalctl --disk-usage    # "Archived and active journals take up ..." - at most ~500M
```

The restart doesn't stop the bot; the oldest journal files are removed
once the limit is reached. To shrink right away: `sudo journalctl
--vacuum-size=500M`.

### The app's log files

`logs/kielikaveri.log` needs nothing extra: `logging_config.py` rotates it
at midnight (`TimedRotatingFileHandler`) and keeps 14 old files, so at most
15 days are on disk. Size isn't capped, but at `LOG_LEVEL=INFO` a day is
small; leave `DEBUG` on only for a while. Don't add logrotate for this
file - two rotators on one file fight each other. The other files in
`logs/`: `deploy-*.log` are deleted after 30 days by `deploy_entry.sh`;
`backup.log` gets one line a night. Look now and then:

```bash
du -sh /opt/kielikaveri/finn_cards/logs /opt/kielikaveri/finn_cards/backups
df -h /
```

### Off-server backup copy (still to do)

Until `pull_backup.sh` runs on your machine, every backup is on the same
disk as the database. Full description in "Backups"; the steps, in WSL:

```bash
sudo apt install sqlite3      # pull_backup.sh checks each copy with it
cd ~/development/finn_cards
scripts/pull_backup.sh -h     # options; check the defaults fit
scripts/pull_backup.sh <user>@46.62.147.103
```

`<user>` is your normal ssh login to the server, not the deploy key. The
first run asks to accept the host key (scheduled runs use
`BatchMode=yes` and can't) and should end with `Fetched
kielikaveri-<timestamp>.db to /home/<you>/kielikaveri-backups
(integrity_check: ok)` and `Local copies: 1 in ...`. Check:

```bash
ls -l ~/kielikaveri-backups   # one fresh kielikaveri-*.db, mode -rw-------
```

Then schedule it - cron in WSL or Windows Task Scheduler, both described
in "Backups" (use `BACKUP_SSH_HOST=46.62.147.103 BACKUP_SSH_USER=<user>`
there). The next day `tail ~/kielikaveri-backups/pull.log` should show a
new `Fetched ...` line.

## Importing phase 0 cards

```bash
python -m kielikaveri.import_cards --user-id <your telegram id>
```

## Tests

```bash
pytest
```

## Lint

```bash
ruff check .
```

## License

GPL-3.0 - the Finnish FST models this project depends on (via omorfi) are
GPL-licensed.
