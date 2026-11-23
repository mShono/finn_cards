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
- `scripts/backup_db.sh` - daily SQLite backup (off-server copy step not
  wired up yet - see the script).
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
`logs/deploy-*.log`). `deploy.sh`: service stop, `git reset --hard`, `uv
sync --frozen`, `backup_db.sh`, `alembic upgrade head`, service start, then
20 seconds to check it stayed up without restarts. If any step fails the
bot stays stopped - see the job log.

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

Ports 80 and 443 must be open (Let's Encrypt). With `READER_URL=https://46-62-147-103.sslip.io/`
in `.env` the bot sets its menu button to "📖 Читать" on start; empty
`READER_URL` restores the default commands button.

## Logs

Logs go to stdout (captured by `journalctl -u kielikaveri` under systemd)
and to a daily-rotated file at `LOG_FILE` (default `logs/kielikaveri.log`,
14 days kept):

```bash
tail -f logs/kielikaveri.log
```

Set `LOG_LEVEL=DEBUG` in `.env` to also see the per-request lines `httpx`/
`openai` log for every OpenAI call.

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
