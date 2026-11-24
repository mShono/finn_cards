#!/usr/bin/env bash
# Deploys one commit on the server. Not run directly: scripts/deploy_entry.sh
# (the deploy key's forced command) takes this file from the commit being
# deployed and runs that copy, so the deploy logic always matches the code it
# deploys. Arguments: <sha>, a commit already fetched from origin/main.
#
# Needs a sudoers rule for the service stop/start - see the one-time setup in
# README.md ("Auto-deploy").
set -euo pipefail

# The sqlite file the bot uses, resolved like scripts/backup_db.sh does:
# DB_PATH, else DATABASE_URL (main() has already taken it from .env), else
# the bot's default from config.py. A relative path is relative to the
# checkout, which is the current directory here.
db_file_path() {
    if [[ -n "${DB_PATH:-}" ]]; then
        printf '%s\n' "$DB_PATH"
        return
    fi
    local db_url="${DATABASE_URL:-sqlite+aiosqlite:///./kielikaveri.db}"
    if [[ "$db_url" != sqlite*:///* ]]; then
        echo "!! DATABASE_URL is not a sqlite URL: $db_url" >&2
        return 1
    fi
    local path="${db_url#*:///}"
    if [[ "$path" != /* ]]; then
        path="$PWD/$path"
    fi
    printf '%s\n' "$path"
}

# Refuses to deploy while a chapter is being translated: the stop would cut
# it off, and on start it goes back to `queued` and is translated - and paid
# for - again from scratch (db/books.py). Runs before anything changes, so a
# refusal leaves the bot running on the old code. FORCE_DEPLOY=1 skips it.
# `queued` chapters are not counted: they lose nothing on a restart. A
# chapter picked up in the second between this check and the stop is not
# caught; that's accepted.
check_no_translating() {
    local sha="$1" app_dir="$2"
    local force_hint="To deploy anyway, on the server as kielikaveri:
     cd $app_dir && FORCE_DEPLOY=1 SSH_ORIGINAL_COMMAND=$sha scripts/deploy_entry.sh"

    if [[ "${FORCE_DEPLOY:-}" == 1 ]]; then
        echo "==> FORCE_DEPLOY=1: not checking for chapters being translated"
        return
    fi

    echo "==> Checking for chapters being translated"
    # A missing sqlite3 or database is an error, not a warning: backup_db.sh
    # needs both a few steps later and would fail anyway - only after the
    # bot was already stopped.
    if ! command -v sqlite3 >/dev/null; then
        echo "!! sqlite3 not found (apt install sqlite3). Not deploying, the bot keeps running." >&2
        echo "   backup_db.sh needs it too, so FORCE_DEPLOY=1 would only fail later." >&2
        exit 1
    fi
    local db_path
    if ! db_path="$(db_file_path)"; then
        echo "!! Not deploying, the bot keeps running." >&2
        exit 1
    fi
    if [[ ! -f "$db_path" ]]; then
        echo "!! database not found: $db_path. Not deploying, the bot keeps running." >&2
        echo "   Check DATABASE_URL in $app_dir/.env (backup_db.sh needs the file too)." >&2
        exit 1
    fi

    # Read-only; waits up to 5s if the bot holds a write lock right now.
    local sql=(sqlite3 -readonly -cmd ".timeout 5000" "$db_path")
    local has_table count
    if ! has_table="$("${sql[@]}" "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'chapters'")"; then
        echo "!! could not read $db_path. Not deploying, the bot keeps running." >&2
        echo "   $force_hint" >&2
        exit 1
    fi
    if [[ "$has_table" == 0 ]]; then
        # A database from before the books migration: nothing to lose.
        echo "==> No chapters table yet"
        return
    fi
    if ! count="$("${sql[@]}" "SELECT count(*) FROM chapters WHERE status = 'translating'")"; then
        echo "!! could not read the chapters table in $db_path. Not deploying, the bot keeps running." >&2
        echo "   $force_hint" >&2
        exit 1
    fi
    if [[ "$count" != 0 ]]; then
        echo "!! $count chapter(s) being translated right now. Not deploying, the bot keeps running:" >&2
        echo "   a stop would throw that paid translation away, and the chapter would start over." >&2
        echo "   Wait until the bot reports the chapter ready (or failed), then run the deploy again." >&2
        echo "   $force_hint" >&2
        exit 1
    fi
    echo "==> No chapters being translated"
}

main() {
    local sha="$1"
    local app_dir="${APP_DIR:-/opt/kielikaveri/finn_cards}"
    local service="${SERVICE:-kielikaveri.service}"
    # uv lives in ~/.local/bin, which isn't on PATH in a non-interactive ssh
    # session; the server venv was made by uv and has no pip.
    local uv="${UV:-$HOME/.local/bin/uv}"

    cd "$app_dir"

    # The bot reads DATABASE_URL from .env itself (pydantic), but alembic only
    # sees the environment - without this it would fall back to alembic.ini
    # and could migrate (or create) a different database than the bot uses.
    # Only this one value: sourcing the whole .env would break on any value
    # with spaces or shell metacharacters. Read before the stop: the
    # translation check below needs it too.
    local db_url
    db_url="$(sed -n 's/^DATABASE_URL=//p' .env | tail -n 1)"
    db_url="${db_url#\"}"
    db_url="${db_url%\"}"
    if [[ -n "$db_url" ]]; then
        export DATABASE_URL="$db_url"
    fi

    # Nothing has changed yet (deploy_entry.sh only fetched): a refusal here
    # leaves the old code on disk and the bot running.
    check_no_translating "$sha" "$app_dir"

    # Stop first, before the code on disk changes: if any later step fails,
    # the bot stays stopped on purpose. Otherwise a failed install would leave
    # new code on disk under the old running process, and the next automatic
    # restart (or reboot) would start it on an unmigrated database. Alembic on
    # SQLite has no transactional DDL, so a half-done migration is worse than
    # a red deploy job and a few minutes of downtime.
    echo "==> Stopping $service"
    sudo systemctl stop "$service"

    echo "==> Checking out $sha"
    # reset, not pull: the server checkout must never carry local commits or
    # edits, and a pull would refuse to move past them.
    git reset --hard "$sha"
    git log -1 --oneline

    echo "==> Installing dependencies from uv.lock"
    # Exactly the versions CI tested. The Finnish FST model must live in
    # ~/.uralicnlp, not inside the uralicNLP package: sync reinstalls a
    # package whenever its version differs from the lock, and a reinstall
    # would wipe a model kept in site-packages.
    "$uv" sync --frozen --quiet
    if ! .venv/bin/python -c 'import sys; from uralicNLP import uralicApi; sys.exit(0 if uralicApi.is_language_installed("fin") else 1)'; then
        echo "==> Finnish FST model missing, downloading (~370MB)"
        .venv/bin/python -c 'from uralicNLP import uralicApi; uralicApi.download("fin")'
    fi

    echo "==> Backing up the database"
    bash scripts/backup_db.sh

    echo "==> Running migrations"
    .venv/bin/alembic upgrade head

    echo "==> Starting $service"
    sudo systemctl start "$service"

    # Type=simple: "active" only means the process was spawned. A crash a few
    # seconds in (bad token, model load) is retried by Restart=on-failure and
    # can look active again by the time we check, so also require that
    # systemd didn't have to restart it. A manual start resets NRestarts.
    local restarts_before restarts_after
    restarts_before="$(systemctl show -p NRestarts --value "$service")"
    sleep 20
    restarts_after="$(systemctl show -p NRestarts --value "$service")"
    if ! systemctl is-active --quiet "$service" || [[ "$restarts_after" != "$restarts_before" ]]; then
        echo "!! $service did not stay up (restarts: $restarts_before -> $restarts_after)" >&2
        systemctl status "$service" --no-pager --lines=0 >&2 || true
        journalctl -u "$service" --no-pager -n 50 >&2 || true
        exit 1
    fi
    echo "==> Deployed $(git rev-parse --short HEAD), $service is up"
}

main "$@"
