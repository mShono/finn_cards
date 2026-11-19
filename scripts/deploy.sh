#!/usr/bin/env bash
# Deploys one commit on the server. Not run directly: scripts/deploy_entry.sh
# (the deploy key's forced command) takes this file from the commit being
# deployed and runs that copy, so the deploy logic always matches the code it
# deploys. Arguments: <sha>, a commit already fetched from origin/main.
#
# Needs a sudoers rule for the service stop/start - see the one-time setup in
# README.md ("Auto-deploy").
set -euo pipefail

main() {
    local sha="$1"
    local app_dir="${APP_DIR:-/opt/kielikaveri/finn_cards}"
    local service="${SERVICE:-kielikaveri.service}"
    # uv lives in ~/.local/bin, which isn't on PATH in a non-interactive ssh
    # session; the server venv was made by uv and has no pip.
    local uv="${UV:-$HOME/.local/bin/uv}"

    cd "$app_dir"

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

    # The bot reads DATABASE_URL from .env itself (pydantic), but alembic only
    # sees the environment - without this it would fall back to alembic.ini
    # and could migrate (or create) a different database than the bot uses.
    # Only this one value: sourcing the whole .env would break on any value
    # with spaces or shell metacharacters.
    local db_url
    db_url="$(sed -n 's/^DATABASE_URL=//p' .env | tail -n 1)"
    db_url="${db_url#\"}"
    db_url="${db_url%\"}"
    if [[ -n "$db_url" ]]; then
        export DATABASE_URL="$db_url"
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
