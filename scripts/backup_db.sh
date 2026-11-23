#!/usr/bin/env bash
# Daily SQLite backup - meant to run from cron on the Hetzner server, and run
# by scripts/deploy.sh before every migration.
# The copies here live on the same disk as the database. The off-server copy
# is pulled from your own machine by scripts/pull_backup.sh (see "Backups" in
# README.md); nothing is pushed from here.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Same database the bot uses: DATABASE_URL from the environment, else from
# .env, else the bot's default (config.py). A relative sqlite path is
# relative to the bot's working directory, i.e. the repo root.
if [[ -z "${DB_PATH:-}" ]]; then
    db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" && -f "$REPO_DIR/.env" ]]; then
        db_url="$(sed -n 's/^DATABASE_URL=//p' "$REPO_DIR/.env" | tail -n 1)"
        db_url="${db_url#\"}"
        db_url="${db_url%\"}"
    fi
    db_url="${db_url:-sqlite+aiosqlite:///./kielikaveri.db}"
    if [[ "$db_url" != sqlite*:///* ]]; then
        echo "DATABASE_URL is not a sqlite URL: $db_url" >&2
        exit 1
    fi
    DB_PATH="${db_url#*:///}"
    if [[ "$DB_PATH" != /* ]]; then
        DB_PATH="$REPO_DIR/$DB_PATH"
    fi
fi
BACKUP_DIR="${BACKUP_DIR:-$REPO_DIR/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"

# sqlite3 would happily open a missing path as a new empty database and
# "back up" that - fail loudly instead.
if [[ ! -f "$DB_PATH" ]]; then
    echo "database not found: $DB_PATH" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
backup_file="$BACKUP_DIR/kielikaveri-$timestamp.db"

sqlite3 "$DB_PATH" ".backup '$backup_file'"

find "$BACKUP_DIR" -name 'kielikaveri-*.db' -mtime "+$KEEP_DAYS" -delete

echo "Backed up $DB_PATH to $backup_file"
