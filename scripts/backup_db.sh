#!/usr/bin/env bash
# Daily SQLite backup - meant to run from cron on the Hetzner server.
# Off-server copy destination is not wired up yet: pick one (private S3
# bucket via rclone/aws-cli, or pull nightly from your own machine over
# ssh/scp) and add that step below - not decided/configured yet.
set -euo pipefail

DB_PATH="${DB_PATH:-/opt/kielikaveri/finn_cards/kielikaveri.db}"
BACKUP_DIR="${BACKUP_DIR:-/opt/kielikaveri/finn_cards/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"

mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
backup_file="$BACKUP_DIR/kielikaveri-$timestamp.db"

sqlite3 "$DB_PATH" ".backup '$backup_file'"

find "$BACKUP_DIR" -name 'kielikaveri-*.db' -mtime "+$KEEP_DAYS" -delete

echo "Backed up $DB_PATH to $backup_file"
