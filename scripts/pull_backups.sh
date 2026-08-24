#!/usr/bin/env bash
# Run this FROM your own machine (not the server) to pull down the SQLite
# backups that scripts/backup_db.sh creates on the server. This is the
# off-server copy: it survives the server dying, being deleted, etc.,
# because it lives on a different machine entirely.
set -euo pipefail

SERVER="${SERVER:?set SERVER=user@host, e.g. SERVER=kielikaveri@1.2.3.4}"
REMOTE_DIR="${REMOTE_DIR:-/opt/kielikaveri/finn_cards/backups}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/yki/backups/kielikaveri}"

mkdir -p "$LOCAL_DIR"
rsync -avz --progress "$SERVER:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "Synced to $LOCAL_DIR"
