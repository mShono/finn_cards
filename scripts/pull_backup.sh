#!/usr/bin/env bash
# Off-server copy of the SQLite backups. Runs on your own machine (WSL), not
# on the server: fetches the newest backups/kielikaveri-*.db that
# scripts/backup_db.sh made on the server, checks it with
# "PRAGMA integrity_check" and keeps the last N local copies. The check uses
# the sqlite3 CLI if it is installed and python3's sqlite3 module otherwise,
# so nothing here needs sudo.
#
# Uses your normal ssh login to the server, NOT the deploy key: that key has
# a forced command (scripts/deploy_entry.sh) and can't run anything else.
#
# The backup contains the book text - the destination must stay local and
# outside any git repository (the script refuses a destination inside one).
#
# Usage:
#   scripts/pull_backup.sh [options] [[user@]host]
#     -u USER   ssh user              (env BACKUP_SSH_USER)
#     -H HOST   ssh host or ~/.ssh/config alias (env BACKUP_SSH_HOST)
#     -d DIR    local destination     (env BACKUP_DEST, default ~/kielikaveri-backups)
#     -k N      local copies to keep  (env BACKUP_KEEP, default 14)
#     -r DIR    backups dir on the server
#               (env BACKUP_REMOTE_DIR, default /opt/kielikaveri/finn_cards/backups)
#   Arguments override environment variables. The user may be left out if
#   the host is an ~/.ssh/config alias that sets it.
set -euo pipefail

die() {
    echo "pull_backup: $*" >&2
    exit 1
}

# Prints the result of "PRAGMA integrity_check" for the database file $1
# ("ok" for a good file) and fails if the check doesn't pass or can't run.
# $1 is opened read-only, so the check never changes it. Uses $checker, set
# by main().
integrity_check() {
    if [[ "$checker" == sqlite3 ]]; then
        sqlite3 -readonly "$1" "PRAGMA integrity_check"
    else
        # The path goes in as an argument and is percent-encoded into the
        # URI, never pasted into the code. -I: ignore PYTHON* variables and
        # the current directory.
        python3 -I -c '
import pathlib, sqlite3, sys
uri = pathlib.Path(sys.argv[1]).resolve().as_uri() + "?mode=ro"
try:
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = [str(r[0]) for r in con.execute("PRAGMA integrity_check")]
    finally:
        con.close()
except sqlite3.Error as e:
    sys.exit(str(e))
print("\n".join(rows))
sys.exit(0 if rows == ["ok"] else 1)
' "$1"
    fi
}

usage() {
    sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

main() {
    local user="${BACKUP_SSH_USER:-}"
    local host="${BACKUP_SSH_HOST:-}"
    local dest="${BACKUP_DEST:-$HOME/kielikaveri-backups}"
    local keep="${BACKUP_KEEP:-14}"
    local remote_dir="${BACKUP_REMOTE_DIR:-/opt/kielikaveri/finn_cards/backups}"

    while getopts ":u:H:d:k:r:h" opt; do
        case "$opt" in
            u) user="$OPTARG" ;;
            H) host="$OPTARG" ;;
            d) dest="$OPTARG" ;;
            k) keep="$OPTARG" ;;
            r) remote_dir="$OPTARG" ;;
            h) usage; exit 0 ;;
            :) die "option -$OPTARG needs a value (see -h)" ;;
            *) die "unknown option -$OPTARG (see -h)" ;;
        esac
    done
    shift $((OPTIND - 1))
    if [[ $# -gt 1 ]]; then
        die "too many arguments (see -h)"
    fi
    if [[ $# -eq 1 ]]; then
        if [[ "$1" == *@* ]]; then
            user="${1%%@*}"
            host="${1#*@}"
        else
            host="$1"
        fi
    fi

    [[ -n "$host" ]] || die "no server given: pass [user@]host or set BACKUP_SSH_HOST (see -h)"
    [[ "$keep" =~ ^[1-9][0-9]*$ ]] || die "keep count must be a positive integer, got: '$keep'"
    [[ "$remote_dir" == /* ]] || die "remote dir must be an absolute path, got: '$remote_dir'"
    command -v ssh >/dev/null || die "ssh not found"
    command -v scp >/dev/null || die "scp not found"
    # Pick the checker before fetching anything: without one a broken copy
    # can't be told from a good one.
    local checker
    if command -v sqlite3 >/dev/null; then
        checker=sqlite3
    elif command -v python3 >/dev/null && python3 -I -c 'import sqlite3' 2>/dev/null; then
        checker=python3
    else
        die "can't check backups: neither the sqlite3 CLI nor python3 with its sqlite3 module found - nothing fetched"
    fi

    local target="$host"
    if [[ -n "$user" ]]; then
        target="$user@$host"
    fi
    # BatchMode: in a scheduled run there is nobody to type a password, so
    # fail fast instead of hanging.
    local ssh_opts=(-o BatchMode=yes -o ConnectTimeout=20)

    # The copy holds the book text and the whole learning history: keep it
    # readable by you only, and never inside a git checkout.
    umask 077
    mkdir -p "$dest"
    dest="$(cd "$dest" && pwd)"
    chmod 700 "$dest"
    if git -C "$dest" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        die "destination $dest is inside a git repository - pick a directory outside any repo"
    fi

    # Newest by name: backup_db.sh names files by UTC timestamp, so the
    # lexical order is the chronological one.
    local remote_cmd newest
    remote_cmd="cd $(printf '%q' "$remote_dir") && ls -1 | grep -E '^kielikaveri-.*\\.db\$' | sort | tail -n 1"
    if ! newest="$(ssh "${ssh_opts[@]}" "$target" "$remote_cmd")"; then
        die "could not list $remote_dir on $target (ssh failed, or the directory does not exist / has no backups)"
    fi
    if [[ ! "$newest" =~ ^kielikaveri-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}Z\.db$ ]]; then
        die "no backup found in $target:$remote_dir (got: '$newest') - is backup_db.sh running from cron?"
    fi

    local final="$dest/$newest"
    if [[ -f "$final" ]]; then
        echo "Already have $newest, not fetching again"
    else
        # Fetch under a temporary name and move it into place only after the
        # integrity check, so a broken or half-copied file never looks like
        # a good backup (and never pushes a good one out of the rotation).
        local tmp="$dest/.$newest.part"
        rm -f "$tmp"
        trap 'rm -f "$tmp"' EXIT
        if ! scp -q "${ssh_opts[@]}" "$target:$remote_dir/$newest" "$tmp"; then
            die "scp of $target:$remote_dir/$newest failed"
        fi
        local check
        # An empty file opens as an empty database and passes the check.
        if [[ ! -s "$tmp" ]]; then
            die "fetched $newest is empty, not kept"
        fi
        local check_rc=0
        check="$(integrity_check "$tmp" 2>&1)" || check_rc=$?
        if [[ "$check_rc" -ne 0 || "$check" != "ok" ]]; then
            die "integrity check failed for $newest, not kept: $check"
        fi
        mv "$tmp" "$final"
        trap - EXIT
        echo "Fetched $newest to $dest (integrity_check via $checker: ok)"
    fi

    # Rotation: keep the newest $keep copies, touch only files named like
    # backups.
    local -a backups=()
    local f
    while IFS= read -r f; do
        backups+=("$f")
    done < <(find "$dest" -maxdepth 1 -type f -name 'kielikaveri-*.db' -printf '%f\n' | sort)
    local excess=$((${#backups[@]} - keep))
    local i
    for ((i = 0; i < excess; i++)); do
        rm -f -- "$dest/${backups[i]}"
        echo "Removed old copy ${backups[i]}"
    done
    echo "Local copies: $(( ${#backups[@]} > keep ? keep : ${#backups[@]} )) in $dest"
}

main "$@"
