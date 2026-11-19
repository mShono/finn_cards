#!/usr/bin/env bash
# Forced command of the deploy key: in ~kielikaveri/.ssh/authorized_keys the
# key is prefixed with
#   restrict,command="/opt/kielikaveri/finn_cards/scripts/deploy_entry.sh"
# so whatever the client asks to run, sshd runs this instead and passes the
# requested command in SSH_ORIGINAL_COMMAND. CI sends the commit sha there:
#   ssh kielikaveri@host <sha>
# A leaked key can therefore only redeploy commits already on GitHub's main,
# not open a shell next to .env and the database.
#
# By hand on the server: SSH_ORIGINAL_COMMAND=<sha> scripts/deploy_entry.sh
set -euo pipefail

# The whole body is a function so bash parses it before running anything:
# deploy.sh's `git reset --hard` replaces this very file mid-run.
main() {
    local app_dir="${APP_DIR:-/opt/kielikaveri/finn_cards}"
    local sha="${SSH_ORIGINAL_COMMAND:-}"

    if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
        echo "expected a full 40-character commit sha, got: '$sha'" >&2
        exit 2
    fi

    cd "$app_dir"
    mkdir -p logs

    # Held by the detached deploy below too (it inherits fd 9), so a second
    # deploy can't start until the first one has really finished, even if
    # this ssh session is long gone.
    exec 9>>logs/deploy.lock
    if ! flock -n 9; then
        echo "another deploy is still running" >&2
        exit 1
    fi

    git fetch --quiet origin main
    if ! git merge-base --is-ancestor "$sha" origin/main; then
        echo "$sha is not on origin/main" >&2
        exit 2
    fi

    # Run deploy.sh as it is in the commit being deployed, from a file rather
    # than a pipe: bash executes a script while still reading it, so anything
    # that read stdin would swallow the rest of a piped script.
    # script is global, not local: the EXIT trap runs after main returns.
    local log pid
    script="$(mktemp)"
    trap 'rm -f "$script"' EXIT
    git show "$sha:scripts/deploy.sh" > "$script"

    find logs -name 'deploy-*.log' -mtime +30 -delete
    log="logs/deploy-$(date -u +%Y-%m-%dT%H-%M-%SZ)-${sha:0:7}.log"
    echo "==> Log: $app_dir/$log"

    # Detached into its own session with output to a file: if the ssh
    # connection drops, the deploy still runs to the end instead of dying on
    # SIGPIPE halfway through a migration. CI just follows the log.
    setsid bash "$script" "$sha" > "$log" 2>&1 < /dev/null &
    pid=$!
    tail -n +1 -f --pid="$pid" "$log" || true
    wait "$pid"
}

main "$@"
