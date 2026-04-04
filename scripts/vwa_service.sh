#!/usr/bin/env bash
# VWA service manager — start / stop / restart / status / reset-wal / ensure
#
# Each node gets its own isolated postgres data directory (on /tmp), so
# multiple nodes can run VWA simultaneously without conflicts.
#
# Usage:
#   bash scripts/vwa_service.sh start      # stop existing + start fresh
#   bash scripts/vwa_service.sh stop       # stop all VWA processes
#   bash scripts/vwa_service.sh restart    # stop + start
#   bash scripts/vwa_service.sh status     # show process & HTTP status
#   bash scripts/vwa_service.sh reset-wal  # fix corrupted postgres WAL + start
#   bash scripts/vwa_service.sh ensure     # auto-repair: check → restart → reset-wal → verify
#
# Environment:
#   VWA_PORT=9999   # override to run on a different port (default: 9999)

set -uo pipefail

SANDBOX=/projects/bghp/jguo14/vwa-reddit-sandbox
VWA_PORT=${VWA_PORT:-9999}
NODE_ID=$(hostname -s)
# Use SLURM job ID for isolation when available (two jobs on same node get separate state)
JOB_TAG=${SLURM_JOB_ID:-$$}

# ── Per-job isolation ────────────────────────────────────────────────────────
# Postgres data lives on node-local /tmp to prevent cross-node PID conflicts.
# SLURM job ID (or PID) ensures two jobs on the same node don't collide.
LOCAL_STATE="/tmp/vwa-${NODE_ID}-${VWA_PORT}-${JOB_TAG}"
LOCAL_PGDATA="${LOCAL_STATE}/pgdata"

_init_local_pgdata() {
    if [ -d "$LOCAL_PGDATA" ]; then return 0; fi
    echo "[vwa_service] initializing local pgdata: ${LOCAL_PGDATA}"
    mkdir -p "$LOCAL_STATE"
    cp -a "${SANDBOX}/usr/local/pgsql/data" "$LOCAL_PGDATA"
    # Ensure clean state — remove any stale locks from the copy
    rm -f "${LOCAL_PGDATA}/postmaster.pid"
}

# Apptainer: bind local pgdata over the shared one
APT="apptainer exec --writable --no-home --no-mount home --bind ${LOCAL_PGDATA}:/usr/local/pgsql/data ${SANDBOX}"
# For commands that don't need pgdata (pre-init)
APT_BASE="apptainer exec --writable --no-home --no-mount home ${SANDBOX}"

# ── Helpers ──────────────────────────────────────────────────────────────────

_wait_dead() {
    local pattern="$1" timeout="${2:-10}"
    for _i in $(seq 1 "$timeout"); do
        pgrep -f "$pattern" > /dev/null 2>&1 || return 0
        sleep 1
    done
    return 1
}

_save_pids() {
    # Save supervisord PID and its children so we only kill our own processes
    local svpid
    svpid=$(pgrep -f "supervisord.*supervisord.conf" -n 2>/dev/null || true)
    echo "$svpid" > "${LOCAL_STATE}/supervisord.pid" 2>/dev/null || true
}

_nuke_all() {
    local pidfile="${LOCAL_STATE}/supervisord.pid"
    local svpid=""
    [ -f "$pidfile" ] && svpid=$(cat "$pidfile" 2>/dev/null)

    if [ -n "$svpid" ] && kill -0 "$svpid" 2>/dev/null; then
        # Kill our specific supervisord and its process group
        echo "[vwa_service] killing supervisord PID $svpid and children..."
        # Get all children before killing parent
        local children
        children=$(pgrep -P "$svpid" 2>/dev/null | tr '\n' ' ')
        kill -TERM "$svpid" 2>/dev/null || true
        sleep 2
        kill -KILL "$svpid" 2>/dev/null || true
        # Kill children (postgres, nginx, php-fpm spawned by this supervisord)
        for cpid in $children; do
            kill -TERM "$cpid" 2>/dev/null || true
        done
        sleep 2
        for cpid in $children; do
            kill -KILL "$cpid" 2>/dev/null || true
        done
        sleep 1
    else
        # No pidfile or stale — fall back to pattern-based kill (solo mode)
        echo "[vwa_service] no tracked supervisord — killing all matching processes..."
        pkill -TERM -f "supervisord.*supervisord.conf" 2>/dev/null || true
        _wait_dead "supervisord.*supervisord.conf" 5 || pkill -KILL -f "supervisord.*supervisord.conf" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "php-fpm" 2>/dev/null || true
        pkill -KILL -f "nginx" 2>/dev/null || true
        pkill -TERM -x postgres 2>/dev/null || true
        _wait_dead "postgres" 10 || {
            pkill -KILL -x postgres 2>/dev/null || true
            _wait_dead "postgres" 5
        }
    fi
    rm -f "$pidfile" 2>/dev/null || true
}

_clean_pg_locks() {
    rm -f "${LOCAL_PGDATA}/postmaster.pid" 2>/dev/null || true
    $APT rm -f /run/postgresql/.s.PGSQL.5432 /run/postgresql/.s.PGSQL.5432.lock 2>/dev/null || true
}

_is_healthy() {
    local c1 c2
    c1=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
    c2=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}/forums" 2>/dev/null)
    [[ "$c1" =~ ^(200|302)$ ]] && [[ "$c2" =~ ^(200|302)$ ]]
}

_verify_dead() {
    ! pgrep -x postgres > /dev/null 2>&1 && \
    ! pgrep -f "supervisord.*supervisord.conf" > /dev/null 2>&1 && \
    ! pgrep -f "php-fpm" > /dev/null 2>&1
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_status() {
    echo "=== node: ${NODE_ID}, port: ${VWA_PORT}, pgdata: ${LOCAL_PGDATA} ==="

    echo "=== supervisord ==="
    local spid
    spid=$(pgrep -f "supervisord.*supervisord.conf" | head -1)
    [ -n "$spid" ] && echo "  RUNNING (PID $spid)" || echo "  STOPPED"

    echo "=== processes ==="
    local pid
    pid=$(pgrep -f "nginx.*nginx.conf" | head -1); [ -n "$pid" ] && echo "  nginx    RUNNING  pid $pid" || echo "  nginx    STOPPED"
    pid=$(pgrep -x postgres | head -1);             [ -n "$pid" ] && echo "  postgres RUNNING  pid $pid" || echo "  postgres STOPPED"
    pid=$(pgrep -f "php-fpm" | head -1);            [ -n "$pid" ] && echo "  php-fpm  RUNNING  pid $pid" || echo "  php-fpm  STOPPED"

    echo "=== VWA HTTP ==="
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
    echo "  http://localhost:${VWA_PORT} → HTTP ${code}"

    echo "=== health ==="
    _is_healthy && echo "  HEALTHY" || echo "  UNHEALTHY"

    echo "=== pgdata ==="
    [ -d "$LOCAL_PGDATA" ] && echo "  ${LOCAL_PGDATA} ($(du -sh "$LOCAL_PGDATA" 2>/dev/null | cut -f1))" || echo "  not initialized"
}

cmd_stop() {
    echo "[vwa_service] stopping on ${NODE_ID}..."
    _nuke_all
    [ -d "$LOCAL_PGDATA" ] && _clean_pg_locks

    if _verify_dead; then
        echo "[vwa_service] stopped (all processes confirmed dead)"
    else
        echo "[vwa_service] WARNING: some processes may still be alive"
        pgrep -af "postgres|supervisord|php-fpm|nginx" 2>/dev/null | grep -v pgrep || true
    fi
}

cmd_start() {
    echo "[vwa_service] starting on ${NODE_ID} (port ${VWA_PORT})..."

    # If VWA is already healthy on our port, reuse it (another job may have started it)
    if _is_healthy; then
        echo "[vwa_service] VWA already healthy on port ${VWA_PORT} — reusing existing instance"
        return 0
    fi

    # Stop any existing processes on THIS node
    if ! _verify_dead; then
        echo "[vwa_service] found existing processes — stopping first"
        _nuke_all
        sleep 2
    fi

    if ! _verify_dead; then
        echo "[vwa_service] ERROR: processes still alive after cleanup — aborting"
        pgrep -af "postgres|supervisord|php-fpm|nginx" 2>/dev/null | grep -v pgrep || true
        exit 1
    fi

    # Initialize node-local postgres data (first time only)
    _init_local_pgdata
    _clean_pg_locks

    echo "[vwa_service] preparing runtime directories..."
    $APT mkdir -p /run/nginx /run/postgresql /var/tmp/nginx/client_body

    echo "[vwa_service] tuning PHP-FPM and postgres..."
    $APT sed -i 's/^pm\.max_children\s*=.*/pm.max_children = 64/' /usr/local/etc/php-fpm.d/www.conf
    # Nginx: use the right port
    $APT sed -i "s/listen\s*[0-9]*/listen ${VWA_PORT}/" /etc/nginx/http.d/default.conf 2>/dev/null || \
    $APT sed -i "s/listen\s*[0-9]*/listen ${VWA_PORT}/" /etc/nginx/conf.d/default.conf 2>/dev/null || true
    # Postgres: frequent checkpoints
    local pgconf=/usr/local/pgsql/data/postgresql.conf
    $APT sed -i '/^checkpoint_timeout/d' "$pgconf"
    $APT sed -i '/^checkpoint_completion_target/d' "$pgconf"
    $APT bash -c "echo 'checkpoint_timeout = 30s' >> $pgconf"
    $APT bash -c "echo 'checkpoint_completion_target = 0.7' >> $pgconf"

    echo "[vwa_service] launching supervisord..."
    $APT /usr/bin/supervisord -n -c /etc/supervisord.conf &
    local svpid=$!
    disown
    echo "$svpid" > "${LOCAL_STATE}/supervisord.pid"
    echo "[vwa_service] supervisord PID: $svpid"

    echo "[vwa_service] waiting for VWA to be ready..."
    for i in $(seq 1 30); do
        sleep 5
        if _is_healthy; then
            echo "[vwa_service] UP and healthy after ${i}×5s (${NODE_ID}:${VWA_PORT})"
            return 0
        fi
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
        echo "[vwa_service]   attempt $i/30: HTTP ${code}"
    done
    echo "[vwa_service] ERROR: VWA not healthy after 150s"
    cmd_status
    exit 1
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_reset_wal() {
    echo "[vwa_service] resetting postgres WAL..."
    cmd_stop
    sleep 2

    if pgrep -x postgres > /dev/null 2>&1; then
        echo "[vwa_service] ERROR: postgres still running — aborting"
        exit 1
    fi

    _init_local_pgdata
    _clean_pg_locks

    echo "[vwa_service] running pg_resetwal..."
    local out
    out=$($APT /usr/bin/pg_resetwal -f /usr/local/pgsql/data 2>&1)
    echo "[vwa_service] $out"

    if echo "$out" | grep -qi "reset\|success"; then
        echo "[vwa_service] WAL reset done"
    else
        echo "[vwa_service] WARNING: pg_resetwal output unclear — attempting start anyway"
    fi

    cmd_start
}

cmd_ensure() {
    echo "[vwa_service] ensure: checking VWA health on ${NODE_ID}..."

    if _is_healthy; then
        echo "[vwa_service] ensure: VWA is already healthy"
        return 0
    fi

    echo "[vwa_service] ensure: phase 1 — clean restart..."
    cmd_restart
    sleep 5
    if _is_healthy; then
        echo "[vwa_service] ensure: healthy after restart"
        return 0
    fi

    echo "[vwa_service] ensure: phase 2 — WAL reset..."
    cmd_reset_wal
    sleep 5
    if _is_healthy; then
        echo "[vwa_service] ensure: healthy after WAL reset"
        return 0
    fi

    echo "[vwa_service] ensure: phase 3 — nuke pgdata and start fresh..."
    cmd_stop
    rm -rf "$LOCAL_PGDATA"
    cmd_start
    sleep 5
    if _is_healthy; then
        echo "[vwa_service] ensure: healthy after fresh pgdata"
        return 0
    fi

    echo "[vwa_service] ensure: FAILED — VWA still unhealthy after all repair attempts"
    cmd_status
    exit 1
}

# ── Main ─────────────────────────────────────────────────────────────────────

case "${1:-}" in
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    restart)    cmd_restart ;;
    status)     cmd_status ;;
    reset-wal)  cmd_reset_wal ;;
    ensure)     cmd_ensure ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|reset-wal|ensure}"
        echo "  VWA_PORT=9999 $0 start   # override port"
        exit 1
        ;;
esac
