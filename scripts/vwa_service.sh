#!/usr/bin/env bash
# VWA service manager — start / stop / restart / status / reset-wal
#
# Usage:
#   bash scripts/vwa_service.sh start
#   bash scripts/vwa_service.sh stop
#   bash scripts/vwa_service.sh restart
#   bash scripts/vwa_service.sh status
#   bash scripts/vwa_service.sh reset-wal   # fix corrupted postgres WAL

SANDBOX=/projects/bghp/jguo14/vwa-reddit-sandbox
VWA_PORT=9999
APT="apptainer exec --writable --no-home --no-mount home ${SANDBOX}"

_apptainer_running() {
    pgrep -f "supervisord.*supervisord.conf" > /dev/null 2>&1
}

cmd_status() {
    echo "=== supervisord ==="
    if _apptainer_running; then
        echo "  RUNNING (PID $(pgrep -f 'supervisord.*supervisord.conf'))"
    else
        echo "  STOPPED"
    fi
    echo "=== processes ==="
    pid=$(pgrep -f "nginx.*nginx.conf" | head -1); [ -n "$pid" ] && echo "  nginx    RUNNING  pid $pid" || echo "  nginx    STOPPED"
    pid=$(pgrep -f "postgres.*pgsql"   | head -1); [ -n "$pid" ] && echo "  postgres RUNNING  pid $pid" || echo "  postgres STOPPED"
    pid=$(pgrep -f "php-fpm"           | head -1); [ -n "$pid" ] && echo "  php-fpm  RUNNING  pid $pid" || echo "  php-fpm  STOPPED"
    echo "=== VWA HTTP ==="
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
    echo "  http://localhost:${VWA_PORT} → HTTP ${code}"
}

_wait_dead() {
    # _wait_dead PATTERN [TIMEOUT_SECS]
    local pattern="$1" timeout="${2:-10}" i
    for i in $(seq 1 "$timeout"); do
        pgrep -f "$pattern" > /dev/null 2>&1 || return 0
        sleep 1
    done
    echo "[vwa_service] WARNING: '$pattern' still alive after ${timeout}s, forcing SIGKILL"
    pkill -KILL -f "$pattern" 2>/dev/null || true
    sleep 1
}

cmd_stop() {
    echo "[vwa_service] stopping all sandbox processes..."
    pkill -TERM -f "supervisord.*supervisord.conf" 2>/dev/null || true
    _wait_dead "supervisord.*supervisord.conf" 5
    pkill -KILL -f "supervisord.*supervisord.conf" 2>/dev/null || true

    pkill -KILL -f "nginx.*nginx.conf" 2>/dev/null || true
    pkill -KILL -x nginx               2>/dev/null || true
    pkill -KILL -f "php-fpm"           2>/dev/null || true

    # Postgres: TERM first (clean checkpoint), then wait, then KILL
    pkill -TERM -f "postgres.*pgsql/data" 2>/dev/null || true
    _wait_dead "postgres.*pgsql/data" 10
    pkill -KILL -f "postgres.*pgsql/data" 2>/dev/null || true
    _wait_dead "postgres.*pgsql" 5

    # Remove stale lock files only after process is confirmed dead
    rm -f "${SANDBOX}/usr/local/pgsql/data/postmaster.pid"
    $APT rm -f /run/postgresql/.s.PGSQL.5432 /run/postgresql/.s.PGSQL.5432.lock 2>/dev/null || true
    echo "[vwa_service] stopped"
}

cmd_start() {
    if _apptainer_running; then
        echo "[vwa_service] already running"
        cmd_status
        return 0
    fi
    echo "[vwa_service] clearing any stale processes..."
    # nginx and php-fpm: stateless, safe to SIGKILL
    pkill -KILL -f "nginx.*nginx.conf" 2>/dev/null || true
    pkill -KILL -x nginx               2>/dev/null || true
    pkill -KILL -f "php-fpm"           2>/dev/null || true
    # postgres: MUST use SIGTERM + wait for clean checkpoint write.
    # SIGKILL without waiting → no checkpoint → WAL corruption on next start.
    pkill -TERM -f "postgres.*pgsql"   2>/dev/null || true
    _wait_dead "postgres.*pgsql" 15
    # Only SIGKILL if postgres refused to exit (should not happen normally)
    pkill -KILL -f "postgres.*pgsql"   2>/dev/null || true
    # Ensure postmaster.pid is gone so new instance doesn't see a stale PID
    rm -f "${SANDBOX}/usr/local/pgsql/data/postmaster.pid"

    echo "[vwa_service] preparing runtime directories..."
    # These directories are normally created by docker-entrypoint.sh
    $APT mkdir -p /run/nginx /run/postgresql /var/tmp/nginx/client_body
    $APT rm -f /run/postgresql/.s.PGSQL.5432 /run/postgresql/.s.PGSQL.5432.lock 2>/dev/null || true
    $APT rm -f /usr/local/pgsql/data/postmaster.pid 2>/dev/null || true

    echo "[vwa_service] tuning PHP-FPM and postgres for high-concurrency training..."
    # PHP-FPM: default pm.max_children=5 is too low for 32 concurrent browser envs
    $APT sed -i 's/^pm\.max_children\s*=.*/pm.max_children = 32/' /usr/local/etc/php-fpm.d/www.conf
    # Postgres: frequent checkpoints reduce WAL data loss on unclean shutdown
    pgconf=/usr/local/pgsql/data/postgresql.conf
    $APT sed -i '/^checkpoint_timeout/d' "$pgconf"
    $APT sed -i '/^checkpoint_completion_target/d' "$pgconf"
    $APT bash -c "echo 'checkpoint_timeout = 30s' >> $pgconf"
    $APT bash -c "echo 'checkpoint_completion_target = 0.7' >> $pgconf"

    echo "[vwa_service] starting VWA sandbox..."
    $APT /usr/bin/supervisord -n -c /etc/supervisord.conf &
    disown

    echo "[vwa_service] waiting for postgres + nginx to be ready..."
    for i in $(seq 1 24); do
        sleep 5
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
        if [ "$code" = "200" ] || [ "$code" = "302" ]; then
            echo "[vwa_service] UP — HTTP ${code} after ${i}×5s"
            return 0
        fi
        echo "[vwa_service]   attempt $i/24: HTTP ${code}"
    done
    echo "[vwa_service] ERROR: VWA not ready after 120s"
    cmd_status
    exit 1
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_reset_wal() {
    echo "[vwa_service] resetting postgres WAL (fixes corrupted checkpoint records)..."
    cmd_stop
    # Verify postgres is truly dead before touching data directory
    if pgrep -f "postgres.*pgsql" > /dev/null 2>&1; then
        echo "[vwa_service] ERROR: postgres still running after stop — aborting reset"
        exit 1
    fi
    $APT rm -f /usr/local/pgsql/data/postmaster.pid
    out=$($APT pg_resetwal -f /usr/local/pgsql/data 2>&1)
    echo "[vwa_service] pg_resetwal: $out"
    if ! echo "$out" | grep -q "Write-ahead log reset"; then
        echo "[vwa_service] ERROR: pg_resetwal may have failed — check output above"
        exit 1
    fi
    echo "[vwa_service] WAL reset done — starting services"
    cmd_start
}

case "${1:-}" in
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    restart)    cmd_restart ;;
    status)     cmd_status ;;
    reset-wal)  cmd_reset_wal ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|reset-wal}"
        exit 1
        ;;
esac
