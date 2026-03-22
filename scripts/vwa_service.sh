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
        $APT supervisorctl status 2>/dev/null || true
    else
        echo "  STOPPED"
    fi
    echo "=== VWA HTTP ==="
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VWA_PORT}" 2>/dev/null)
    echo "  http://localhost:${VWA_PORT} → HTTP ${code}"
}

cmd_stop() {
    if ! _apptainer_running; then
        echo "[vwa_service] already stopped"
        return 0
    fi
    echo "[vwa_service] stopping services gracefully..."
    # Ask postgres to checkpoint before shutdown
    $APT supervisorctl stop all 2>/dev/null || true
    sleep 2
    # Now kill supervisord itself
    pkill -TERM -f "supervisord.*supervisord.conf" 2>/dev/null || true
    sleep 2
    if _apptainer_running; then
        echo "[vwa_service] WARNING: supervisord still running, sending SIGKILL"
        pkill -KILL -f "supervisord.*supervisord.conf" 2>/dev/null || true
    fi
    # Kill any lingering postgres (shared memory prevents clean restart)
    pkill -TERM -f "postgres.*pgsql/data" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "postgres.*pgsql/data" 2>/dev/null || true
    # Remove stale lock files
    rm -f /projects/bghp/jguo14/vwa-reddit-sandbox/usr/local/pgsql/data/postmaster.pid
    echo "[vwa_service] stopped"
}

cmd_start() {
    if _apptainer_running; then
        echo "[vwa_service] already running"
        cmd_status
        return 0
    fi
    echo "[vwa_service] preparing runtime directories..."
    # These directories are normally created by docker-entrypoint.sh
    $APT mkdir -p /run/nginx /run/postgresql /var/tmp/nginx/client_body
    $APT rm -f /run/postgresql/.s.PGSQL.5432 /run/postgresql/.s.PGSQL.5432.lock 2>/dev/null || true
    $APT rm -f /usr/local/pgsql/data/postmaster.pid 2>/dev/null || true

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
    sleep 2
    # Remove stale lock file left by unclean shutdown
    $APT rm -f /usr/local/pgsql/data/postmaster.pid
    $APT pg_resetwal -f /usr/local/pgsql/data
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
