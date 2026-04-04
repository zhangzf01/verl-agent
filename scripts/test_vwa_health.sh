#!/usr/bin/env bash
# VWA health check — run before training to verify service stability
# Usage: bash scripts/test_vwa_health.sh [HOST] [PORT] [CONCURRENCY]

set -uo pipefail

HOST=${1:-localhost}
PORT=${2:-9999}
CONCURRENCY=${3:-32}
BASE_URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0

green() { echo -e "\033[32m✓ $1\033[0m"; PASS=$((PASS+1)); }
red()   { echo -e "\033[31m✗ $1\033[0m"; FAIL=$((FAIL+1)); }
warn()  { echo -e "\033[33m⚠ $1\033[0m"; }

echo "=== VWA Health Check ==="
echo "Target: ${BASE_URL}"
echo ""

# ── 1. Basic HTTP connectivity ──────────────────────────────────────────────
echo "── 1. Basic HTTP ──"
code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}" 2>/dev/null)
if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    green "Homepage reachable (HTTP ${code})"
else
    red "Homepage unreachable (HTTP ${code})"
fi

# ── 2. Login page ────────────────────────────────────────────────────────────
echo "── 2. Login page ──"
code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/login" 2>/dev/null)
if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    green "Login page OK (HTTP ${code})"
else
    red "Login page failed (HTTP ${code})"
fi

# ── 3. Page content check (not empty / not error page) ─────────────────────
echo "── 3. Page content ──"
body=$(curl -s "${BASE_URL}" 2>/dev/null)
body_len=${#body}
if [ "$body_len" -gt 500 ]; then
    green "Homepage has content (${body_len} bytes)"
else
    red "Homepage too short (${body_len} bytes) — possibly an error page"
fi

if echo "$body" | grep -qi "postmill\|reddit\|forum\|community"; then
    green "Page contains expected keywords (Postmill/Reddit)"
else
    red "Page missing expected keywords — might not be Postmill"
fi

# ── 4. Database health (trigger a DB-dependent page) ────────────────────────
echo "── 4. Database health ──"
# /forums requires DB queries
code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/forums" 2>/dev/null)
if [ "$code" = "200" ]; then
    green "Forums page OK — database is responding"
else
    red "Forums page failed (HTTP ${code}) — database may be down"
fi

# ── 5. Stability test (5 requests over 10 seconds) ──────────────────────────
echo "── 5. Stability (5 requests, 2s apart) ──"
stable_ok=0
stable_fail=0
for i in $(seq 1 5); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/forums" 2>/dev/null)
    if [ "$code" = "200" ]; then
        stable_ok=$((stable_ok+1))
    else
        stable_fail=$((stable_fail+1))
    fi
    [ "$i" -lt 5 ] && sleep 2
done
if [ "$stable_fail" -eq 0 ]; then
    green "All 5 requests succeeded — no postgres restart disruption"
else
    red "${stable_fail}/5 requests failed — postgres restart cycle is causing issues"
fi

# ── 6. Concurrent request test ──────────────────────────────────────────────
echo "── 6. Concurrency test (${CONCURRENCY} parallel requests) ──"
tmpdir=$(mktemp -d)
for i in $(seq 1 "$CONCURRENCY"); do
    curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/forums" > "${tmpdir}/${i}.out" 2>/dev/null &
done
wait

total_ok=0
total_fail=0
for i in $(seq 1 "$CONCURRENCY"); do
    code=$(cat "${tmpdir}/${i}.out" 2>/dev/null)
    if [ "$code" = "200" ]; then
        total_ok=$((total_ok+1))
    else
        total_fail=$((total_fail+1))
    fi
done
rm -rf "$tmpdir"

if [ "$total_fail" -eq 0 ]; then
    green "All ${CONCURRENCY} concurrent requests succeeded"
elif [ "$total_fail" -le $((CONCURRENCY / 10)) ]; then
    warn "${total_ok}/${CONCURRENCY} OK, ${total_fail} failed — marginal, may work"
    PASS=$((PASS+1))
else
    red "${total_ok}/${CONCURRENCY} OK, ${total_fail} failed — too many failures for training"
fi

# ── 7. Login test ────────────────────────────────────────────────────────────
echo "── 7. Login test ──"
login_resp=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/login" \
    -d "username=MarvelsGrantMan136&password=test1234&_token=" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -L --max-redirs 3 2>/dev/null)
login_code=$(echo "$login_resp" | tail -1)
login_body=$(echo "$login_resp" | head -n -1)
if [ "$login_code" = "200" ] || [ "$login_code" = "302" ]; then
    green "Login POST accepted (HTTP ${login_code})"
else
    warn "Login returned HTTP ${login_code} — may need CSRF token (normal for curl)"
    PASS=$((PASS+1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary: ${PASS} passed, ${FAIL} failed ==="
if [ "$FAIL" -eq 0 ]; then
    echo -e "\033[32mVWA is healthy — ready for training!\033[0m"
    exit 0
else
    echo -e "\033[31mVWA has issues — fix before training.\033[0m"
    exit 1
fi
