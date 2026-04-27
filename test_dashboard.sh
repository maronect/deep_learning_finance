#!/usr/bin/env bash
# test_dashboard.sh
#
# Starts the API locally, waits for it to become healthy, runs a smoke test
# against every main endpoint, then opens the dashboard in the default browser.
#
# Usage:  ./test_dashboard.sh
# Requires: docker (compose v2), curl

set -uo pipefail

BASE="http://localhost:8000"
TIMEOUT=30

# ---- ANSI colors (safe on Linux and macOS) ----
RED=$'\033[0;31m'
GRN=$'\033[0;32m'
YLW=$'\033[1;33m'
CYN=$'\033[0;36m'
BLD=$'\033[1m'
RST=$'\033[0m'

# ---- Helpers ----
info()  { printf '\n%s%s%s%s\n' "$BLD" "$CYN" "$*" "$RST"; }
ok()    { printf '%s%s%s\n'     "$GRN" "$*" "$RST"; }
warn()  { printf '%s%s%s\n'     "$YLW" "$*" "$RST"; }
fail()  { printf '%s%s%s\n'     "$RED" "$*" "$RST"; }

status_color() {
    local code=$1
    if   [[ "$code" -ge 200 && "$code" -lt 300 ]] 2>/dev/null; then printf '%s' "$GRN"
    elif [[ "$code" -ge 300 && "$code" -lt 400 ]] 2>/dev/null; then printf '%s' "$YLW"
    else printf '%s' "$RED"
    fi
}

check_endpoint() {
    local label=$1
    local path=$2
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE$path" 2>/dev/null || echo "000")
    local col
    col=$(status_color "$code")
    printf '  %-40s %s%s%s\n' "$label" "$col" "$code" "$RST"
}

# ============================================================
# 1. Start the API
# ============================================================
info "1. Starting API with docker compose ..."
docker compose up -d

# ============================================================
# 2. Wait for /health to return HTTP 200
# ============================================================
info "2. Waiting for /health (timeout: ${TIMEOUT}s) ..."

elapsed=0
while true; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BASE/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
        ok "   API is healthy (${elapsed}s elapsed)"
        break
    fi
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
        fail "   ERROR: API did not become healthy within ${TIMEOUT}s."
        printf '   Check logs with: docker compose logs\n'
        exit 1
    fi
    printf '   %ds — waiting (last status: %s) ...\n' "$elapsed" "$code"
    sleep 2
    elapsed=$((elapsed + 2))
done

# ============================================================
# 3. Endpoint smoke test
# ============================================================
info "3. Endpoint status codes:"
printf '  %-40s %s\n'  "Endpoint" "HTTP"
printf '  %s\n' "$(printf -- '-%.0s' {1..48})"

check_endpoint "GET /"                        "/"
check_endpoint "GET /health"                  "/health"
check_endpoint "GET /assets"                  "/assets"
check_endpoint "GET /predictions"             "/predictions"
check_endpoint "GET /portfolio/weights"       "/portfolio/weights"
check_endpoint "GET /portfolio/frontier"      "/portfolio/frontier?n_points=10"
check_endpoint "GET /metrics/portfolio"       "/metrics/portfolio"
check_endpoint "GET /metrics/model"           "/metrics/model"
check_endpoint "GET /pipeline/runs"           "/pipeline/runs"
check_endpoint "GET /pipeline/stages"         "/pipeline/stages"
check_endpoint "GET /history/runs"            "/history/runs"
check_endpoint "GET /scheduler/status"        "/scheduler/status"
check_endpoint "GET /scheduler/logs"          "/scheduler/logs"

printf '\n  Note: 404 on /predictions, /portfolio/*, /metrics/* is expected\n'
printf '        if no pipeline run has been executed yet.\n'

# ============================================================
# 4. Open the dashboard in the default browser
# ============================================================
info "4. Opening dashboard in browser ..."

if command -v xdg-open &>/dev/null; then
    xdg-open "$BASE" &>/dev/null &
elif command -v open &>/dev/null; then
    open "$BASE"
else
    warn "   Could not detect a browser launcher."
    printf '   Open manually: %s\n' "$BASE"
fi

# ============================================================
# 5. Instructions
# ============================================================
printf '\n%s%s-----------------------------------------------------------%s\n' "$BLD" "$GRN" "$RST"
printf '%sDone.%s\n' "$BLD" "$RST"
printf '\n  Dashboard : %s%s%s\n'      "$CYN" "$BASE" "$RST"
printf '  Swagger   : %s%s/docs%s\n'  "$CYN" "$BASE" "$RST"
printf '  ReDoc     : %s%s/redoc%s\n' "$CYN" "$BASE" "$RST"
printf '\nWhen finished, stop the API and remove containers with:\n'
printf '\n  %sdocker compose down%s\n\n' "$BLD" "$RST"
