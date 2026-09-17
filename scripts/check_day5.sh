#!/usr/bin/env bash
# Day 5 check: API key + rate limit + single worker.
#
#   bash scripts/check_day5.sh                      # local: builds and runs the image
#   bash scripts/check_day5.sh https://<app>.onrender.com   # live: asks for the key
#
# Note: the rate limit is 10 requests/minute per IP and every /ask or /resume
# call below counts toward it, including the rejected ones.
set -u

BASE=${1:-}
IMAGE=rag-app:day5
NAME=rag-day5
PORT=8000
PASS=0; FAIL=0

ok()  { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

# HTTP status only. Usage: code METHOD PATH [KEY] [BODY]
code() {
  local args=(-s -o /dev/null -w '%{http_code}' -X "$1" "$BASE$2" -H 'Content-Type: application/json')
  [ -n "${3:-}" ] && args+=(-H "X-API-Key: $3")
  [ -n "${4:-}" ] && args+=(-d "$4")
  curl "${args[@]}"
}

wait_ready() {
  local start=$(date +%s)
  until curl -sf "$BASE/ready" >/dev/null; do
    if [ -z "$REMOTE" ] && [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" != "true" ]; then
      echo "container exited:"; docker logs --tail 30 $NAME; return 1
    fi
    [ $(( $(date +%s) - start )) -gt 600 ] && { echo "not ready after 600s"; return 1; }
    sleep 2
  done
  echo $(( $(date +%s) - start ))
}

if [ -z "$BASE" ]; then
  REMOTE=""
  BASE="http://localhost:$PORT"
  KEY=$(grep '^API_KEY=' .env | cut -d= -f2-)
  [ -z "$KEY" ] && { echo "API_KEY missing from .env"; exit 1; }

  echo "== 0. local setup"
  docker build -q -t $IMAGE . >/dev/null || exit 1

  echo "-- app refuses to start without API_KEY"
  OUT=$(docker run --rm --env-file <(grep -v '^API_KEY=' .env) $IMAGE 2>&1 | tail -3)
  if echo "$OUT" | grep -q "API_KEY"; then ok "startup fails fast: $(echo "$OUT" | tail -1)"
  else bad "started (or failed for another reason) without API_KEY: $OUT"; fi

  echo "-- single worker"
  if docker inspect -f '{{json .Config.Cmd}}' $IMAGE | grep -q -- "--workers 1"; then ok "CMD has --workers 1"
  else bad "--workers 1 not in image CMD"; fi

  docker rm -f $NAME >/dev/null 2>&1
  docker run -d --name $NAME --env-file .env -p $PORT:8000 $IMAGE >/dev/null || exit 1
else
  REMOTE=1
  BASE=${BASE%/}
  read -rsp "API key for $BASE: " KEY; echo
fi

echo "== 1. open endpoints (no key)"
T=$(wait_ready) || { echo "$T"; exit 1; }
echo "  ready after ${T}s  (on Render, a sleeping app's wake-up time)"
[ "$(code GET /health)" = 200 ] && ok "/health 200 without key" || bad "/health not 200"
[ "$(code GET /ready)"  = 200 ] && ok "/ready 200 without key"  || bad "/ready not 200"
DOCS=$(curl -s "$BASE/ready" | python3 -c "import sys,json; print(json.load(sys.stdin).get('documents',0))" 2>/dev/null || echo 0)
# The full corpus is ~700 chunks. A handful means the papers never made it into
# the image and only data/README.md was indexed.
[ "${DOCS:-0}" -ge 500 ] && ok "index has $DOCS chunks" \
  || bad "index has only ${DOCS:-0} chunks (want ~700): papers missing from the image?"

Q='{"thread_id":"day5-check","question":"What are reflection tokens in Self-RAG?"}'

echo "== 2. key check                       (uses 5 of the 10 requests)"
C=$(code POST /ask "" "$Q");           [[ $C =~ ^40[13]$ ]] && ok "no key -> $C"    || bad "no key -> $C (want 401/403)"
C=$(code POST /ask "wrong-key" "$Q");  [[ $C =~ ^40[13]$ ]] && ok "wrong key -> $C" || bad "wrong key -> $C (want 401/403)"
C=$(code POST /resume "wrong-key" '{"thread_id":"x","approved":true}')
[[ $C =~ ^40[13]$ ]] && ok "/resume wrong key -> $C" || bad "/resume wrong key -> $C (want 401/403)"

echo "  asking a real question with the key (small OpenAI cost)..."
RESP=$(curl -s -X POST "$BASE/ask" -H 'Content-Type: application/json' -H "X-API-Key: $KEY" -d "$Q")
echo "$RESP" | grep -q '"status":"completed"' && ok "right key -> answer returned" \
  || bad "right key -> unexpected: $(echo "$RESP" | head -c 300)"

C=$(code POST /resume "$KEY" '{"thread_id":"day5-nothing-pending","approved":true}')
[ "$C" = 409 ] && ok "/resume with key, nothing pending -> 409" || bad "/resume with key -> $C (want 409)"

echo "== 3. rate limit (empty bodies: rejected before the graph, cost nothing)"
CODES=""
for i in $(seq 1 8); do CODES="$CODES $(code POST /ask "$KEY" '{}')"; done
echo "  responses:$CODES"
N422=$(echo "$CODES" | tr ' ' '\n' | grep -c '^422$')
N429=$(echo "$CODES" | tr ' ' '\n' | grep -c '^429$')
# 5 requests used in step 2, so 5 more should pass (422) and the rest be 429.
[ "$N422" = 5 ] && [ "$N429" = 3 ] && ok "limit hit after exactly 10 requests" \
  || bad "expected 5x422 then 3x429, got ${N422}x422 ${N429}x429"

C=$(code POST /ask "" "$Q")
[ "$C" = 429 ] && ok "no-key request also throttled (limit checked before key)" || echo "  note: no-key request while throttled -> $C"

echo "== 4. window resets (waiting 61s)"
sleep 61
C=$(code POST /ask "$KEY" '{}')
[ "$C" = 422 ] && ok "after 60s the IP is allowed again" || bad "after 60s -> $C (want 422)"

if [ -n "$REMOTE" ]; then
  echo "== 5. proxy check (live only)"
  echo "  Behind Render's proxy, all visitors can end up sharing ONE limit."
  echo "  Test: from a second network (phone on mobile data), send 10 /ask calls"
  echo "  with no key, then immediately run this script from your laptop. If step 2"
  echo "  fails with 429s, the limit is keyed on the proxy IP, not the visitor."
fi

echo
echo "== result: $PASS passed, $FAIL failed"
[ -z "$REMOTE" ] && echo "clean up with: docker rm -f $NAME"
[ "$FAIL" = 0 ]
