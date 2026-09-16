#!/usr/bin/env bash
# Day 4 check: fresh container (no mounts) builds its index at boot,
# serves /ask, and skips ingestion on restart.
set -u

IMAGE=rag-app:day4
NAME=rag-day4
PORT=8000
QUESTION='What are reflection tokens in Self-RAG?'

# Poll /ready once a second; print seconds taken, or fail if the container dies.
wait_ready() {
  local start=$(date +%s)
  until curl -sf "http://localhost:$PORT/ready" >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" != "true" ]; then
      echo "FAIL: container exited before /ready returned 200. Last logs:"
      docker logs --tail 30 $NAME
      return 1
    fi
    if [ $(( $(date +%s) - start )) -gt 300 ]; then
      echo "FAIL: not ready after 300s"; docker logs --tail 30 $NAME; return 1
    fi
    sleep 1
  done
  echo $(( $(date +%s) - start ))
}

echo "== 1. build"
docker build -t $IMAGE . || exit 1

echo "== 2. fresh container, no volumes"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --env-file .env -p $PORT:8000 $IMAGE >/dev/null || exit 1
COLD=$(wait_ready) || { echo "$COLD"; exit 1; }
echo "cold boot: ${COLD}s"
curl -s "http://localhost:$PORT/ready"; echo
echo "--- ingestion log lines:"
docker logs $NAME 2>&1 | grep -iE "ingest|chunks|took" || echo "(no ingestion log line found; add one)"

echo "== 3. /ask"
curl -s -X POST "http://localhost:$PORT/ask" \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\": \"day4-check\", \"question\": \"$QUESTION\"}" \
  | python3 -m json.tool

echo "== 4. memory use (Render free tier = 512MB)"
docker stats --no-stream --format "{{.Name}}: {{.MemUsage}}" $NAME

echo "== 5. restart: should skip ingestion"
BEFORE=$(docker logs $NAME 2>&1 | grep -ciE "ingested")
docker restart $NAME >/dev/null
WARM=$(wait_ready) || { echo "$WARM"; exit 1; }
AFTER=$(docker logs $NAME 2>&1 | grep -ciE "ingested")
echo "warm boot: ${WARM}s"
if [ "$AFTER" -gt "$BEFORE" ]; then
  echo "FAIL: ingestion ran again on restart"
else
  echo "PASS: restart reused the existing index"
fi

echo "== done. cold=${COLD}s warm=${WARM}s (record both in deploy-notes.md)"
echo "clean up with: docker rm -f $NAME"
