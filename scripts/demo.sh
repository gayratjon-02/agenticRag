#!/usr/bin/env bash
# Two-minute demo of the Agentic RAG chatbot.
#
# Prerequisites (in separate terminals):
#   docker compose up -d
#   uv run alembic upgrade head
#   uv run uvicorn app.main:app --reload
#   uv run celery -A app.celery_app:celery_app worker --loglevel=info
#
# Then run: ./scripts/demo.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_HEADER=()
if [ -n "${ADMIN_API_KEY:-}" ]; then ADMIN_HEADER=(-H "X-Admin-Key: ${ADMIN_API_KEY}"); fi

echo "1) Creating a tenant…"
KEY=$(curl -s -X POST "$BASE_URL/tenants" "${ADMIN_HEADER[@]+"${ADMIN_HEADER[@]}"}" \
  -H 'Content-Type: application/json' -d '{"name":"Acme Corp"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")
echo "   tenant API key: ${KEY:0:12}…"

echo "2) Uploading the sample document…"
DID=$(curl -s -X POST "$BASE_URL/documents" -H "X-API-Key: $KEY" \
  -F "file=@samples/acme_handbook.txt;type=text/plain" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "   document id: $DID"

echo "3) Waiting for background ingestion…"
for _ in $(seq 1 60); do
  STATUS=$(curl -s "$BASE_URL/documents/$DID" -H "X-API-Key: $KEY" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ] && break
  sleep 1
done
echo "   status: $STATUS"

ask() {
  echo
  echo "Q: $1"
  curl -s -X POST "$BASE_URL/chat" -H "X-API-Key: $KEY" \
    -H 'Content-Type: application/json' -d "{\"question\": \"$1\"}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('A:', d['answer']); print('   grounded=%s escalated=%s sources=%s' % (d['grounded'], d['escalated'], [s['source'] for s in d['sources']]))"
}

ask "Who founded Acme Corp and where is it headquartered?"
ask "What is the return policy?"
ask "What is the boiling point of mercury?"

echo
echo "Done. Open the web widget at $BASE_URL/widget/ to chat interactively (paste the API key above)."
