#!/usr/bin/env bash
set -euo pipefail

cat >/tmp/rag_query.json <<'JSON'
{"question":"上海市城乡规划条例的适用范围是什么？","top_k":3}
JSON

curl -sS http://127.0.0.1:18080/query \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/rag_query.json \
  | python3 -m json.tool
