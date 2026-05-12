#!/usr/bin/env bash
set -euo pipefail

. "$HOME/.config/lammarag_benchmark/env.sh"

curl --connect-timeout 10 --max-time 60 -sS \
  https://api.siliconflow.cn/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LAZYLLM_SILICONFLOW_API_KEY}" \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":["hello"]}' \
  | python3 -c 'import sys,json; data=json.load(sys.stdin); print("keys", sorted(data.keys())); print("dim", len(data.get("data", [{}])[0].get("embedding", []))); print("message", data.get("message", ""))'
