#!/usr/bin/env bash
# Loads the local news-assessment model into Ollama's memory ahead of market
# open so llm_news.py's first request of the day isn't a cold start. Reads
# the model name out of config.json so it never drifts from what the
# strategy actually calls. Run once daily from ollama-warmup.timer.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$(python3 -c "import json; print(json.load(open('${PROJECT_DIR}/config.json'))['LLM_NEWS_MODEL'])")"

curl -s -m 300 http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"ok\"}], \"max_tokens\": 1}" \
    >/dev/null
