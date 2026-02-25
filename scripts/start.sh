#!/bin/bash
set -e

echo "=== AI Virtual Try-On Startup ==="

# Wait for Redis
until redis-cli -u "${REDIS_URL:-redis://redis:6379/0}" ping 2>/dev/null; do
  echo "Waiting for Redis…"
  sleep 2
done
echo "✅ Redis is up"

# Start Uvicorn
echo "Starting FastAPI on :8000 …"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop \
  --timeout-keep-alive 65
