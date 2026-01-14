#!/bin/sh

exec gunicorn felAPI:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind "${BACKEND_API_IP}:${BACKEND_API_PORT}"