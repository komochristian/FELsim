#!/bin/sh
exec uvicorn felAPI:app --host="${BACKEND_API_IP}" --port="${BACKEND_API_PORT}" --reload
