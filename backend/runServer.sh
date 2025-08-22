#!/bin/sh
exec uvicorn apitest:app --host="${BACKEND_API_IP}" --port="${BACKEND_API_PORT}" --reload
