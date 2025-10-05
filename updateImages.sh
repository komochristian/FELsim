#!/bin/sh

set -e

# Load environment variables safely
if [ -f .env ]; then
  set -o allexport
  . .env
  set +o allexport
fi

cd fel-app

docker buildx build \
--push \
--build-arg FRONTEND_PORT="$FRONTEND_PORT" \
--platform linux/arm64,linux/amd64 --tag "komo04/felsimfront:$VERSION" .

cd ../backend
docker buildx build \
--push \
--platform linux/arm64,linux/amd64 --tag "komo04/felsimback:$VERSION" .
