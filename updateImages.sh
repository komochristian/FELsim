#!/bin/sh

set -e

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

cd fel-app

docker buildx build \
--push \
--platform linux/arm64,linux/amd64 --tag komo04/felsimfront:$VERSION .

cd ../backend
docker buildx build \
--push \
--platform linux/arm64,linux/amd64 --tag komo04/felsimback:$VERSION .
