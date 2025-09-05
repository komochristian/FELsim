#!/bin/sh
VERSION=2.2

set -e

cd fel-app

docker buildx build \
--push \
--platform linux/arm64,linux/amd64 --tag komo04/felsimfront:$VERSION .

cd ../backend
docker buildx build \
--push \
--platform linux/arm64,linux/amd64 --tag komo04/felsimback:$VERSION .
