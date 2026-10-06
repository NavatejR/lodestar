#!/bin/sh
# The image holds two programs: `lodestar` (the CLI) and `lodestar-server`.
#
# `docker run lodestar:dev` starts the server; `docker run lodestar:dev lodestar
# sample /data demo` runs the CLI against the same volume. An argument that
# looks like a flag is assumed to belong to the server, so
# `docker run lodestar:dev --read-only` works without naming it.
set -eu

if [ "$#" -eq 0 ]; then
    set -- lodestar-server
elif [ "${1#-}" != "$1" ]; then
    set -- lodestar-server "$@"
fi

exec "$@"
