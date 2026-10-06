# The Lodestar demo

A single container that serves a vector index, a CLI that writes to the same
files, and a browser console over the HTTP API. No cloud accounts, no external
services, nothing leaves your machine.

```bash
docker compose -f demo/docker-compose.yml up --build
open http://localhost:8080/demo
```

The first run builds the release binaries, seeds a 20,000-vector synthetic
collection (`demo`, 128 dimensions, l2) into a named volume, and starts the
service on port 8080. The seed is guarded by the collection's manifest, so
re-running `up` does not append a second copy of the corpus.

## What is where

| path | what it is |
|---|---|
| `Dockerfile` | multi-stage build: `rust:1` builder, `debian:bookworm-slim` runtime |
| `entrypoint.sh` | runs `lodestar-server` by default, the CLI when you name it |
| `docker-compose.yml` | seed → server, plus the opt-in `ollama` profile |
| `corpus/sentences.txt` | the bundled text corpus, `group: sentence` per line |
| `corpus/embed.py` | stdlib-only loader that embeds it with a local model |

## Endpoints worth trying

```bash
curl -s localhost:8080/healthz                       # version, kernel, uptime
curl -s localhost:8080/v1/index                     # every collection
curl -s localhost:8080/v1/index/demo/stats          # counters and sizes
curl -s localhost:8080/metrics                      # Prometheus text
open http://localhost:8080/docs                     # self-contained docs
open http://localhost:8080/demo                     # the console
```

A search, in one line:

```bash
curl -s localhost:8080/v1/search -H 'content-type: application/json' \
  -d '{"index":"demo","query":[0.1,0.2,...],"k":10}'
```

The console does the same thing with a form, and its filter box takes a
metadata expression such as `group == news AND score >= 0.8`.

## The CLI shares the volume

The image contains both programs, and they read the same directory. With
`lodestar-data` mounted at `/data`:

```bash
docker compose -f demo/docker-compose.yml exec lodestar lodestar stats demo --root /data
docker compose -f demo/docker-compose.yml exec lodestar lodestar verify demo --root /data
```

Writes made by the CLI appear in the API on the next `GET /v1/index`, which
re-scans the root. That is the same contract the Python bindings honour: one
data directory, three front doors.

## The `ollama` profile: real text, real embeddings

```bash
docker compose -f demo/docker-compose.yml --profile ollama up --build
```

This additionally starts [Ollama](https://ollama.com) and runs
`corpus/embed.py`, which pulls `nomic-embed-text`, embeds
`corpus/sentences.txt` and upserts it into a second collection called `docs`
(cosine). The model weights live in a named volume, so only the first run
downloads anything. Search `docs` from the console for sentences that mean
roughly the same thing.

The script is standard library Python on purpose: it waits for both services,
treats a repeat create as success, and exits non-zero if the model never comes
up, so a failed profile is visible rather than silent.

## Tear down

```bash
docker compose -f demo/docker-compose.yml down          # keep the data
docker compose -f demo/docker-compose.yml down -v       # and the volume
```
