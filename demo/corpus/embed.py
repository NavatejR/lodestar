#!/usr/bin/env python3
"""Embed the bundled sentence corpus with a local Ollama model.

This script exists for the Docker demo's opt-in `ollama` profile. It has one
job: turn `sentences.txt` into a `docs` collection served by Lodestar, with no
dependency other than the Python standard library — a demo that installs
packages at start-up is a demo that fails offline.

It is deliberately defensive about a distributed system it does not control:

* **Waiting.** Ollama and the Lodestar server both come up on their own
  schedule, so every call retries with a bounded backoff before giving up.
* **Idempotence.** Creating a collection that already exists is a 409, not an
  error; upserting the same ids again replaces them. Re-running the profile
  converges instead of duplicating.
* **Honesty.** If the model cannot be reached, it says so and exits non-zero so
  `docker compose up` reports a failing service rather than a silent success.

Usage:
    python embed.py --server http://localhost:8080 --ollama http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "sentences.txt"
MODEL = "nomic-embed-text"
COLLECTION = "docs"
BATCH = 32


def call(method: str, url: str, payload: dict | None = None, timeout: int = 60):
    """One HTTP request, returning parsed JSON. Raises on a non-2xx status."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def wait_for(url: str, what: str, attempts: int = 60, delay: float = 2.0) -> None:
    """Block until `url` answers, so container start order does not matter."""
    for attempt in range(attempts):
        try:
            call("GET", url, timeout=5)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise SystemExit(f"gave up waiting for {what} at {url}: {error}")
            time.sleep(delay)


def ensure_model(ollama: str) -> None:
    """Pull the embedding model if this Ollama instance does not have it."""
    try:
        models = call("GET", f"{ollama}/api/tags") or {}
        if any(m.get("name", "").startswith(MODEL) for m in models.get("models", [])):
            return
    except (urllib.error.URLError, OSError):
        pass
    print(f"pulling {MODEL} (this can take a minute)", flush=True)
    request = urllib.request.Request(
        f"{ollama}/api/pull",
        data=json.dumps({"model": MODEL}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if line.strip():
                print(".", end="", flush=True)
    print(flush=True)


def embed(ollama: str, text: str) -> list[float]:
    """Embed one string. Ollama answers `/api/embeddings` with {embedding: [...]}."""
    try:
        return call(
            "POST",
            f"{ollama}/api/embeddings",
            {"model": MODEL, "prompt": text},
            timeout=120,
        )["embedding"]
    except (urllib.error.URLError, OSError, KeyError) as error:
        raise SystemExit(f"embedding failed: {error}")


def sentences() -> list[tuple[str, str]]:
    """Parse `group: text` lines, skipping blanks and comments."""
    out: list[tuple[str, str]] = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        group, sep, text = line.partition(":")
        if sep:
            out.append((group.strip(), text.strip()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="embed the demo corpus")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--ollama", default="http://localhost:11434")
    args = parser.parse_args()

    wait_for(f"{args.server}/healthz", "lodestar")
    wait_for(f"{args.ollama}/api/tags", "ollama")
    ensure_model(args.ollama)

    rows = sentences()
    if not rows:
        raise SystemExit(f"no sentences found in {CORPUS}")

    dim = len(embed(args.ollama, rows[0][1]))
    print(f"{len(rows)} sentences, dim={dim}", flush=True)

    try:
        call("PUT", f"{args.server}/v1/index/{COLLECTION}", {"dim": dim, "metric": "cosine"})
        print(f"created collection {COLLECTION}", flush=True)
    except urllib.error.HTTPError as error:
        if error.code != 409:
            raise
        print(f"collection {COLLECTION} already exists", flush=True)

    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        points = [
            {
                "id": start + offset,
                "vector": embed(args.ollama, text),
                "metadata": {"group": group, "text": text, "line": start + offset},
            }
            for offset, (group, text) in enumerate(chunk)
        ]
        call("POST", f"{args.server}/v1/index/{COLLECTION}/upsert", {"points": points})
        print(f"upserted {start + len(chunk)}/{len(rows)}", flush=True)

    call("POST", f"{args.server}/v1/index/{COLLECTION}/flush", {})
    stats = call("GET", f"{args.server}/v1/index/{COLLECTION}/stats")
    print(f"done: {json.dumps(stats, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
