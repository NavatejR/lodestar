# Getting Started with Lodestar

*A beginner's guide to vector search — no prior experience required.*

The [README](README.md) explains what Lodestar is and how fast it is. This
guide explains how to actually use it, starting from "what is a vector, and
why would I search one?" If anything here feels unclear, that's a bug in the
guide — please [open an issue](https://github.com/NavatejR/lodestar/issues).

---

## 1. What is vector search?

A **vector** is just a list of numbers, like `[0.2, 0.9, 0.1, 0.4]`. The trick
is that *things* — sentences, photos, songs, products — can be turned into
vectors in a way that keeps their meaning: two sentences that say similar
things end up as vectors that are numerically close together. (The turning is
called **embedding**, and it's done by machine-learning models like Ollama or
OpenAI's — Lodestar doesn't do this part; it assumes you already have vectors.)

**Vector search** answers: *given this vector, which of my stored vectors are
closest to it?* In practice that means "find me the most similar things":

- paste a question → find the most relevant paragraphs of your notes
- paste a photo's vector → find visually similar products
- find duplicate or near-duplicate items in a catalogue

Comparing your query against every stored vector (exact search) is simple but
slow at scale. Lodestar builds an **approximate** search structure — a graph
of "neighbours" called **HNSW** — that finds almost exactly the same results
hundreds of times faster. "Almost" is measured as **recall** (1.0 = same
answers as exact search; Lodestar's default settings reach 0.99+).

### Words you'll see in this guide

| Word | Meaning |
|---|---|
| **dimension** | How many numbers are in each vector. All vectors in one collection must share it. |
| **collection** | A named, saved set of vectors on disk (like a database table). |
| **metric** | How "closeness" is measured: `l2` (straight-line distance), `cosine` (angle — most common for text), `inner_product`. |
| **ef** | Effort knob for searching: higher = better recall but slower. Defaults are fine. |
| **flush** | Save the recent in-memory writes into a permanent, indexed file on disk. |
| **compact** | Rewrite the collection to reclaim space after deletions. |
| **verify** | Check every stored byte against its checksum — answers "is my data intact?" |

---

## 2. What you need

- A Mac, Linux machine, or Windows with WSL.
- **Rust**, which compiles Lodestar. Install it in one command:

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  (Close and reopen your terminal afterwards so `cargo` is on your PATH.)

- *Optional:* **Python 3.9+** for the Python bindings, and **Docker Desktop**
  for the one-command demo.

> Windows note: Rust runs fine on Windows, but Lodestar is developed and
> tested on macOS/Linux. WSL2 is the smoothest Windows experience.

## 3. Set up (two commands)

```bash
git clone https://github.com/NavatejR/lodestar && cd lodestar
make setup
```

`make setup` builds everything and takes a few minutes the first time (after
that, rebuilds are fast). When it finishes without red text, you're ready.

---

## 4. Your first vector search (5 minutes, no ML involved)

We'll build a tiny collection of 4-number vectors you can see and reason
about. Every command below is copy-pasteable.

**1. Create a collection** — a folder on disk (under `./data` by default)
that will hold our vectors:

```bash
cargo run --quiet --release -p lodestar-ann-cli -- create intro --dim 4
```

Yes, that's long. Later you can `cargo install --path crates/cli` once, which
puts a real `lodestar` command on your PATH, and from then on it's just
`lodestar create intro --dim 4`. We'll use `cargo run --quiet --release -p
lodestar-ann-cli --` (in a box: **`LODESTAR`**) below for clarity.

**2. Put some vectors in.** Create a file `vectors.jsonl` with one JSON
object per line — an `id` (any whole number) and a `vector` (4 numbers,
matching `--dim 4`):

```bash
cat > vectors.jsonl <<'EOF'
{"id": 1, "vector": [1.0, 0.0, 0.0, 0.0]}
{"id": 2, "vector": [0.9, 0.1, 0.0, 0.0]}
{"id": 3, "vector": [0.0, 1.0, 0.0, 0.0]}
{"id": 4, "vector": [0.0, 0.0, 0.0, 1.0]}
EOF
```

**3. Insert them**, and seal them into the on-disk index:

```bash
LODESTAR insert intro --file vectors.jsonl --flush
```

**4. Search.** "Which stored vectors are closest to `[1, 0, 0, 0]`?"

```bash
LODESTAR search intro --vector "1.0, 0.0, 0.0, 0.0" --k 2
```

You should see ids `1` and `2` — the two vectors pointing mostly in the
same direction — with id 1 first, because it's *identical* (distance 0.0).
That's vector search: similarity by geometry.

**5. Look around:**

```bash
LODESTAR list                 # your collections
LODESTAR stats intro          # how many vectors, how big on disk
LODESTAR verify intro         # check every byte is intact
```

**Clean up when you're done** (a collection is just a folder):

```bash
LODESTAR delete-collection intro 2>/dev/null || rm -rf data/intro
```

(`delete` on the CLI tombstones *ids inside* a collection; to remove a whole
collection folder, remove its directory.)

### Playing with something bigger

```bash
LODESTAR sample demo --count 20000 --dim 128   # a synthetic 20k-vector corpus
LODESTAR search demo --file query.jsonl --k 5  # or search with a JSONL of queries
LODESTAR stats demo
```

---

## 5. From Python

Lodestar ships real Python bindings (built with PyO3/maturin, fully typed).
This needs Python 3.9+ in addition to Rust:

```bash
make wheel && .venv/bin/pip install dist/*.whl    # build and install the package
```

Now, in a Python session started from this folder (`./.venv/bin/python`):

```python
import numpy as np
import lodestar

# An in-memory index: fastest, dies with the process.
index = lodestar.Index(dim=4, metric="l2")
index.add_batch(
    np.array([[1.0, 0, 0, 0], [0.9, 0.1, 0, 0], [0.0, 1, 0, 0]]),
    np.array([1, 2, 3], dtype=np.uint64),
)
print(index.search(np.array([1.0, 0, 0, 0]), k=2))
# -> [(1, 0.0), (2, 0.009999...)]  — (id, distance) pairs, same as the CLI

# A durable collection: every write goes through a crash-safe log first.
col = lodestar.Collection.create("./data", "from_python", 4, "l2")
col.add_batch(
    np.array([[0.5, 0.5, 0, 0], [0.0, 0.0, 1.0, 0.0]]),
    np.array([10, 11], dtype=np.uint64),
)
col.flush()                       # seal into an immutable, memory-mapped file
print(col.search(np.array([0.5, 0.5, 0, 0]), k=1))
col.verify()                      # raises if any byte on disk is corrupt
```

Why two APIs? `Index` is for experiments and ephemeral data. `Collection`
acknowledges writes only once they're durable — kill the process with SIGKILL
mid-write and everything acknowledged is still there.

---

## 6. The HTTP API and the web console

Start the server (in one terminal):

```bash
make server          # serves on http://127.0.0.1:8080
```

Then open **<http://localhost:8080/demo>** in your browser. It's a small
console — compiled into the server binary, no CDN, no JavaScript framework —
where you can create collections, upsert vectors, search, and watch the
metrics, all against the same engine.

Prefer `curl`? The same things over HTTP:

```bash
# create a collection named "papers"
curl -s localhost:8080/v1/index/papers -X PUT \
  -H 'content-type: application/json' -d '{"dim": 4, "metric": "l2"}'

# upsert vectors *with metadata* (metadata is a server-side feature)
curl -s localhost:8080/v1/index/papers/upsert -X POST \
  -H 'content-type: application/json' -d '{
    "points": [
      {"id": 1, "vector": [1, 0, 0, 0], "metadata": {"topic": "intro"}},
      {"id": 2, "vector": [0, 1, 0, 0], "metadata": {"topic": "math"}}
    ]}'

# search *with a filter*: nearest to [1,0,0,0], but only topic = intro
curl -s localhost:8080/v1/search -X POST \
  -H 'content-type: application/json' \
  -d '{"index": "papers", "query": [1, 0, 0, 0], "k": 5,
       "filter": "topic == intro", "include_metadata": true}'
```

Other addresses worth knowing: `/docs` (human API reference),
`/openapi.json` (machine-readable), `/metrics` (Prometheus), `/healthz`.

---

## 7. One-command demo (Docker)

If you have Docker, this does everything — builds the image, seeds a 20,000
vector corpus, starts the server — and leaves you at the web console:

```bash
make demo           # then open http://localhost:8080/demo
```

Stop it later with `docker compose -f demo/docker-compose.yml down`.

---

## 8. Where to go next

| Curious about… | Read |
|---|---|
| How HNSW works, and what `ef` really does | [docs/ALGORITHMS.md](docs/ALGORITHMS.md) |
| How the storage engine survives crashes | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Running this for real (ops guide) | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| Why some things are the way they are | [docs/adr/](docs/adr) — short decision records |
| The measured numbers | [BENCHMARKS.md](BENCHMARKS.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## 9. Troubleshooting

<details>
<summary><code>cargo: command not found</code></summary>

Rust isn't on your PATH. Install rustup (step 2), then open a new terminal —
or run `source "$HOME/.cargo/env"`.
</details>

<details>
<summary><code>ModuleNotFoundError: No module named 'lodestar'</code></summary>

The Python bindings aren't built/installed. Run `make build-py`, and start
Python as `./.venv/bin/python` from the repository folder.
</details>

<details>
<summary><code>address already in use</code> when starting the server</summary>

Port 8080 is taken. Choose another: `LODESTAR_ADDR=127.0.0.1:8090 make server`.
</details>

<details>
<summary>error mentioning <em>dimensionality</em> or vector length</summary>

Every vector in a collection must have exactly the collection's `--dim`
numbers. Count yours — this is by far the most common first-day error.
</details>

<details>
<summary>Where is my data?</summary>

In `./data/<collection-name>/` (or whatever you passed to `--root`). It's
plain files: segments, a write-ahead log, a manifest. Copy that folder and
you've backed the collection up.
</details>

---

*Vectors don't bite. Start with four dimensions and see where it takes you.*
