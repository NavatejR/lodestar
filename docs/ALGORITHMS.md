# Algorithms

Three ideas do most of the work: a navigable small-world graph for the common
case, an inverted file with quantized postings for the memory-bound case, and
an exact oracle to check both against.

## Measuring before optimising

Recall@k is only meaningful when the top-k is a meaningful question. On a
corpus where the 10th and 100th nearest neighbours differ by ~1% in distance,
recall mostly measures which side of a tie the search landed on, and no
implementation scores well: on such data, with `m=16, ef_construction=200,
ef_search=64`, hnswlib measures 0.64 and Lodestar 0.67. Both exceed 0.94 on the
corpus used by the gate, which plants the structure real embeddings have.

The gate's corpus therefore plants structure the way real embeddings have it:
40 clusters in **8 intrinsic dimensions**, mapped into 64 ambient dimensions by
a fixed random projection, so distances stay separated and the ranking is
determined by the data rather than by rounding. 20,000 vectors, 200 held-out
queries, seeds fixed. `recall@10` is computed against exact brute force over
the same rows — the oracle is `lodestar_ann_core::brute`, and the benchmark
harness re-derives it independently in numpy so the oracle itself has an
oracle.

## HNSW — the default index

Hierarchical Navigable Small World graphs: nodes are assigned a random maximum
level, and a node at level *l* links only to other nodes at level *l*. Search
enters at the topmost non-empty level, greedily descends, and runs a
best-first beam search with width `ef` on the bottom level.

| parameter | default | effect |
|---|---:|---|
| `m` | 16 | links per node above the base layer — memory and navigability |
| `m0` | 32 | links per node on the base layer (usually `2m`) |
| `ef_construction` | 200 | beam width while inserting — build time, quality |
| `ef_search` | 64 | beam width while searching — recall, latency |
| `seed` | fixed | level assignment, hence the graph itself |

**Neighbour selection** follows Algorithm 4 of the paper: keep a candidate
only when it is closer to the query point than to every already-selected
neighbour. There is deliberately **no top-up pass** to refill the list to its
budget, and that was measured rather than assumed — a rejected candidate is one
an existing link already covers, so refilling puts near-duplicates back and
washes out the long-range links that make the graph navigable. On the gate
corpus the top-up cost **13.5 points** of recall@10 at `ef_search=64`: 0.821
with it, **0.956** without. The measured mean degree at level 0 is ~12 against
a budget of 32 — diversity beats degree.

Skipping the top-up cannot strand a node either: the heuristic accepts the
first candidate unconditionally and candidates arrive in increasing distance,
so every insert is connected to at least its nearest neighbour.

**Deletes** do not touch the graph. A tombstone marks the id dead and searches
skip it; the links remain, which costs a little recall on stale structures and
buys an O(1) delete. `compact` rebuilds the graph without the dead nodes, and
that is when the cost is paid back.

## IVF — inverted files

Coarse quantization partitions the corpus into `nlist` lists with k-means
(Lloyd, seeded, k-means++ init). A query finds the `nprobe` nearest centroids
and scans only those lists.

* **`ivf-flat`** stores full float32 postings. On the gate corpus it reaches
  **1.0000** recall at `nprobe=16` — it is exact within the probed lists, so
  the only error is which lists were probed.
* **`ivf-pq`** stores product-quantized codes instead, cutting a 128-d float32
  vector from 512 bytes to `pq_m` bytes (8 with the default, a 64× reduction).

Choose `nlist` between `sqrt(n)` and `16*sqrt(n)`; the config clamps values
rather than rejecting them, so `nprobe = usize::MAX` is a supported way to ask
for an exhaustive scan.

## Product quantization

Split the vector into `pq_m` subvectors, train an independent codebook of up to
256 centroids per subspace (k-means++ init, Lloyd iterations, seeded), and
replace each subvector with one byte. Training caps at `train_limit` rows
chosen by a deterministic shuffle, because k-means on the whole corpus costs
more than it buys — and determinism keeps the trained index reproducible.

**Asymmetric distance computation** builds a lookup table per query: for each
subspace and each of the 256 possible codes, the distance from the query's
subvector to that centroid. Scoring a code is then `pq_m` table lookups and
adds — no decompression, no floating-point multiply over the corpus.

Two consequences worth knowing:

* **ADC is an estimate.** It sums per-subspace terms that ignore
  cross-subspace interaction, so its ranking degrades. Measured on the gate
  corpus: **0.5945** recall by ADC alone.
* **Reranking fixes it.** Keep the top candidates by ADC, decode those few,
  and recompute exact distances: **0.9895**. The cost is proportional to the
  rerank depth, not to the corpus.

PQ rejects inner product (the additive estimate has no useful ordering for a
max-similarity metric) and reports cosine distances as `adc * 0.5`, which puts
the ADC estimate on the same `1 - cos` scale as the rest of the engine.

## Scalar quantization (SQ8)

Per-dimension min/scale, one byte per component, values clamped to the training
range. The round-trip error is bounded by half a step by construction, and the
tests assert exactly that, including for degenerate constant dimensions.

## Distance kernels

`l2_squared` and `inner_product` have three implementations each — NEON on
aarch64, AVX2+FMA on x86_64, and a scalar fallback — plus batch variants that
compute a whole row of distances into a caller-provided buffer. Selection
happens **once**, at index construction, through a function pointer resolved by
feature detection; there is no feature test inside the loop.

Cosine is not a fourth kernel. Vectors are L2-normalised on the way in and
cosine distance is `1 - inner_product`; inner product is `-inner_product`. One
kernel, three orderings. A zero vector is left unnormalised rather than turned
into NaN.

## Filters

The filter language (`group == "news" AND score >= 0.8`, `tag IN [...]`) is
parsed and compiled **before** traversal into a bitset over node ids. The graph
walk then tests one bit per candidate instead of calling back into user code
per node — which matters because a callback in the inner loop would dominate
the distance computation it wraps.

Filtering interacts with the beam in the only way it can: non-matching nodes do
not enter the result set but *do* remain traversable, so a filter that matches
almost nothing widens the search rather than returning an empty list.

## Determinism

Seeds are explicit everywhere — level assignment, k-means init, training
sampling, the corpus generator — and no hash map iteration order reaches a
decision. Same seed, same data, same index, same results, on any platform the
CI runs. That is what makes a recall *floor* a regression test rather than a
vibe; see [BENCHMARKS.md](../BENCHMARKS.md) for the measured values and
[`crates/index/tests/recall_gate.rs`](../crates/index/tests/recall_gate.rs)
for the floors themselves.
