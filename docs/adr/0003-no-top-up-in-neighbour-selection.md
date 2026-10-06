# 3. No top-up in neighbour selection

* Status: accepted
* Date: 2026-09

## Context

Algorithm 4 of the HNSW paper selects neighbours with a heuristic that rejects
a candidate when an already-selected neighbour is closer to it than the query
is. Implementations commonly then refill the list to its budget from the
rejected candidates — a "top-up". Our first implementation did too, and its
recall plateaued below where the reference implementation sat.

## Decision

Do not top up. The heuristic runs to the end of the candidate list and
whatever it keeps is the link set.

The decision was measured on the recall-gate corpus (20k vectors, `m=16`,
`m0=32`, `ef_construction=200`), recall@10 at `ef_search=64`:

| neighbour selection | recall@10 |
|---|---:|
| heuristic **with** top-up | 0.821 |
| heuristic **without** top-up | **0.956** |

The measured mean level-0 degree is about 12 links against a budget of 32.

## Consequences

* A rejected candidate is one an existing link already covers; refilling the
  list reinstates near-duplicates and evicts the long-range links that make the
  graph navigable in the first place. Fewer, more diverse links win by 13.5
  recall points.
* Under-filled link lists are expected, not a bug — degree is not the
  objective.
* No node can be stranded: the heuristic accepts the first candidate
  unconditionally and candidates arrive in increasing distance, so every insert
  connects to at least its nearest neighbour.

## Alternatives considered

* **Top-up to the budget** — measured, worse, reverted. This record exists so
  that the "obvious" fix is not re-applied by a future contributor who finds a
  node with 12 links and a budget of 32.
* **A different heuristic entirely (e.g. relative neighbourhood graph)** — a
  larger change with its own recall profile; revisit only if the gate moves.
