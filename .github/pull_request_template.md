## What changed

<!-- Why the change exists, and what now works that did not before. -->

## How it was verified

<!-- Commands run and their outcomes: `make verify`, `make gate`,
     `make test-py`, `cargo +nightly fuzz run <target>` — whatever applies. -->

## Recall / performance impact

<!-- Delete this section if the change cannot reach the search path. For
     algorithm changes, include before/after recall from `make gate` or a
     `make bench SUITE=quick` comparison. -->

## Checklist

- [ ] `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass
- [ ] Tests cover the changed behaviour (or no behaviour changed)
- [ ] `README.md` / `docs/` / `CHANGELOG.md` reflect the change
