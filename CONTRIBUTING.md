# Contributing

Thanks for looking. Lodestar is a from-scratch engine, so most contributions
are one of three kinds: fixing something the tests did not catch, making a
measurement, or widening a surface without widening the engine.

## Getting started

```bash
git clone https://github.com/NavatejR/lodestar && cd lodestar
make setup        # toolchain components + a full build
make test         # every Rust suite
make test-py      # the Python suite (needs python3 and a virtualenv)
make gate         # the recall gate, release mode
make verify       # all of the above plus lint: what CI runs
```

Rust 1.85 or newer (see `rust-toolchain.toml`), Python 3.9+ for the bindings,
Docker only for the demo.

## Ground rules

* **The recall gate is the contract.** If your change moves it, say so in the
  pull request with numbers, and update the floor deliberately — never to make
  a failing build pass. A lower floor with a reason in the commit message is
  an acceptable outcome; a lower floor without one is not.
* **Measurements beat intuitions.** For performance and recall, include the
  machine, the corpus, the seed and the before/after numbers. `make bench`
  prints the environment for exactly this reason.
* **A bug gets a test first.** If you can write the failing assertion, write
  it; the diff that follows is then evidence rather than opinion. The crash
  test and the property tests exist because someone got surprised once.
* **Determinism is a feature.** Same seed, same data, same results. Do not
  introduce an unseeded RNG, a hash-map iteration order in a decision, or a
  platform-specific branch that changes answers.
* **Unsafe code has to argue for itself.** The crates `forbid(unsafe_op_in_unsafe_fn)`
  and document what each block relies on. Most code does not need it: the SIMD
  kernels are the existing exception, and they are checked against the scalar
  implementation in tests.

## Style

* `cargo fmt` is not a suggestion; `make fmt-check` runs in CI.
* `cargo clippy -- -D warnings`. Lint suppressions need a comment naming the
  lint and why the code is right.
* Doc comments on everything public — `missing_docs` is a warning in every
  crate. Explain *why* in prose where the code cannot; the good comments in
  this repository are the ones that record a measurement or a rejected
  alternative.
* Error messages are for the person who receives them: name the object, say
  what was expected, and never `unwrap()` on user input.
* Commit messages explain the reason, not the diff.

## Pull requests

1. One change per PR. A refactor and a behaviour change in the same diff
   cannot be reviewed or reverted independently.
2. Describe the *why*, and for anything measurable, the numbers.
3. `make verify` must be green.
4. New endpoints belong in the OpenAPI document, the route table in
   `crates/server/src/lib.rs`, and the routing test — all three, or the
   "every documented route is reachable" test will tell you.
5. A change to the segment or log format needs an ADR in `docs/adr/` and a
   version bump discussion, because it is the one thing that cannot be
   rewritten in place.

## Where to start

* `make gate` failures and open issues labelled `good first issue`.
* The fuzz targets (`fuzz/`) are the cheapest place to find real bugs: extend
  a dictionary, add a grammar, run it for an hour. Each target ships seed
  inputs in `fuzz/seed/` — copy them into `fuzz/corpus/<target>/` before a
  long run, so mutation starts from files the format actually produces.
* Documentation that is wrong is a bug — especially benchmarks, which go
  stale silently.

## Releases

Maintainers only: bump `version` in the workspace `Cargo.toml`, update
`CHANGELOG.md`, tag `vX.Y.Z`, and let CI publish the crates, the wheel and the
container image from the tag.
