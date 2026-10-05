# Lodestar — common developer entry points.
#
# Everything a reviewer needs is reachable from here: `make setup && make test`
# from a fresh clone, `make bench` for the benchmark report, `make demo` for the
# Docker demo. Targets are deliberately thin wrappers so that CI and humans run
# exactly the same commands.

SHELL := /bin/bash
CARGO ?= cargo
PYTHON ?= python3
VENV ?= .venv

# The Python bindings are an `extension-module`: on macOS their Python symbols
# are resolved when Python loads the module, so no test executable can link
# them. They are verified by `make test-py` (maturin + pytest) instead, and
# excluded from the cargo targets that link binaries. `clippy` and `doc` still
# cover the crate, because neither links anything.
RUST_EXCLUDE ?= --exclude lodestar-ann-py

.DEFAULT_GOAL := help
.PHONY: help setup build release test test-core test-index test-store test-py lint fmt fmt-check clippy doc wheel venv bench bench-full gate demo docker clean verify

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install toolchain components and build everything once
	rustup component add rustfmt clippy || true
	$(CARGO) build --workspace --all-targets $(RUST_EXCLUDE)

build: ## Debug build of the workspace
	$(CARGO) build --workspace $(RUST_EXCLUDE)

release: ## Optimised build of the CLI and server
	$(CARGO) build --release --bin lodestar

test: ## Run every Rust test suite in the workspace
	$(CARGO) test --workspace $(RUST_EXCLUDE)

test-core: ## Run only the core crate tests
	$(CARGO) test -p lodestar-ann-core

test-index: ## Run only the index crate tests
	$(CARGO) test -p lodestar-ann-index

test-store: ## Run only the storage crate tests (includes the crash test)
	$(CARGO) test -p lodestar-ann-store

test-py: venv ## Build the wheel and run the Python test suite
	$(VENV)/bin/pip install -q maturin pytest
	$(VENV)/bin/maturin develop --manifest-path crates/py/Cargo.toml
	cd python && ../$(VENV)/bin/pytest -q

lint: fmt-check clippy ## Run formatting and lint gates
fmt: ## Format the workspace
	$(CARGO) fmt --all

fmt-check: ## Verify formatting without changing files
	$(CARGO) fmt --all -- --check

clippy: ## Lint with warnings denied
	$(CARGO) clippy --workspace --all-targets -- -D warnings

doc: ## Build documentation with warnings denied
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps

wheel: venv ## Build the Python wheel into dist/
	$(VENV)/bin/pip install -q maturin
	$(VENV)/bin/maturin build --release --manifest-path crates/py/Cargo.toml --out dist

venv: ## Create the local Python virtual environment
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

bench: ## Run the benchmark suite (synthetic + bundled corpus)
	$(VENV)/bin/python bench/run.py --suite standard

bench-full: ## Run the full benchmark suite including downloaded datasets
	$(VENV)/bin/python bench/run.py --suite full

gate: ## Recall regression gate used by CI
	$(CARGO) test --release -p lodestar-ann-index --test recall_gate -- --ignored --nocapture

demo: ## Start the Docker demo on http://localhost:8080
	docker compose up --build

clean: ## Remove build and benchmark artefacts
	$(CARGO) clean
	rm -rf dist .criterion

verify: lint test gate ## Everything a reviewer should run before merging
	@echo "verification complete"
