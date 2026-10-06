# The demo image: one static-ish binary pair, one data volume, no toolchain at
# runtime.
#
# Build from the repository root:
#   docker build -f demo/Dockerfile -t lodestar:dev .
#
# The builder is a `rust:1` image rather than a pinned patch release so that a
# rebuild picks up current security fixes; the lockfile pins the dependencies,
# which is where reproducibility actually comes from.

FROM rust:1-bookworm AS builder

WORKDIR /src
# Dependencies first, so editing a source file does not re-download the world.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin lodestar --bin lodestar-server

FROM debian:bookworm-slim AS runtime

# curl is not for the service; it is for the compose healthcheck and for poking
# the API from inside the container.
RUN apt-get update \
 && apt-get install --yes --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/lodestar /usr/local/bin/lodestar
COPY --from=builder /src/target/release/lodestar-server /usr/local/bin/lodestar-server
COPY demo/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# The data root is the only thing worth persisting, and `--root` is the only
# flag that matters to a demo. Everything else has a default; the compose file
# overrides what it needs to.
ENV LODESTAR_ROOT=/data \
    LODESTAR_ADDR=:8080 \
    LODESTAR_LOG=info
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8080/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["lodestar-server"]
