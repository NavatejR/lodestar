# 6. No authentication on the HTTP API

* Status: accepted
* Date: 2026-09

## Context

The service binds loopback by default and exposes read and write endpoints.
Adding authentication would mean credentials, key rotation, a token store and
a decision about which of the three front doors owns identity — none of which
are decisions a vector engine is well placed to make, and all of which would
be guessed at rather than designed.

## Decision

The service has no authentication and no authorization beyond a `--read-only`
switch. It is built to run **behind a reverse proxy or on a trusted network**,
bound to loopback by default. The proxy terminates TLS and identity; the
service enforces the contract it does own:

* `--read-only` refuses every mutation with `403`, which is how you run a
  search-only replica of a directory someone else writes;
* collection names are validated against a fixed alphabet before they are
  joined onto the data root, so no request can name a path;
* request bodies, points per upsert, queries per search and `k` are bounded;
* CORS is **off** unless `--cors` says otherwise, so a random page cannot read
  responses from a local instance.

## Consequences

* Exposing the port to a network without a proxy exposes the data. That is
  stated in the README and in [SECURITY.md](../../SECURITY.md), not discovered
  in an incident.
* Multi-tenant authorization (per-key scoping of collections) is out of scope;
  a deployment that needs it puts a proxy in front that understands its own
  tenancy model.
* The `read-only` flag is the only operational identity the service has, and
  it is visible in `/healthz` and in a `lodestar_read_only` gauge.

## Alternatives considered

* **API keys in a header** — a credential the service would have to store,
  rotate and log, with no way to revoke it short of a restart, and a false
  sense of coverage for anything that can reach the port.
* **mTLS** — the right answer for service-to-service, but it belongs to the
  deployment, not to the engine.
* **Nothing, not even `--read-only`** — leaves no way to lock a shared
  directory down, which operators reasonably expect.
