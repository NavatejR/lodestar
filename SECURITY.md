# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Use GitHub's
private vulnerability reporting for
[`NavatejR/lodestar`](https://github.com/NavatejR/lodestar/security/advisories/new),
or email the maintainer at the address in `Cargo.toml` with the subject line
`Lodestar security`.

Include, if you have it:

* the version (`lodestar --version`) and the platform;
* a reproducer — a corpus, a request, or a segment file. A one-line script that
  crashes the process is worth more than a paragraph of description;
* the impact you can demonstrate: crash, untrusted memory read, path escape,
  data loss, or a recall manipulation.

You should get an acknowledgment within a few days and a statement of scope
within two weeks. If a fix is accepted, credit is yours unless you prefer
otherwise, and a GitHub security advisory is published with the release that
carries the fix.

## Threat model, in short

Lodestar is a **single-node engine for a trusted network.** It assumes the
process, the filesystem under `--root`, and the callers are not hostile. It
does *not* assume the network is, which is why the API ships with no
authentication: run it on loopback, or behind a proxy that owns identity and
TLS. See [ADR 6](docs/adr/0006-no-authentication-on-the-http-api.md).

What the engine does defend against:

| risk | defence |
|---|---|
| path traversal via a collection name | names are validated against a fixed alphabet before joining `--root` |
| oversized or malicious requests | bounded body size, points per upsert, queries per search, and `k` |
| cross-origin reads from a web page | CORS is off unless `--cors` says otherwise |
| corrupted or truncated segment files | header/footer CRC32 and structural validation before any offset is trusted |
| a torn write-ahead log | torn tails are truncated on replay; a complete-but-bad record is an error, not a guess |
| reading a format written by a newer version | `FORMAT_VERSION` / `MANIFEST_VERSION` are refused, not parsed optimistically |
| accidental writes against a shared directory | `--read-only` refuses every mutation with `403` |

What it does not defend against:

* **An attacker who can reach the API.** They can read and write every
  collection. Put a proxy in front.
* **A hostile filesystem or a hostile operator.** Segments are checked, not
  signed; anyone who can write the directory can replace it wholesale.
* **Denial of service.** There is no rate limiting. Request sizes are bounded,
  but a determined caller can still keep the blocking pool busy.
* **Side channels.** Distances and timings are not constant-time, and were not
  designed to be. Do not serve secrets through them.

## Dependencies

Dependabot opens pull requests for both the Rust lockfile and the GitHub
Actions used in CI. Runtime dependencies are deliberately few — `axum`,
`tower-http`, `serde`, `tracing`, `clap`, `pyo3`, `numpy` — because every one
of them is on the request path of something that holds data.

## Supported versions

The `main` branch and the latest released tag are supported. This project is
pre-1.0: the segment format may change between minor versions, and such
changes are called out in `CHANGELOG.md` and refused by the reader rather
than silently accepted.
