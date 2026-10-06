"""Lodestar: vector search from scratch.

Python bindings over the Rust engine. `Index` keeps vectors in memory,
`Collection` persists them with write-ahead-logged, crash-safe semantics,
and `sample` generates synthetic clustered corpora for demos and tests.
"""

from lodestar_native import Collection, Index, __version__, sample

__all__ = ["Collection", "Index", "__version__", "sample"]
