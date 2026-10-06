"""Type stubs for the Lodestar bindings."""

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: Final[str]

Float32 = NDArray[np.float32]


class Index:
    """An in-memory HNSW index."""

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        m: int = 16,
        m0: int = 32,
        ef_construction: int = 200,
        seed: int | None = None,
    ) -> None: ...

    @property
    def dim(self) -> int: ...
    @property
    def metric(self) -> str: ...
    @property
    def len(self) -> int: ...
    @property
    def node_count(self) -> int: ...
    @property
    def memory_bytes(self) -> int: ...

    def add(self, vector: ArrayLike, id: int) -> None:
        """Insert one vector; re-inserting an id replaces its vector."""
    def add_batch(self, vectors: ArrayLike, ids: ArrayLike) -> int:
        """Insert an (n, dim) float32 array with an (n,) uint64 id array."""
    def remove(self, id: int) -> bool:
        """Tombstone an id; True when a live vector was removed."""
    def search(
        self, vector: ArrayLike, k: int, ef: int | None = None
    ) -> list[tuple[int, float]]:
        """The k nearest neighbours as (id, distance) pairs."""
    def compact(self) -> "Index":
        """A copy with tombstoned nodes physically removed."""


class Collection:
    """A durable, searchable collection of vectors on disk."""

    @classmethod
    def create(cls, root: str, name: str, dim: int, metric: str) -> "Collection": ...
    @classmethod
    def open(cls, root: str, name: str) -> "Collection": ...
    @classmethod
    def open_or_create(
        cls, root: str, name: str, dim: int, metric: str = "l2"
    ) -> "Collection": ...

    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def metric(self) -> str: ...
    @property
    def len(self) -> int: ...

    def add(self, vector: ArrayLike, id: int) -> None:
        """Durable insert; re-inserting an id replaces its vector."""
    def add_batch(self, vectors: ArrayLike, ids: ArrayLike) -> int:
        """Durable batch insert, synced once."""
    def remove(self, id: int) -> None:
        """Durable delete."""
    def search(
        self, vector: ArrayLike, k: int, ef: int | None = None
    ) -> list[tuple[int, float]]:
        """The k nearest neighbours as (id, distance) pairs."""
    def flush(self) -> str | None:
        """Seal the in-memory tail; the new segment's path, or None."""
    def compact(self) -> str | None:
        """Rewrite every live vector into one segment; its path, or None."""
    def verify(self) -> None:
        """Run the full checksum pass; raises ValueError on corruption."""


def sample(
    count: int,
    dim: int,
    metric: str = "l2",
    clusters: int = 16,
    seed: int = 7,
) -> tuple[list[int], Float32]:
    """A synthetic clustered corpus: (ids, (count, dim) float32 vectors)."""
