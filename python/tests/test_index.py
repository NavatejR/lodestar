"""Tests for the in-memory Index bindings."""

import numpy as np
import pytest

import lodestar


@pytest.fixture(scope="module")
def corpus():
    ids, vectors = lodestar.sample(count=2000, dim=16, metric="l2", clusters=8, seed=11)
    return np.asarray(ids, dtype=np.uint64), vectors


@pytest.fixture()
def index(corpus):
    ids, vectors = corpus
    index = lodestar.Index(dim=16, metric="l2")
    added = index.add_batch(vectors, ids)
    assert added == len(ids)
    return index


def test_module_exports_a_version():
    assert isinstance(lodestar.__version__, str)
    assert lodestar.__version__.count(".") >= 1


def test_exact_vector_query_finds_its_own_id(index, corpus):
    ids, vectors = corpus
    query = vectors[137]
    hits = index.search(query, k=5, ef=64)
    assert len(hits) == 5
    assert hits[0][0] == ids[137]
    assert hits[0][1] < 1e-5


def test_distances_are_ascending(index, corpus):
    _, vectors = corpus
    hits = index.search(vectors[0], k=20, ef=128)
    distances = [distance for _, distance in hits]
    assert distances == sorted(distances)


def test_counters_and_memory(index):
    assert index.len == 2000
    assert index.node_count == 2000
    assert index.memory_bytes > 0
    assert index.dim == 16
    assert index.metric == "l2"


def test_remove_tombstones_and_compact_purges(corpus):
    ids, vectors = corpus
    index = lodestar.Index(dim=16, metric="l2")
    index.add_batch(vectors, ids)
    assert index.remove(int(ids[42]))
    assert not index.remove(int(ids[42]))  # already gone
    assert index.len == 1999
    assert index.node_count == 2000  # the tombstone is still there

    compacted = index.compact()
    assert compacted.len == 1999
    assert compacted.node_count == 1999


def test_reinserting_an_id_replaces_its_vector():
    index = lodestar.Index(dim=2, metric="l2")
    index.add(np.array([1.0, 0.0], dtype=np.float32), id=1)
    index.add(np.array([0.0, 5.0], dtype=np.float32), id=1)
    hits = index.search(np.array([0.0, 5.0], dtype=np.float32), k=1, ef=16)
    assert hits[0][0] == 1
    assert hits[0][1] < 1e-6


def test_wrong_dimensions_raise(index):
    with pytest.raises(ValueError, match="dimensions"):
        index.search(np.ones(3, dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="dimensions"):
        index.add(np.ones(3, dtype=np.float32), id=999)
    with pytest.raises(ValueError, match="dimensions"):
        index.add_batch(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.uint64))


def test_batch_and_id_count_must_agree(index):
    with pytest.raises(ValueError, match="ids"):
        index.add_batch(
            np.zeros((4, 16), dtype=np.float32), np.zeros(3, dtype=np.uint64)
        )


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        lodestar.Index(dim=4, metric="manhattan")


def test_cosine_corpus_sits_on_the_unit_sphere(corpus):
    ids, vectors = lodestar.sample(count=64, dim=8, metric="cosine", clusters=3, seed=3)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_sample_is_reproducible_for_a_seed():
    first = lodestar.sample(count=32, dim=4, clusters=2, seed=9)
    second = lodestar.sample(count=32, dim=4, clusters=2, seed=9)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])


def test_single_vector_index_round_trip():
    index = lodestar.Index(dim=3, metric="cosine")
    index.add(np.array([1, 2, 3], dtype=np.float32), id=1)
    hits = index.search(np.array([1, 2, 3], dtype=np.float32), k=1)
    # Cosine distance is 1 - dot, so the self-match is zero only up to f32 rounding.
    assert [hit[0] for hit in hits] == [1]
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)
