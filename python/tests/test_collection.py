"""Tests for the durable Collection bindings."""

import numpy as np
import pytest

import lodestar


@pytest.fixture()
def root(tmp_path):
    return str(tmp_path / "data")


def vector_for(id, dim=8):
    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = id
    vector[1] = id % 7
    vector[2] = 1.0
    return vector


def test_create_add_search_round_trip(root):
    collection = lodestar.Collection.create(root, "docs", dim=8, metric="l2")
    ids = np.arange(200, dtype=np.uint64)
    vectors = np.stack([vector_for(int(id)) for id in ids])
    assert collection.add_batch(vectors, ids) == 200
    assert collection.len == 200

    hits = collection.search(vector_for(42), k=1, ef=64)
    assert hits[0][0] == 42
    assert hits[0][1] < 1e-5

    reopened = lodestar.Collection.open(root, "docs")
    assert reopened.len == 200
    hits = reopened.search(vector_for(42), k=1, ef=64)
    assert hits[0][0] == 42


def test_open_or_create_is_idempotent(root):
    first = lodestar.Collection.open_or_create(root, "either", dim=4)
    second = lodestar.Collection.open_or_create(root, "either", dim=4)
    assert first.name == second.name == "either"


def test_create_twice_is_an_error(root):
    lodestar.Collection.create(root, "once", dim=4)
    with pytest.raises(ValueError):
        lodestar.Collection.create(root, "once", dim=4)


def test_open_missing_collection_is_an_error(root):
    with pytest.raises(ValueError):
        lodestar.Collection.open(root, "nope")


def test_deletes_are_durable_across_reopen(root):
    collection = lodestar.Collection.open_or_create(root, "deletes", dim=8)
    ids = np.arange(100, dtype=np.uint64)
    collection.add_batch(np.stack([vector_for(int(id)) for id in ids]), ids)

    collection.remove(7)
    collection.remove(70)
    assert collection.len == 98

    reopened = lodestar.Collection.open(root, "deletes")
    assert reopened.len == 98
    hits = reopened.search(vector_for(7), k=3, ef=64)
    assert all(hit_id != 7 for hit_id, _ in hits)


def test_flush_seals_a_segment_and_verify_passes(root):
    collection = lodestar.Collection.open_or_create(root, "flushed", dim=8)
    ids = np.arange(150, dtype=np.uint64)
    collection.add_batch(np.stack([vector_for(int(id)) for id in ids]), ids)

    sealed = collection.flush()
    assert sealed is not None and sealed.endswith(".seg")

    # The tail is empty now, so a second flush has nothing to seal.
    assert collection.flush() is None

    collection.verify()
    reopened = lodestar.Collection.open(root, "flushed")
    reopened.verify()
    assert reopened.len == 150


def test_flush_with_nothing_to_seal_keeps_tombstones(root):
    # The regression behind fix(store): deletes of ids in sealed segments,
    # then a flush with an empty tail, used to resurrect the ids on reopen.
    collection = lodestar.Collection.open_or_create(root, "tombstones", dim=8)
    ids = np.arange(100, dtype=np.uint64)
    collection.add_batch(np.stack([vector_for(int(id)) for id in ids]), ids)
    collection.flush()

    collection.remove(3)
    collection.remove(93)
    collection.flush()  # nothing to seal: the fixed path

    reopened = lodestar.Collection.open(root, "tombstones")
    assert reopened.len == 98
    hits = reopened.search(vector_for(93), k=3, ef=64)
    assert all(hit_id != 93 for hit_id, _ in hits)


def test_compaction_drops_tombstones_and_keeps_answers(root):
    collection = lodestar.Collection.open_or_create(root, "compact", dim=8)
    ids = np.arange(120, dtype=np.uint64)
    collection.add_batch(np.stack([vector_for(int(id)) for id in ids]), ids)
    collection.flush()

    collection.remove(0)
    collection.remove(60)
    collection.compact()

    reopened = lodestar.Collection.open(root, "compact")
    assert reopened.len == 118
    assert all(
        hit_id != 60 for hit_id, _ in reopened.search(vector_for(60), k=3, ef=64)
    )
    reopened.verify()


def test_reinserting_replaces_the_old_copy(root):
    collection = lodestar.Collection.open_or_create(root, "replace", dim=8)
    collection.add(vector_for(1), id=1)
    collection.flush()

    moved = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    collection.add(moved, id=1)
    hits = collection.search(moved, k=1, ef=64)
    assert hits[0][0] == 1
    assert hits[0][1] < 1e-5


def test_wrong_dimensions_raise(root):
    collection = lodestar.Collection.open_or_create(root, "dims", dim=8)
    with pytest.raises(ValueError, match="dimensions"):
        collection.search(np.ones(3, dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="dimensions"):
        collection.add(np.ones(3, dtype=np.float32), id=1)


def test_metric_and_dim_getters(root):
    collection = lodestar.Collection.create(root, "meta", dim=12, metric="cosine")
    assert collection.dim == 12
    assert collection.metric == "cosine"
    assert collection.name == "meta"
