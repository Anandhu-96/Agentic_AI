"""Tests for the centroid-based worker tracker."""

from isip.vision.tracking import WorkerTracker


def test_stable_ids_for_consistent_workers():
    tracker = WorkerTracker()
    f1 = tracker.update([(0.5, 0.5), (0.2, 0.2)])
    f2 = tracker.update([(0.501, 0.502), (0.198, 0.201)])
    assert f1 == f2  # tiny movement reuses the same IDs


def test_new_position_gets_new_id():
    tracker = WorkerTracker()
    ids1 = tracker.update([(0.5, 0.5)])
    ids2 = tracker.update([(0.5, 0.5), (0.9, 0.9)])  # second worker appears
    assert ids2[0] == ids1[0]      # original worker keeps its ID
    assert len(set(ids2)) == 2     # new worker got a different ID


def test_leaves_and_reenters_reuses_id():
    tracker = WorkerTracker()
    first = tracker.update([(0.5, 0.5)])[0]
    tracker.update([])              # worker gone for a frame
    later = tracker.update([(0.51, 0.49)])[0]
    assert later == first


def test_far_apart_workers_do_not_cross_match():
    tracker = WorkerTracker(match_distance=0.3)
    ids1 = tracker.update([(0.5, 0.5), (0.6, 0.6)])
    ids2 = tracker.update([(0.6, 0.6), (0.5, 0.5)])  # swapped order
    assert set(ids2) == set(ids1)