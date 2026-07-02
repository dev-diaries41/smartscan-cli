import pytest
import numpy as np
from smartscan.cluster import IncrementalClusterer
from smartscan.cluster.types import Cluster


class TestIncrementalClusterer:
    @pytest.fixture
    def embeddings(self):
        return {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.9, 0.1, 0.0]),
            "c": np.array([0.0, 1.0, 0.0]),
            "d": np.array([0.0, 0.9, 0.1]),
            "e": np.array([0.5, 0.5, 0.0]),
        }

    def test_basic_clustering_integrity(self, embeddings):
        clusterer = IncrementalClusterer()

        result = clusterer.cluster(embeddings)

        ids = list(embeddings.keys())

        assert len(result.assignments) == len(ids)
        for item_id in ids:
            assert item_id in result.assignments
            cid = result.assignments[item_id]
            assert cid in result.clusters
            cluster = result.clusters[cid]
            assert isinstance(cluster, Cluster)
            assert cluster.metadata.prototype_size >= 1

        for cid, cluster in result.clusters.items():
            assigned_items = [i for i, c in result.assignments.items() if c == cid]
            assert cluster.metadata.prototype_size == len(assigned_items)

    def test_nearest_neighbor_assignment(self, embeddings):
        clusterer = IncrementalClusterer()

        ids = ["a", "b", "c", "d"]
        clusterer.cluster({i: embeddings[i] for i in ids})

        assert clusterer.assignments["a"] == clusterer.assignments["b"]
        assert clusterer.assignments["c"] == clusterer.assignments["d"]
        assert clusterer.assignments["a"] != clusterer.assignments["c"]

    def test_incremental_update_counts(self, embeddings):
        clusterer = IncrementalClusterer()

        clusterer.cluster({"a": embeddings["a"]})
        first_cid = clusterer.assignments["a"]
        assert clusterer._counts[first_cid] == 1

        clusterer.cluster({"b": embeddings["b"]})
        second_cid = clusterer.assignments["b"]

        assert first_cid == second_cid
        assert clusterer._counts[first_cid] == 2

    def test_clear_resets_clusterer(self, embeddings):
        clusterer = IncrementalClusterer()
        clusterer.cluster(embeddings)

        assert clusterer.clusters
        assert clusterer.assignments
        assert clusterer._counts

        clusterer.clear()

        assert clusterer.clusters == {}
        assert clusterer.assignments == {}
        assert clusterer._counts == {}

    def test_no_stale_clusters_or_assignments(self, embeddings):
        clusterer = IncrementalClusterer()

        result = clusterer.cluster(embeddings)

        for item_id, cid in result.assignments.items():
            assert cid in result.clusters

        for cid, cluster in result.clusters.items():
            assigned_items = [i for i, c in result.assignments.items() if c == cid]
            assert len(assigned_items) > 0