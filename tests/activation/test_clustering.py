from __future__ import annotations

import numpy as np

from neurofence.activation.clustering import cluster_activations


def test_two_well_separated_groups_cluster_separately() -> None:
    # Regression: the eps heuristic used to be the median of *all* pairwise
    # distances, which is dominated by inter-group distances whenever a
    # minority cluster sits far from a majority cluster (100 cross-group
    # pairs vs. 90 within-group pairs here) -- eps ended up so large that
    # DBSCAN merged both groups into one cluster, exactly the scenario
    # this function exists to detect correctly.
    rng = np.random.default_rng(0)
    group_a = rng.normal(loc=0.0, scale=0.1, size=(10, 5))
    group_b = rng.normal(loc=50.0, scale=0.1, size=(10, 5))
    x = np.vstack([group_a, group_b])

    result = cluster_activations(x, min_samples=3)

    assert result.status == "computed"
    labels = np.array(result.labels)
    assert labels[:10].tolist().count(labels[0]) == 10  # group A all one label
    assert labels[10:].tolist().count(labels[10]) == 10  # group B all one label
    assert labels[0] != labels[10]


def test_single_tight_cluster_no_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=0.01, size=(10, 4))
    result = cluster_activations(x, min_samples=3)
    assert result.status == "computed"
    assert result.n_clusters == 1
    assert result.noise_count == 0


def test_too_few_samples_skipped() -> None:
    result = cluster_activations(np.zeros((2, 4)), min_samples=3)
    assert result.status == "skipped"


def test_zero_samples_skipped() -> None:
    result = cluster_activations(np.zeros((0, 4)), min_samples=3)
    assert result.status == "skipped"


def test_1d_input_skipped() -> None:
    result = cluster_activations(np.array([1.0, 2.0, 3.0]))
    assert result.status == "skipped"


def test_nan_input_skipped() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    result = cluster_activations(x, min_samples=2)
    assert result.status == "skipped"


def test_explicit_eps_respected() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3))
    result = cluster_activations(x, eps=0.5, min_samples=3)
    assert result.status == "computed"
    assert result.parameters["eps"] == 0.5


def test_identical_points_no_crash() -> None:
    x = np.ones((10, 4))
    result = cluster_activations(x, min_samples=3)
    assert result.status == "computed"
    assert result.n_clusters == 1
