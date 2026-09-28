"""DBSCAN clustering over per-prompt activation feature vectors, used to
check whether a group of prompts (e.g. candidate-trigger prompts) forms a
separate cluster from a baseline group in activation space.

DBSCAN (rather than k-means) is used because we do not know the number of
clusters in advance and want an explicit "noise"/outlier label rather than
forcing every point into some cluster -- appropriate when most prompts are
expected to look similar and only a few (if any) are expected to diverge.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

MIN_SAMPLES_FOR_CLUSTERING = 3


class ClusteringResult(BaseModel):
    status: str  # "computed" | "skipped"
    skip_reason: str | None = None
    algorithm: str = "dbscan"
    parameters: dict[str, float] = {}
    labels: list[int] = []  # -1 = noise/outlier
    n_clusters: int = 0
    noise_count: int = 0


def _estimate_eps(x: np.ndarray, k: int) -> float:
    """Heuristic default: 90th percentile of k-th-nearest-neighbor
    distances (a simple version of the standard "k-distance graph" DBSCAN
    eps heuristic).

    Deliberately NOT the median of all pairwise distances: that statistic
    is dominated by *inter*-group distances whenever multiple tight
    clusters sit far apart (exactly the case this module exists to
    detect -- a small trigger-prompt cluster far from a larger baseline
    cluster), which pushes eps up until DBSCAN merges every group into
    one. k-distance measures local density instead, so it stays small
    regardless of how far apart separate clusters are.

    The 90th percentile (rather than the median) of k-distances is used so
    most points' k-th neighbor falls within eps and qualifies as a DBSCAN
    core point -- the median leaves ~half of points just outside eps by
    construction, which manifests as spurious noise-labeled points even
    within an obviously uniform, tight cluster.
    """
    from scipy.spatial.distance import squareform

    distances = squareform(pdist(x, metric="euclidean"))
    n = distances.shape[0]
    k = max(1, min(k, n - 1))
    kth_distances = np.sort(distances, axis=1)[:, k]  # column 0 is self (0.0)
    finite = kth_distances[np.isfinite(kth_distances)]
    if finite.size == 0 or np.all(finite == 0):
        return 1.0
    return float(np.percentile(finite, 90)) or 1.0


def cluster_activations(
    x: np.ndarray,
    eps: float | None = None,
    min_samples: int = MIN_SAMPLES_FOR_CLUSTERING,
) -> ClusteringResult:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        return ClusteringResult(
            status="skipped", skip_reason=f"Expected a 2D array, got shape {x.shape}."
        )

    n_samples = x.shape[0]
    if n_samples < min_samples:
        return ClusteringResult(
            status="skipped",
            skip_reason=f"Need at least {min_samples} samples for clustering, got {n_samples}.",
        )
    if not np.all(np.isfinite(x)):
        return ClusteringResult(status="skipped", skip_reason="Input contains NaN/Inf values.")

    resolved_eps = eps if eps is not None else _estimate_eps(x, k=min_samples)
    labels = DBSCAN(eps=resolved_eps, min_samples=min_samples).fit_predict(x)

    unique_labels = set(labels.tolist())
    n_clusters = len({label for label in unique_labels if label != -1})
    noise_count = int(np.count_nonzero(labels == -1))

    return ClusteringResult(
        status="computed",
        parameters={"eps": resolved_eps, "min_samples": float(min_samples)},
        labels=[int(label) for label in labels],
        n_clusters=n_clusters,
        noise_count=noise_count,
    )
