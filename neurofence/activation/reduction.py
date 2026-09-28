"""PCA-based dimensionality reduction over per-prompt activation feature
vectors, used to visualize/inspect whether groups of prompts separate in
activation space.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from sklearn.decomposition import PCA

MIN_SAMPLES_FOR_PCA = 2


class PCAResult(BaseModel):
    status: str  # "computed" | "skipped"
    skip_reason: str | None = None
    n_components: int = 0
    explained_variance_ratio: list[float] = []
    coordinates: list[list[float]] = []  # one row per input sample


def reduce_dimensionality(x: np.ndarray, n_components: int = 2) -> PCAResult:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        return PCAResult(status="skipped", skip_reason=f"Expected a 2D array, got shape {x.shape}.")

    n_samples, n_features = x.shape
    if n_samples < MIN_SAMPLES_FOR_PCA:
        return PCAResult(
            status="skipped",
            skip_reason=f"Need at least {MIN_SAMPLES_FOR_PCA} samples for PCA, got {n_samples}.",
        )
    if n_features == 0:
        return PCAResult(status="skipped", skip_reason="Feature vectors have zero length.")
    if not np.all(np.isfinite(x)):
        return PCAResult(status="skipped", skip_reason="Input contains NaN/Inf values.")

    effective_components = max(1, min(n_components, n_samples, n_features))

    if np.allclose(x, x[0]):
        # Every sample identical: total variance is 0, so sklearn's
        # explained_variance_ratio_ (variance / total_variance) is a 0/0
        # NaN. There is genuinely no variance to explain; report 0.0 for
        # each component rather than propagating NaN.
        return PCAResult(
            status="computed",
            n_components=effective_components,
            explained_variance_ratio=[0.0] * effective_components,
            coordinates=np.zeros((n_samples, effective_components)).tolist(),
        )

    pca = PCA(n_components=effective_components)
    coords = pca.fit_transform(x)

    return PCAResult(
        status="computed",
        n_components=effective_components,
        explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
        coordinates=coords.tolist(),
    )
