"""Per-tensor descriptive statistics.

Every statistic here must be defined for degenerate input (empty tensor,
constant tensor, all-NaN/Inf tensor) -- a crash or a silent NaN in a
security scanner is worse than a reported "not computable" status.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats as sp_stats

from neurofence.weight_forensics.robust import median_absolute_deviation


class TensorStatistics(BaseModel):
    shape: list[int]
    dtype: str
    element_count: int
    status: str  # "ok" | "empty" | "all_non_finite"
    notes: list[str] = []

    nan_count: int = 0
    inf_count: int = 0
    finite_count: int = 0

    mean: float | None = None
    std: float | None = None
    variance: float | None = None
    median: float | None = None
    mad: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    percentile_1: float | None = None
    percentile_5: float | None = None
    percentile_25: float | None = None
    percentile_75: float | None = None
    percentile_95: float | None = None
    percentile_99: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    l1_norm: float | None = None
    l2_norm: float | None = None
    sparsity: float | None = None  # fraction of finite elements == 0
    zero_ratio: float | None = None  # alias of sparsity, kept for spec naming


def compute_tensor_statistics(tensor: np.ndarray) -> TensorStatistics:
    """Compute descriptive statistics for a single tensor.

    Non-finite (NaN/Inf) elements are excluded from all statistics beyond
    the counts themselves; this keeps a single corrupted value from
    poisoning every other reported statistic with NaN.
    """
    arr = np.asarray(tensor)
    shape = list(arr.shape)
    dtype_name = str(arr.dtype)
    element_count = int(arr.size)

    if element_count == 0:
        return TensorStatistics(
            shape=shape,
            dtype=dtype_name,
            element_count=0,
            status="empty",
            notes=["Tensor has zero elements; no statistics computable."],
        )

    # Upcast to float64 for numerically stable statistics regardless of the
    # tensor's storage dtype (float16 in particular overflows/underflows
    # easily in intermediate sums of squares, skew, kurtosis).
    flat = arr.ravel().astype(np.float64, copy=False)

    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    finite_count = int(finite.size)

    notes: list[str] = []
    if nan_count:
        notes.append(f"{nan_count} NaN element(s) excluded from statistics.")
    if inf_count:
        notes.append(f"{inf_count} Inf element(s) excluded from statistics.")

    if finite_count == 0:
        return TensorStatistics(
            shape=shape,
            dtype=dtype_name,
            element_count=element_count,
            status="all_non_finite",
            notes=notes + ["All elements are NaN/Inf; no statistics computable."],
            nan_count=nan_count,
            inf_count=inf_count,
            finite_count=0,
        )

    def _safe(value: float, label: str) -> float | None:
        """Guard against silent overflow: extremely large tensor values can
        push intermediate sums-of-squares/cubes past float64 range, turning
        a statistic into inf/NaN. Reporting that as if it were a real value
        would silently corrupt everything downstream (feature vectors,
        anomaly scores). Report None with an explicit note instead.
        """
        if np.isfinite(value):
            return float(value)
        notes.append(f"{label} could not be computed: numerical overflow (values too extreme).")
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        mean_val = _safe(np.mean(finite), "mean")
        std_val = _safe(np.std(finite), "std")
        variance_val = _safe(np.var(finite), "variance")
        l1_val = _safe(np.sum(np.abs(finite)), "l1_norm")
        l2_val = _safe(np.linalg.norm(finite), "l2_norm")

        if std_val == 0.0:
            # Constant tensor: skewness/kurtosis are mathematically 0/0.
            # SciPy returns 0.0 for this case already, but we make the
            # assumption explicit rather than relying on library behavior.
            skewness = 0.0
            kurtosis = 0.0
            notes.append("Zero variance (constant tensor): skewness/kurtosis reported as 0.")
        elif std_val is None:
            skewness = None
            kurtosis = None
        else:
            skewness = _safe(sp_stats.skew(finite), "skewness")
            kurtosis = _safe(sp_stats.kurtosis(finite), "kurtosis")

    percentiles = np.percentile(finite, [1, 5, 25, 75, 95, 99])
    zero_count = int(np.count_nonzero(finite == 0.0))
    sparsity = zero_count / finite_count

    return TensorStatistics(
        shape=shape,
        dtype=dtype_name,
        element_count=element_count,
        status="ok",
        notes=notes,
        nan_count=nan_count,
        inf_count=inf_count,
        finite_count=finite_count,
        mean=mean_val,
        std=std_val,
        variance=variance_val,
        median=float(np.median(finite)),
        mad=median_absolute_deviation(finite),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        percentile_1=float(percentiles[0]),
        percentile_5=float(percentiles[1]),
        percentile_25=float(percentiles[2]),
        percentile_75=float(percentiles[3]),
        percentile_95=float(percentiles[4]),
        percentile_99=float(percentiles[5]),
        skewness=skewness,
        kurtosis=kurtosis,
        l1_norm=l1_val,
        l2_norm=l2_val,
        sparsity=sparsity,
        zero_ratio=sparsity,
    )
