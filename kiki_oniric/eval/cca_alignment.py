"""CCA Studyforrest alignment module (cycle-3 C3.17 — Phase 2 track c).

Canonical Correlation Analysis between (a) dream-state time-
series (produced by ``HmmAligner`` from ``state_alignment.py`` and
dream-op execution traces) and (b) BOLD signals (from the C3.15
Studyforrest loader). CCA finds the pair of linear projections
that maximise the correlation between the two views ; significance
is established via a permutation null that shuffles Y's rows to
break the X↔Y correspondence.

Dependency policy : ``scikit-learn`` is an **optional** dependency.
When not importable the aligner falls back to a pure numpy + scipy
SVD implementation so CI and local unit tests work without the
heavier dependency. Pattern matches C3.11 (Norse), C3.15 (nibabel),
C3.16 (hmmlearn).

The scratch algorithm is the classical Hotelling CCA via joint
whitening + SVD :

    X_w = X · cov_x^{-1/2}                 (whitened X)
    Y_w = Y · cov_y^{-1/2}                 (whitened Y)
    U · diag(σ) · V^T = SVD(X_w^T · Y_w)   (correlations = σ)

Canonical weights are then A = cov_x^{-1/2} · U[:, :k] and
B = cov_y^{-1/2} · V[:, :k]. The top-k canonical correlations
are the leading singular values, which are bounded in [0, 1] by
construction.

Determinism : seeded via ``numpy.random.default_rng(seed)`` for
the permutation null — part of the R1 run-registry contract.

References :
- docs/interfaces/fmri-schema.yaml (schema v0.7.0+PARTIAL)
- kiki_oniric/eval/state_alignment.py (C3.16 HmmAligner output)
- harness/fmri/studyforrest.py (C3.15 BoldSeries)
- framework-C spec §6.2 (DR-3 Conformance Criterion condition 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as _linalg

if TYPE_CHECKING:  # pragma: no cover - type-check only
    from numpy.typing import NDArray

# Optional-dependency probe — sklearn.cross_decomposition.CCA
# gives a faster pathway for very wide inputs. The SVD fallback
# covers all unit tests so the dependency is not required.
try:  # pragma: no cover - branch depends on env
    from sklearn.cross_decomposition import CCA  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _SKLEARN_AVAILABLE = False


@dataclass(frozen=True)
class CcaResult:
    """Outcome of a single CCA fit between a dream-state source
    matrix X and a BOLD target matrix Y.

    ``canonical_correlations`` are sorted descending in [0, 1],
    truncated to ``n_components``. ``x_projection`` / ``y_projection``
    are the paired projected time-series ready for downstream RSA
    or scoring. ``p_value`` is the one-sided permutation p-value
    P(null r̄ >= observed r̄) and ``is_significant`` is
    ``p_value < alpha`` at the configured family-wise threshold.
    """

    canonical_correlations: "NDArray[np.floating]"   # (n_components,)
    x_projection: "NDArray[np.floating]"             # (n_samples, n_components)
    y_projection: "NDArray[np.floating]"             # (n_samples, n_components)
    p_value: float
    is_significant: bool
    null_distribution: "NDArray[np.floating]"        # (n_permutations,)


def _whitening_transform(
    cov: "NDArray[np.floating]",
    ridge: float = 1e-6,
) -> "NDArray[np.floating]":
    """Return cov^{-1/2} via symmetric eigendecomposition.

    A small ridge is added to guard near-singular covariance
    matrices (common on short BOLD runs).
    """
    p = cov.shape[0]
    cov_reg = cov + ridge * np.eye(p)
    # ``eigh`` is symmetric-only and numerically stabler than ``eig``.
    eigvals, eigvecs = _linalg.eigh(cov_reg)
    # Clip tiny / negative eigenvalues before inverse-sqrt.
    eigvals = np.clip(eigvals, ridge, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt


def _cca_svd(
    X: "NDArray[np.floating]",
    Y: "NDArray[np.floating]",
    n_components: int,
) -> tuple[
    "NDArray[np.floating]",    # canonical correlations (k,)
    "NDArray[np.floating]",    # x_projection (n, k)
    "NDArray[np.floating]",    # y_projection (n, k)
]:
    """Classical Hotelling CCA via whitening + SVD.

    Centres X, Y ; whitens both ; takes the SVD of the whitened
    cross-covariance ; returns the top-k canonical correlations
    and projections.
    """
    n, _ = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cov_x = (Xc.T @ Xc) / max(n - 1, 1)
    cov_y = (Yc.T @ Yc) / max(n - 1, 1)
    cov_xy = (Xc.T @ Yc) / max(n - 1, 1)

    Wx = _whitening_transform(cov_x)
    Wy = _whitening_transform(cov_y)
    M = Wx @ cov_xy @ Wy
    U, s, Vt = _linalg.svd(M, full_matrices=False)

    k = min(n_components, s.size)
    corrs = np.clip(s[:k], 0.0, 1.0)
    # Canonical weights
    A = Wx @ U[:, :k]
    B = Wy @ Vt[:k].T
    # Paired projections
    x_proj = Xc @ A
    y_proj = Yc @ B
    return corrs, x_proj, y_proj


@dataclass
class CcaAligner:
    """CCA aligner with permutation-null significance test.

    Parameters
    ----------
    n_components : int
        Number of canonical components to retain. Defaults to 4 —
        matches the framework's 4-channel dream→awake structure.
    n_permutations : int
        Size of the permutation null distribution. 1000 by default ;
        unit tests drop it to keep runtime bounded.
    seed : int
        RNG seed for the permutation null (R1 contract).
    alpha : float
        Significance threshold used to set ``is_significant``.
    """

    n_components: int = 4
    n_permutations: int = 1000
    seed: int = 0
    alpha: float = 0.05

    def fit(
        self,
        X: "NDArray[np.floating]",
        Y: "NDArray[np.floating]",
    ) -> CcaResult:
        """Fit CCA on (X, Y) of shape (n_samples, n_features_{x,y})."""
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        if X_arr.ndim != 2 or Y_arr.ndim != 2:
            raise ValueError(
                "CCA fit expects 2-D arrays ; got "
                f"{X_arr.shape} and {Y_arr.shape}"
            )
        if X_arr.shape[0] != Y_arr.shape[0]:
            raise ValueError(
                "CCA fit requires equal sample counts ; got "
                f"{X_arr.shape[0]} vs {Y_arr.shape[0]}"
            )
        k = min(self.n_components, X_arr.shape[1], Y_arr.shape[1])
        if k < 1:
            raise ValueError("n_components must yield at least 1 retained axis")

        corrs, x_proj, y_proj = _cca_svd(X_arr, Y_arr, k)
        observed_mean_r = float(np.mean(corrs))

        # Permutation null — shuffle Y's rows to break the X↔Y
        # correspondence, refit, record the mean canonical
        # correlation. Seeded for R1 determinism.
        rng = np.random.default_rng(self.seed)
        null = np.empty(self.n_permutations, dtype=float)
        n_samples = X_arr.shape[0]
        for i in range(self.n_permutations):
            perm = rng.permutation(n_samples)
            Y_perm = Y_arr[perm]
            corrs_null, _, _ = _cca_svd(X_arr, Y_perm, k)
            null[i] = float(np.mean(corrs_null))

        # One-sided p-value : fraction of null samples ≥ observed.
        # Add-one smoothing keeps p strictly positive when every
        # null sample is below the observed mean.
        ge = int(np.sum(null >= observed_mean_r))
        p_value = (ge + 1) / (self.n_permutations + 1)
        is_significant = bool(p_value < self.alpha)

        return CcaResult(
            canonical_correlations=corrs,
            x_projection=x_proj,
            y_projection=y_proj,
            p_value=float(p_value),
            is_significant=is_significant,
            null_distribution=null,
        )
