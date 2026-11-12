"""Stable auxiliary Coulomb whitening for three-dimensional positive metrics."""
import numpy as np
import scipy.linalg as la


ALGORITHM = "equilibrated-cholesky-svd-absolute-v1"


def scaled_eigen_metric(metric, threshold, scales=None):
    """Return W and R with W J W† = I, W R = I in the retained space.

    Equilibrate J = D S D, factor S = L L†, then use the SVD of A = D L.
    The singular values of A are sqrt(eigenvalues(J)). Unlike eigh(J), this
    does not square the condition number before resolving the small modes.
    W = V† L^-1 D^-1 and R = D L V avoid division by inaccurate tiny raw
    eigenvalues. The cutoff retains PySCF's absolute eigenvalue units.
    R is a stable right inverse of W, not an independently computed pseudoinverse.
    """
    matrix = np.asarray(metric)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("Coulomb metric must be a nonempty square matrix")
    if not np.isfinite(matrix).all() or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Coulomb metric and absolute threshold must be finite and nonnegative")
    matrix = matrix.astype(np.result_type(matrix.dtype, np.float64), copy=False)
    diagonal = matrix.diagonal().real
    if np.any(diagonal < 0):
        raise la.LinAlgError("Positive Coulomb metric requires a strictly positive diagonal")
    scale = np.sqrt(np.where(diagonal > 0, diagonal, 1.)) if scales is None else np.asarray(scales)
    if (scale.shape != diagonal.shape or np.iscomplexobj(scale)
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Metric scales must be positive finite values for every auxiliary function")
    normalized = matrix / scale[:, None] / scale[None, :]
    if not np.allclose(normalized, normalized.conj().T, atol=1e-10, rtol=1e-12):
        raise ValueError("Normalized Coulomb metric is not Hermitian")
    normalized = (normalized + normalized.conj().T) * 0.5
    try:
        factor = la.cholesky(normalized, lower=True)
        inverse = la.solve_triangular(factor, np.eye(len(matrix)), lower=True)
    except la.LinAlgError:
        values, vectors = la.eigh(normalized)
        floor = len(values)*np.finfo(values.dtype).eps*max(values[-1], 0.)
        if values[-1] <= 0 or values[0] < -floor:
            raise la.LinAlgError("Three-dimensional Coulomb metric is not positive semidefinite")
        resolved = values > floor
        factor = vectors[:, resolved]*np.sqrt(values[resolved])[None, :]
        inverse = vectors[:, resolved].conj().T/np.sqrt(values[resolved])[:, None]
    raw_factor = scale[:, None]*factor
    _, singular, rotation = la.svd(raw_factor, full_matrices=False, lapack_driver="gesvd")
    keep = singular > np.sqrt(threshold)
    if not keep.any():
        raise la.LinAlgError("No auxiliary Coulomb modes survive the absolute threshold")
    rotation = rotation[keep]
    whitener = (rotation@inverse)/scale[None, :]
    dual = raw_factor@rotation.conj().T
    return whitener, dual
