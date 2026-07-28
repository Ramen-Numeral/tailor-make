from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.floating]


def normalize_vector(vector: FloatArray) -> FloatArray:
    """Return a unit-length copy of one vector."""
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("vector must be one-dimensional")

    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return array / norm


def normalize_matrix(vectors: FloatArray) -> FloatArray:
    """Normalize every row in a two-dimensional vector matrix."""

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")

    if matrix.shape[0] == 0:
        raise ValueError("vectors cannot be empty")

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("vectors cannot contain zero vectors")

    return matrix / norms
