"""Cosine similarity indexing and retrieval model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from app.utils.math import normalize_matrix, normalize_vector

ItemT = TypeVar("ItemT")
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class CosineIndex:
    """Normalized vectors paired with their original items."""

    vectors: FloatArray
    items: list[ItemT]


@dataclass(frozen=True)
class CosineResult:
    """One ranked cosine-similarity result."""

    index: int
    item: ItemT
    score: float
    rank: int


def build_cosine_index(
    vectors: FloatArray,
    items: list[ItemT],
) -> CosineIndex:
    """Build a cosine index from vectors and their associated items.

    Args:
        vectors: N x D matrix of vectors
        items: List of N items corresponding to vectors

    Returns:
        CosineIndex ready for querying
    """
    if not items:
        raise ValueError("items cannot be empty")

    normalized_vectors = normalize_matrix(vectors)

    if normalized_vectors.shape[0] != len(items):
        raise ValueError("number of vectors must match number of items")

    return CosineIndex(
        vectors=normalized_vectors,
        items=list(items),
    )


def query_cosine_index(
    index: CosineIndex,
    query_vector: FloatArray,
    *,
    k: int = 5,
    minimum_score: float | None = None,
) -> list[CosineResult]:
    """Return the items most similar to a query vector.

    Args:
        index: CosineIndex to query
        query_vector: Query vector
        k: Number of results
        minimum_score: Filter results below this score

    Returns:
        List of CosineResult ranked by similarity
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    normalized_query = normalize_vector(query_vector)

    if normalized_query.shape[0] != index.vectors.shape[1]:
        raise ValueError("query vector dimension must match indexed vector dimension")

    scores = index.vectors @ normalized_query

    result_count = min(k, len(index.items))
    ranked_indexes = np.argsort(scores)[::-1]

    results: list[CosineResult] = []

    for document_index in ranked_indexes:
        score = float(scores[document_index])

        if minimum_score is not None and score < minimum_score:
            continue

        results.append(
            CosineResult(
                index=int(document_index),
                item=index.items[int(document_index)],
                score=score,
                rank=len(results) + 1,
            )
        )

        if len(results) == result_count:
            break

    return results
