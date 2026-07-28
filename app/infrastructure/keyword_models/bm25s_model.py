"""BM25S text indexing and retrieval model.

Wrapper around bm25s library for building and querying text indexes.
"""

from dataclasses import dataclass

import bm25s
import Stemmer

stemmer = Stemmer.Stemmer("english")


@dataclass
class BM25Index:
    """Indexed BM25 retriever with corpus."""

    retriever: bm25s.BM25
    corpus: list[str]


def index_bm25(corpus: str | list[str]) -> BM25Index:
    """Build BM25 index from corpus.

    Args:
        corpus: Text or list of texts to index

    Returns:
        BM25Index ready for querying
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    if not corpus:
        raise ValueError("corpus cannot be empty")

    tokens = bm25s.tokenize(
        corpus,
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    )

    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)

    return BM25Index(
        retriever=retriever,
        corpus=corpus,
    )


def query_bm25_index(
    index: BM25Index,
    query: str,
    k: int = 5,
) -> list[tuple[int, str, float]]:
    """Query BM25 index.

    Args:
        index: BM25Index to query
        query: Query text
        k: Number of results

    Returns:
        List of (doc_index, document, score) tuples
    """
    if not query.strip():
        raise ValueError("query cannot be empty")

    k = min(k, len(index.corpus))

    query_tokens = bm25s.tokenize(
        [query],
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    )

    result_indexes, scores = index.retriever.retrieve(
        query_tokens,
        k=k,
        show_progress=False,
    )

    return [
        (
            int(document_index),
            index.corpus[int(document_index)],
            float(score),
        )
        for document_index, score in zip(
            result_indexes[0],
            scores[0],
            strict=True,
        )
    ]
