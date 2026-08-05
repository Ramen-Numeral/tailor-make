# Retrieval benchmark summary

This study evaluates requirement-to-resume-fragment retrieval independently from resume generation. 

## Protocol

- 30 resumes and 240 deterministic source fragments
- 40 development and 200 held-out source fragments, split by resume
- Five query types per fragment: lexical, semantic, abstraction, constraint-focused, and compositional
- 1,200 known-positive queries and 240 synthetic unsupported requirements
- Fixed seed of 42; thresholds selected on development resumes
- Hard negatives selected from high-ranking non-source BM25 and embedding results

## Retrieval results

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | Miss rate |
|---|---:|---:|---:|---:|---:|
| Embedding | **0.932** | 0.998 | **1.000** | **0.964** | 0.000 |
| Hybrid | 0.925 | 0.994 | **1.000** | 0.959 | 0.000 |
| BM25 | 0.922 | **0.999** | **1.000** | 0.958 | 0.000 |
| Exact/alias | 0.795 | 0.870 | 0.877 | 0.833 | 0.122 |

Embeddings were strongest overall, while BM25 and hybrid retrieval were close behind. All three learned methods recovered every correct source by rank five, providing strong coverage even when the top-ranked result varied.

## Hybrid comparison

| Comparison | Metric | Delta | Paired bootstrap 95% CI |
|---|---|---:|---|
| Hybrid − embedding | Recall@1 | -0.0070 | [-0.0150, +0.0010] |
| Hybrid − embedding | MRR | -0.0052 | [-0.0102, -0.0004] |
| Hybrid − BM25 | Recall@1 | +0.0030 | [-0.0120, +0.0180] |
| Hybrid − BM25 | MRR | +0.0009 | [-0.0082, +0.0092] |

Under this synthetic protocol, embeddings produced the strongest overall ranking, while hybrid and BM25 performance were statistically comparable. The hybrid achieved perfect Recall@5 and performed better than embeddings and BM25 on the 100 hard-negative cases, although exact matching was strongest on that specific slice. The remaining opportunity is to refine how exact matches are prioritized when a longer fragment provides fuller context.

## Unsupported requirements

| Method | Precision | Recall | F1 | Unsupported false-positive rate |
|---|---:|---:|---:|---:|
| Hybrid | 0.997 | **1.000** | **0.999** | 0.015 |
| Embedding | 0.988 | 0.992 | 0.990 | 0.060 |
| BM25 | 0.999 | 0.927 | 0.962 | 0.005 |
| Exact/alias | **1.000** | 0.891 | 0.942 | **0.000** |

These values use development-selected score thresholds. The hybrid provided the strongest precision-recall balance and rejected 98.5% of synthetic unsupported requirements. Exact matching produced no false positives but naturally covered fewer supported requirements. Because the unsupported cases came from an absence template bank, the rates are best treated as encouraging directional evidence rather than production estimates.

## Interpretation

The benchmark shows that the production retrieval components can reliably surface source-backed experience: embeddings retrieved a correct fragment first in 93.2% of held-out cases, while embeddings, BM25, and the hybrid found one within five results in every case. For a recruiter-facing workflow, that supports fast evidence discovery while preserving direct provenance to the original resume fragment.

Embeddings are the strongest individual default candidate from this study. The hybrid remains competitive on ranking and provides the strongest supported-versus-unsupported classification balance, making it a useful foundation for further threshold and ordering refinement.

The high absolute scores partly reflect known-answer construction where queries are deterministic transformations of their source fragments. The results therefore demonstrate controlled retrieval consistency and provenance preservation rather than final recruiter relevance. A requirement-to-evidence set would be the next step for validating the same strengths under real hiring language.

