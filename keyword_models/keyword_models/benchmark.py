"""CLI for the self-contained retrieval-only benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"

import numpy as np
import pandas as pd

from .dataset_adapter import download_dataset, load_records
from .evidence_parser import extract_benchmark_evidence
from .metrics import (
    choose_thresholds,
    ndcg_at_k,
    paired_bootstrap,
    retrieval_metrics,
    slice_metrics,
    support_metrics,
)
from .query_generation import (
    RetrievalCase,
    generate_positive_cases,
    generate_unsupported_cases,
)
from .retrievers import RetrieverSuite, result_score

SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "full"), default="full")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    results = run(args.mode, offline=args.offline)
    for name in ("retrieval", "support", "bootstrap"):
        print(f"\n{name.title()}\n{results[name].to_string(index=False)}")
    return 0


def run(mode: str = "full", *, offline: bool = False) -> dict[str, object]:
    """Run the benchmark and return data frames without writing artifacts."""
    if mode not in {"fixture", "full"}:
        raise ValueError("mode must be 'fixture' or 'full'")
    CACHE.mkdir(parents=True, exist_ok=True)
    dataset_dir = download_dataset(CACHE, offline=offline)
    records = load_records(dataset_dir)
    if not records:
        raise RuntimeError("No usable primary-dataset records found")

    target_dev, target_test, unsupported_dev, unsupported_test = (
        (2, 6, 2, 4) if mode == "fixture" else (40, 200, 40, 200)
    )
    corpora, source_fragments, split_by_resume, annotations = construct_fragments(
        records, target_dev=target_dev, target_test=target_test
    )
    cases = construct_cases(
        corpora, source_fragments, split_by_resume, annotations,
        unsupported_dev=unsupported_dev, unsupported_test=unsupported_test,
    )
    suite = RetrieverSuite(CACHE, offline=offline)
    predictions, cases = evaluate_retrievers(cases, corpora, suite)
    dev = predictions[predictions["split"] == "dev"]
    test = predictions[predictions["split"] == "test"]
    thresholds = choose_thresholds(dev)
    return {
        "predictions": predictions,
        "cases": cases,
        "thresholds": thresholds,
        "retrieval": retrieval_metrics(test),
        "support": support_metrics(test, thresholds),
        "slices": slice_metrics(test),
        "bootstrap": paired_bootstrap(test, seed=SEED),
    }


def construct_fragments(records, *, target_dev: int, target_test: int):
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)
    parsed = []
    annotations: dict[str, dict[str, str]] = {}
    for record in shuffled:
        _, corpus, record_annotations = extract_benchmark_evidence(record)
        corpus = corpus[:30]
        if len(corpus) >= 5:
            parsed.append((record.resume_id, corpus))
            annotations.update(
                {
                    item.evidence_id: record_annotations[item.evidence_id]
                    for item in corpus
                }
            )

    corpora = {}
    sources = []
    split_by_resume: dict[str, str] = {}
    remaining = {"dev": target_dev, "test": target_test}
    for index, (resume_id, corpus) in enumerate(parsed):
        split = "dev" if index % 5 == 0 else "test"
        if remaining[split] <= 0:
            other = "test" if split == "dev" else "dev"
            if remaining[other] <= 0:
                break
            split = other
        candidates = _balanced_sources(
            corpus,
            annotations,
            min(8, remaining[split]),
        )
        if not candidates:
            continue
        corpora[resume_id] = corpus
        split_by_resume[resume_id] = split
        sources.extend(candidates)
        remaining[split] -= len(candidates)
    if remaining["dev"] or remaining["test"]:
        raise RuntimeError(f"Dataset could not meet fragment targets: {remaining}")
    selected_ids = {
        item.evidence_id for corpus in corpora.values() for item in corpus
    }
    annotations = {
        evidence_id: value
        for evidence_id, value in annotations.items()
        if evidence_id in selected_ids
    }
    return corpora, sources, split_by_resume, annotations


def _balanced_sources(corpus, annotations, count: int):
    by_type = defaultdict(list)
    for item in corpus:
        fact_type = annotations[item.evidence_id]["fact_type"]
        if fact_type != "category_label":
            by_type[fact_type].append(item)
    selected = []
    keys = sorted(by_type)
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if by_type[key] and len(selected) < count:
                selected.append(by_type[key].pop(0))
            if by_type[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def construct_cases(
    corpora, source_fragments, split_by_resume, annotations, *,
    unsupported_dev: int, unsupported_test: int,
):
    cases = []
    for fragment in source_fragments:
        annotation = annotations[fragment.evidence_id]
        cases.extend(
            generate_positive_cases(
                fragment,
                corpora[annotation["resume_id"]],
                annotation,
                split_by_resume[annotation["resume_id"]],
            )
        )
    for split, target in (("dev", unsupported_dev), ("test", unsupported_test)):
        candidates = []
        for resume_id, corpus in sorted(corpora.items()):
            if split_by_resume[resume_id] != split:
                continue
            domain = annotations[corpus[0].evidence_id]["domain"]
            candidates.extend(
                generate_unsupported_cases(
                    resume_id,
                    corpus,
                    domain,
                    split,
                    12,
                    SEED,
                )
            )
        cases.extend(candidates[:target])
    return cases


def evaluate_retrievers(cases, corpora, suite):
    rows = []
    updated = []
    for case in cases:
        corpus = corpora[case.resume_id]
        rankings = suite.rank_all(
            case.requirement(),
            corpus,
            corpus_key=case.resume_id,
            k=len(corpus),
        )
        gold = set(case.gold_evidence_ids)
        negatives = []
        for method in ("bm25", "embedding"):
            candidate = next(
                (
                    item for item in rankings[method]
                    if item.evidence.evidence_id not in gold
                ),
                None,
            )
            if candidate and candidate.evidence.evidence_id not in {
                item["evidence_id"] for item in negatives
            }:
                negatives.append({
                    "evidence_id": candidate.evidence.evidence_id,
                    "selection_method": f"high_{method}_non_source",
                    "rank": rankings[method].index(candidate) + 1,
                    "score": result_score(method, candidate),
                })
        while len(negatives) < 2:
            candidate = next(
                item for item in rankings["hybrid"]
                if item.evidence.evidence_id not in gold
                and item.evidence.evidence_id not in {
                    row["evidence_id"] for row in negatives
                }
            )
            negatives.append({
                "evidence_id": candidate.evidence.evidence_id,
                "selection_method": "hybrid_non_source",
                "rank": rankings["hybrid"].index(candidate) + 1,
                "score": result_score("hybrid", candidate),
            })
        hard = any(
            ranking and ranking[0].evidence.evidence_id not in gold
            for method, ranking in rankings.items()
            if method in {"bm25", "embedding"} and gold
        )
        current = dataclasses.replace(
            case, hard_negatives=negatives[:2],
            negative_setting="hard" if hard else "easy",
        )
        updated.append(current)
        for method, ranking in rankings.items():
            ranked_ids = [item.evidence.evidence_id for item in ranking]
            gold_ranks = [rank + 1 for rank, evidence_id in enumerate(ranked_ids) if evidence_id in gold]
            rows.append({
                **{key: value for key, value in current.to_dict().items() if key != "hard_negatives"},
                "method": method,
                "gold_rank": min(gold_ranks) if gold_ranks else np.nan,
                "top_evidence_id": ranked_ids[0] if ranked_ids else "",
                "top_score": result_score(method, ranking[0] if ranking else None),
                "ranked_evidence_ids": json.dumps(ranked_ids),
                "ndcg_at_5": ndcg_at_k(ranked_ids, gold, 5),
            })
    return pd.DataFrame(rows), updated


def write_examples(predictions, corpora, *, prefix):
    evidence = {item.evidence_id: item.text for corpus in corpora.values() for item in corpus}
    test = predictions[predictions["split"] == "test"].copy()
    test["gold_text"] = test["source_evidence_id"].map(evidence).fillna("")
    test["top_text"] = test["top_evidence_id"].map(evidence).fillna("")
    successful = test[(test["supported"]) & (test["gold_rank"] == 1)].head(12)[
        ["query_id", "method", "query", "gold_text", "top_text", "category"]
    ]
    failures = test[(test["supported"]) & ((test["gold_rank"].isna()) | (test["gold_rank"] > 1))].head(20)[
        ["query_id", "method", "query", "gold_text", "top_text", "gold_rank", "category"]
    ]
    pivot = test.pivot(index="query_id", columns="method", values="top_evidence_id")
    disagree_ids = pivot[pivot["bm25"] != pivot["embedding"]].index
    disagreements = test[(test["query_id"].isin(disagree_ids)) & (test["method"] == "hybrid")].head(20)[
        ["query_id", "query", "source_evidence_id", "category"]
    ].copy()
    disagreements["bm25_top"] = disagreements["query_id"].map(pivot["bm25"])
    disagreements["embedding_top"] = disagreements["query_id"].map(pivot["embedding"])
    outputs = {
        "successful retrievals": successful,
        "retrieval failures": failures,
        "BM25/embedding disagreements": disagreements,
    }
    filenames = {
        "successful retrievals": "successful_retrievals.csv",
        "retrieval failures": "retrieval_failures.csv",
        "BM25/embedding disagreements": "bm25_embedding_disagreements.csv",
    }
    for name, frame in outputs.items():
        frame.to_csv(REPORTS / f"{prefix}{filenames[name]}", index=False)
    return outputs

def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
