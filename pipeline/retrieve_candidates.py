#!/usr/bin/env python3
"""
Top-K candidate retrieval for MMLU (or other MCQ) questions against Investopedia.

Uses the same hybrid_rerank pipeline as rag_benchmarks.py:
  TF-IDF + Dense (MiniLM) -> RRF fusion -> Cross-encoder rerank

Input:  trace JSONL (one record per question, must have row_key, question, choices, gold)
Output: JSON with per-question top-K candidate term lists

Usage:
  python3 analysis/retrieve_candidates.py \\
    --trace outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl \\
    --investopedia data/investopedia/InvestopediaDatasetFinal.json \\
    --top_k 10 \\
    --output outputs/analysis/annotation/mmlu_candidates_top10.json
"""

import argparse
import json
import sys
from pathlib import Path

# Reuse retrieval primitives from rag_benchmarks.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_benchmarks import (
    read_investopedia,
    TfidfRetriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
)


def query_text(question, choices):
    choices_str = " ".join(choices) if isinstance(choices, list) else str(choices)
    return f"{question} {choices_str}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Source trace JSONL")
    ap.add_argument("--investopedia", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--rerank_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final candidate pool size (after rerank)")
    ap.add_argument("--retrieve_pool", type=int, default=30,
                    help="Initial retrieval pool size before reranking")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Load corpus
    docs = read_investopedia(args.investopedia)
    print(f"[1/4] Loaded {len(docs)} Investopedia terms")

    # Load questions from trace (deduplicated by row_key)
    seen = set()
    questions = []
    with open(args.trace) as f:
        for line in f:
            r = json.loads(line)
            k = r["row_key"]
            if k in seen:
                continue
            seen.add(k)
            questions.append({
                "row_key": k,
                "benchmark": r.get("benchmark", ""),
                "subject": r.get("subject", ""),
                "type": r.get("type", "mcq"),
                "question": r["question"],
                "choices": r["choices"],
                "gold": r["gold"],
            })
    print(f"[2/4] Loaded {len(questions)} unique questions from {args.trace}")

    # Build retrievers
    print("[3/4] Building retrievers (TF-IDF, Dense, Reranker)...")
    print("       (CPU encoding of 13k+ terms takes ~5 min on Mac)")
    tfidf = TfidfRetriever(docs)
    dense = DenseRetriever(docs, args.dense_model, device=None)
    hybrid = HybridRetriever(docs, [tfidf, dense])
    reranker = Reranker(args.rerank_model, device=None)
    print("       retrievers ready")

    # Retrieve + rerank
    print(f"[4/4] Retrieving top-{args.top_k} (from pool of {args.retrieve_pool}) per question...")
    results = []
    for i, q in enumerate(questions):
        qt = query_text(q["question"], q["choices"])
        pool = hybrid.query(qt, top_k=args.retrieve_pool)
        reranked = reranker.rerank(qt, pool, top_k=args.top_k)
        candidates = []
        for doc, score in reranked:
            # doc.text == f"{term}. {definition}"; recover definition
            definition = doc.text[len(doc.term) + 2:].strip() if doc.text.startswith(doc.term + ".") else doc.text
            candidates.append({
                "doc_id": doc.doc_id,
                "term": doc.term,
                "definition": definition,
                "score": float(score),
            })
        results.append({**q, "candidates": candidates})
        if (i + 1) % 100 == 0 or (i + 1) == len(questions):
            print(f"       {i+1}/{len(questions)} done")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.output} with {len(results)} records (top-{args.top_k} candidates each)")


if __name__ == "__main__":
    main()
