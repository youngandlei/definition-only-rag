#!/usr/bin/env python3
"""
Term-knowledge-based gating: only apply RAG when retrieved terms
have METEOR < threshold (i.e., model doesn't know them well).

Usage:
    python3 analysis/term_knowledge_gating.py --threshold 0.30
    python3 analysis/term_knowledge_gating.py --threshold 0.25
    python3 analysis/term_knowledge_gating.py --threshold 0.20
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent

# Trace file pairs: (baseline, rag)
TRACE_PAIRS = {
    "mmlu": {
        "base": "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl",
        "rag_dense": "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__dense__defs_first.jsonl",
        "rag_hybrid": "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
        "rag_dense_rerank": "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__dense_rerank__defs_last.jsonl",
    },
    "xfinbench": {
        "base": "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__none__defs_first.jsonl",
        "rag_dense": "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__dense__defs_last.jsonl",
        "rag_hybrid": "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__hybrid_rerank__defs_last.jsonl",
    },
    "scenario": {
        "base": "outputs/traces/main/2026-04-15_qwen7_scenario/vllm_qwen_qwen2_5_7b_instruct__scenario_bench__none__defs_first.jsonl",
        "rag_hybrid": "outputs/traces/main/2026-04-15_qwen7_scenario/vllm_qwen_qwen2_5_7b_instruct__scenario_bench__hybrid_rerank__defs_last.jsonl",
    },
}

TERM_SCORES_PATH = "outputs/analysis/term_eval/qwen7b_scored.json"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_term_meteor_scores(path):
    """Load term -> METEOR score mapping."""
    with open(path) as f:
        data = json.load(f)
    term_scores = {}
    for r in data["results"]:
        term_scores[r["term"].lower()] = r["meteor"]
    return term_scores


def get_top_term_meteor(record, term_scores):
    """Get the METEOR score of the top-1 retrieved term for a question.
    Returns None if no retrieved terms or term not found in scores."""
    defs = record.get("retrieved_definitions", [])
    if not defs:
        return None
    # Use top-1 retrieved term (highest retrieval score)
    top_term = defs[0].get("term", "")
    return term_scores.get(top_term.lower(), None)


def get_any_below_threshold(record, term_scores, threshold):
    """Check if ANY retrieved term has METEOR below threshold."""
    defs = record.get("retrieved_definitions", [])
    if not defs:
        return False
    for d in defs:
        term = d.get("term", "")
        meteor = term_scores.get(term.lower())
        if meteor is not None and meteor < threshold:
            return True
    return False


def get_avg_meteor(record, term_scores):
    """Get average METEOR of all retrieved terms."""
    defs = record.get("retrieved_definitions", [])
    if not defs:
        return None
    scores = []
    for d in defs:
        term = d.get("term", "")
        m = term_scores.get(term.lower())
        if m is not None:
            scores.append(m)
    return sum(scores) / len(scores) if scores else None


def evaluate_gating(base_records, rag_records, term_scores, threshold,
                    strategy="top1", min_retrieval_score=0.4):
    """
    Evaluate gating strategy:
    - For each question, if the gating condition is met → use RAG answer
    - Otherwise → use baseline answer

    strategy: "top1" (top retrieved term < threshold),
              "any" (any retrieved term < threshold),
              "avg" (average METEOR of retrieved terms < threshold)
    """
    assert len(base_records) == len(rag_records), \
        f"Mismatch: {len(base_records)} base vs {len(rag_records)} rag"

    n = len(base_records)
    gated_correct = 0
    base_correct = 0
    rag_correct = 0
    oracle_correct = 0
    n_use_rag = 0
    n_use_base = 0

    # Per-category stats
    stats = {
        "rag_used_fixed": 0,    # RAG used and it fixed a wrong answer
        "rag_used_broke": 0,    # RAG used and it broke a correct answer
        "rag_used_both_right": 0,
        "rag_used_both_wrong": 0,
        "base_kept_right": 0,   # Baseline kept and it was correct
        "base_kept_wrong": 0,   # Baseline kept and it was wrong
    }

    for b, r in zip(base_records, rag_records):
        bc = b["correct"]
        rc = r["correct"]
        base_correct += bc
        rag_correct += rc
        oracle_correct += max(bc, rc)

        # Decide whether to gate
        if strategy == "top1":
            meteor = get_top_term_meteor(r, term_scores)
            use_rag = meteor is not None and meteor < threshold
        elif strategy == "any":
            use_rag = get_any_below_threshold(r, term_scores, threshold)
        elif strategy == "avg":
            avg_m = get_avg_meteor(r, term_scores)
            use_rag = avg_m is not None and avg_m < threshold
        elif strategy == "relevant_weak":
            # Only use RAG if a retrieved term is BOTH relevant (high retrieval score)
            # AND weak (low METEOR)
            defs = r.get("retrieved_definitions", [])
            use_rag = False
            for d in defs:
                ret_score = d.get("score", 0)
                term = d.get("term", "")
                meteor = term_scores.get(term.lower())
                if ret_score >= min_retrieval_score and meteor is not None and meteor < threshold:
                    use_rag = True
                    break
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if use_rag:
            n_use_rag += 1
            gated_correct += rc
            if bc == 0 and rc == 1:
                stats["rag_used_fixed"] += 1
            elif bc == 1 and rc == 0:
                stats["rag_used_broke"] += 1
            elif bc == 1 and rc == 1:
                stats["rag_used_both_right"] += 1
            else:
                stats["rag_used_both_wrong"] += 1
        else:
            n_use_base += 1
            gated_correct += bc
            if bc == 1:
                stats["base_kept_right"] += 1
            else:
                stats["base_kept_wrong"] += 1

    return {
        "n": n,
        "threshold": threshold,
        "strategy": strategy,
        "base_acc": base_correct / n,
        "rag_acc": rag_correct / n,
        "gated_acc": gated_correct / n,
        "oracle_acc": oracle_correct / n,
        "n_use_rag": n_use_rag,
        "n_use_base": n_use_base,
        "pct_use_rag": n_use_rag / n * 100,
        "delta_vs_base": (gated_correct - base_correct) / n * 100,
        "delta_vs_rag": (gated_correct - rag_correct) / n * 100,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="METEOR threshold for gating (default: 0.30)")
    ap.add_argument("--strategy", default="top1",
                    choices=["top1", "any", "avg", "relevant_weak"],
                    help="Gating strategy: top1, any, avg, relevant_weak")
    ap.add_argument("--min_retrieval_score", type=float, default=0.4,
                    help="Min retrieval score for relevant_weak strategy")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep thresholds from 0.10 to 0.50")
    args = ap.parse_args()

    # Load term METEOR scores
    scores_path = REPO / TERM_SCORES_PATH
    print(f"Loading term scores from {scores_path}")
    term_scores = load_term_meteor_scores(scores_path)
    print(f"  Loaded {len(term_scores)} terms")

    thresholds = [args.threshold]
    if args.sweep:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    all_results = []

    for bench_name, paths in TRACE_PAIRS.items():
        base_path = REPO / paths["base"]
        if not base_path.exists():
            print(f"\n  Skipping {bench_name} (baseline not found)")
            continue

        base_records = load_jsonl(base_path)

        # Try each RAG strategy
        for rag_key, rag_path_str in paths.items():
            if rag_key == "base":
                continue
            rag_path = REPO / rag_path_str
            if not rag_path.exists():
                continue

            rag_records = load_jsonl(rag_path)
            rag_label = rag_key.replace("rag_", "")

            for threshold in thresholds:
                result = evaluate_gating(
                    base_records, rag_records, term_scores,
                    threshold=threshold,
                    strategy=args.strategy,
                    min_retrieval_score=args.min_retrieval_score,
                )
                result["benchmark"] = bench_name
                result["rag_strategy"] = rag_label
                all_results.append(result)

    # Print results
    print("\n" + "=" * 90)
    print(f"TERM-KNOWLEDGE GATING RESULTS (strategy={args.strategy})")
    print("=" * 90)

    for bench_name in TRACE_PAIRS:
        bench_results = [r for r in all_results if r["benchmark"] == bench_name]
        if not bench_results:
            continue

        print(f"\n{'─' * 90}")
        print(f"  {bench_name.upper()}")
        print(f"{'─' * 90}")
        print(f"  {'RAG Strategy':<16} {'Thr':>5} {'Base':>7} {'Always RAG':>11} "
              f"{'Gated':>7} {'Oracle':>7} {'Δ vs Base':>10} {'Δ vs RAG':>9} "
              f"{'%RAG':>6}")

        for r in bench_results:
            print(f"  {r['rag_strategy']:<16} {r['threshold']:>5.2f} "
                  f"{r['base_acc']:>7.4f} {r['rag_acc']:>11.4f} "
                  f"{r['gated_acc']:>7.4f} {r['oracle_acc']:>7.4f} "
                  f"{r['delta_vs_base']:>+9.2f}% {r['delta_vs_rag']:>+8.2f}% "
                  f"{r['pct_use_rag']:>5.1f}%")

            if not args.sweep:
                s = r["stats"]
                print(f"    RAG used: {r['n_use_rag']} questions "
                      f"(fixed {s['rag_used_fixed']}, broke {s['rag_used_broke']}, "
                      f"both_right {s['rag_used_both_right']}, "
                      f"both_wrong {s['rag_used_both_wrong']})")
                print(f"    Base kept: {r['n_use_base']} questions "
                      f"(right {s['base_kept_right']}, wrong {s['base_kept_wrong']})")

    # Save results
    out_dir = REPO / "outputs" / "analysis" / "term_gating"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gating_{args.strategy}_t{args.threshold:.2f}.json"
    if args.sweep:
        out_file = out_dir / f"gating_{args.strategy}_sweep.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
