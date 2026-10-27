#!/usr/bin/env python3
"""
Retrieval score threshold gating: skip RAG when top retrieval score is below threshold.
Sweeps thresholds and evaluates on all questions (not just disagree cases).
"""

import json
import numpy as np
from pathlib import Path


def load_trace(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["row_key"]] = r
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_trace", required=True)
    ap.add_argument("--rag_trace", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    base_trace = load_trace(args.baseline_trace)
    rag_trace = load_trace(args.rag_trace)
    keys = sorted(set(base_trace) & set(rag_trace))
    n = len(keys)

    acc_base = sum(base_trace[k]["correct"] for k in keys) / n
    acc_rag = sum(rag_trace[k]["correct"] for k in keys) / n
    oracle = sum(max(base_trace[k]["correct"], rag_trace[k]["correct"]) for k in keys) / n

    disagree_keys = [k for k in keys if base_trace[k]["prediction"] != rag_trace[k]["prediction"]]
    d_bw = sum(1 for k in disagree_keys if base_trace[k]["correct"] and not rag_trace[k]["correct"])
    d_rw = sum(1 for k in disagree_keys if rag_trace[k]["correct"] and not base_trace[k]["correct"])

    print(f"Total: {n}, disagree: {len(disagree_keys)} (base_wins={d_bw}, rag_wins={d_rw})")
    print(f"Baseline: {acc_base:.4f} | RAG: {acc_rag:.4f} | Oracle: {oracle:.4f}")

    # Get top retrieval score for each question
    scores = {}
    for k in keys:
        r = rag_trace[k]
        defs = r.get("retrieved_definitions") or []
        def_scores = [d.get("score", 0.0) for d in defs if isinstance(d, dict)]
        scores[k] = max(def_scores) if def_scores else 0.0

    # Distribution of scores
    all_scores = [scores[k] for k in keys]
    disagree_scores_bw = [scores[k] for k in disagree_keys if base_trace[k]["correct"] and not rag_trace[k]["correct"]]
    disagree_scores_rw = [scores[k] for k in disagree_keys if rag_trace[k]["correct"] and not base_trace[k]["correct"]]

    print(f"\n=== Score distributions ===")
    print(f"  All questions:     mean={np.mean(all_scores):.3f}, median={np.median(all_scores):.3f}")
    print(f"  RAG hurts (67):    mean={np.mean(disagree_scores_bw):.3f}, median={np.median(disagree_scores_bw):.3f}")
    print(f"  RAG helps (104):   mean={np.mean(disagree_scores_rw):.3f}, median={np.median(disagree_scores_rw):.3f}")

    # Percentiles
    print(f"\n  Percentiles (all questions):")
    for p in [10, 25, 50, 75, 90]:
        print(f"    P{p}: {np.percentile(all_scores, p):.3f}")

    # Sweep thresholds
    print(f"\n=== Threshold sweep ===")
    print(f"  {'Threshold':<12} {'Overall':>8} {'Δ RAG':>7} {'Disagree':>10} {'Used base':>10} {'Used RAG':>10} {'Skip%':>6}")
    print(f"  {'-'*70}")

    results = []
    for threshold in np.arange(0.0, 1.01, 0.05):
        correct = 0
        d_correct = 0
        used_base = 0
        used_rag = 0
        for k in keys:
            use_rag = scores[k] >= threshold
            if use_rag:
                used_rag += 1
                chosen = rag_trace[k]
            else:
                used_base += 1
                chosen = base_trace[k]
            correct += chosen["correct"]
            if k in disagree_keys:
                d_correct += chosen["correct"]

        acc = correct / n
        skip_pct = used_base / n * 100
        delta = acc - acc_rag
        marker = " ***" if delta > 0.01 else " **" if delta > 0.005 else " *" if delta > 0 else ""
        print(f"  {threshold:<12.2f} {acc:>8.4f} {delta:>+7.4f} {d_correct:>7}/{len(disagree_keys):<3} {used_base:>10} {used_rag:>10} {skip_pct:>5.1f}%{marker}")

        results.append({
            "threshold": float(threshold),
            "overall_acc": acc,
            "delta_vs_rag": delta,
            "disagree_correct": d_correct,
            "used_base": used_base,
            "used_rag": used_rag,
        })

    # Find best
    best = max(results, key=lambda x: x["overall_acc"])
    print(f"\n  Best threshold: {best['threshold']:.2f} → {best['overall_acc']:.4f} (Δ vs RAG: {best['delta_vs_rag']:+.4f})")
    print(f"  Always RAG: {acc_rag:.4f}")
    print(f"  Oracle:     {oracle:.4f}")

    # Detailed analysis at best threshold
    t = best["threshold"]
    print(f"\n=== Analysis at threshold {t:.2f} ===")
    # Among disagree cases, how many base_wins did we correctly recover?
    recovered = 0
    lost = 0
    for k in disagree_keys:
        use_rag = scores[k] >= t
        b = base_trace[k]
        r = rag_trace[k]
        if b["correct"] and not r["correct"]:
            # base_wins case
            if not use_rag:
                recovered += 1  # correctly chose base
        elif r["correct"] and not b["correct"]:
            # rag_wins case
            if not use_rag:
                lost += 1  # incorrectly chose base, should have used RAG
    print(f"  base_wins recovered (correctly skipped RAG): {recovered}/{d_bw}")
    print(f"  rag_wins lost (incorrectly skipped RAG):     {lost}/{d_rw}")
    print(f"  Net improvement on disagree: +{recovered} - {lost} = {recovered - lost}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "baseline_acc": acc_base,
            "rag_acc": acc_rag,
            "oracle_acc": oracle,
            "best_threshold": best["threshold"],
            "best_acc": best["overall_acc"],
            "results": results,
        }, indent=2))
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
