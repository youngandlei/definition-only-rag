#!/usr/bin/env python3
"""
Confidence Gain selection (inspired by CK-PLUG, Bai et al. 2025).

For each disagreement case, compare P(answer | question) vs P(answer | question + defs).
If adding definitions shifts probability toward the RAG-chosen answer, trust RAG.
Otherwise, trust baseline.

This operates on pre-computed choice_scores (logprobs) from trace files — no extra
forward passes needed.
"""

import argparse
import json
import math
import numpy as np
from collections import Counter
from pathlib import Path


def load_trace(path: str) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["row_key"]] = r
    return rows


def logprobs_to_probs(logprobs: list[float]) -> np.ndarray:
    """Convert logprobs to normalized probabilities."""
    lp = np.array(logprobs)
    # Numerically stable softmax (logprobs may not be normalized)
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), with smoothing to avoid inf."""
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def entropy(p: np.ndarray) -> float:
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def run(args):
    base_trace = load_trace(args.baseline_trace)
    rag_trace = load_trace(args.rag_trace)

    keys = sorted(set(base_trace) & set(rag_trace))
    agree_keys = [k for k in keys if base_trace[k]["prediction"] == rag_trace[k]["prediction"]]
    disagree_keys = [k for k in keys if base_trace[k]["prediction"] != rag_trace[k]["prediction"]]

    n = len(keys)
    print(f"Total: {n}, agree: {len(agree_keys)}, disagree: {len(disagree_keys)}")

    acc_base = sum(base_trace[k]["correct"] for k in keys) / n
    acc_rag = sum(rag_trace[k]["correct"] for k in keys) / n
    oracle = sum(max(base_trace[k]["correct"], rag_trace[k]["correct"]) for k in keys) / n

    d_bw = sum(1 for k in disagree_keys if base_trace[k]["correct"] and not rag_trace[k]["correct"])
    d_rw = sum(1 for k in disagree_keys if rag_trace[k]["correct"] and not base_trace[k]["correct"])
    d_xx = sum(1 for k in disagree_keys if not base_trace[k]["correct"] and not rag_trace[k]["correct"])
    print(f"Baseline: {acc_base:.4f} | RAG: {acc_rag:.4f} | Oracle: {oracle:.4f}")
    print(f"Disagree: base_wins={d_bw}, rag_wins={d_rw}, both_wrong={d_xx}")

    # --- Confidence Gain methods ---
    results = {}

    for k in disagree_keys:
        b = base_trace[k]
        r = rag_trace[k]

        b_scores = b["choice_scores"]
        r_scores = r["choice_scores"]

        b_probs = logprobs_to_probs(b_scores)
        r_probs = logprobs_to_probs(r_scores)

        b_pred = b["prediction"]  # index of baseline's chosen answer
        r_pred = r["prediction"]  # index of RAG's chosen answer

        # --- Method 1: Probability shift on RAG's chosen answer ---
        # If adding defs increases P(rag_answer), trust RAG
        prob_shift_rag = float(r_probs[r_pred] - b_probs[r_pred])

        # --- Method 2: Probability shift on baseline's chosen answer ---
        # If adding defs decreases P(base_answer), baseline was fragile → trust RAG
        prob_shift_base = float(r_probs[b_pred] - b_probs[b_pred])

        # --- Method 3: Entropy change ---
        # If adding defs reduces entropy → model becomes more confident with context → trust RAG
        b_ent = entropy(b_probs)
        r_ent = entropy(r_probs)
        entropy_change = r_ent - b_ent  # negative = RAG more confident

        # --- Method 4: KL divergence ---
        # How much did the distribution shift? Large shift + RAG more confident → trust RAG
        kl_br = kl_divergence(r_probs, b_probs)

        # --- Method 5: Combined confidence gain ---
        # Trust RAG if: (a) RAG answer gained probability AND (b) base answer lost probability
        # OR equivalently: the "swing" toward RAG's answer
        swing = prob_shift_rag - prob_shift_base

        # --- Method 6: Max-prob comparison ---
        # Simply: whose top choice has higher probability?
        max_prob_base = float(b_probs[b_pred])
        max_prob_rag = float(r_probs[r_pred])

        results[k] = {
            "prob_shift_rag": prob_shift_rag,
            "prob_shift_base": prob_shift_base,
            "entropy_change": entropy_change,
            "kl_divergence": kl_br,
            "swing": swing,
            "max_prob_base": max_prob_base,
            "max_prob_rag": max_prob_rag,
            "b_entropy": b_ent,
            "r_entropy": r_ent,
            "b_probs": b_probs.tolist(),
            "r_probs": r_probs.tolist(),
        }

    # --- Evaluate different selection strategies ---

    def eval_method(select_fn, name):
        """select_fn(k) -> 'base' or 'rag' for disagree cases."""
        correct = 0
        for k in keys:
            if k in results:
                chosen = select_fn(k)
            else:
                chosen = "base"  # agree case
            rec = base_trace[k] if chosen == "base" else rag_trace[k]
            correct += rec["correct"]
        acc = correct / n

        # Disagree-only accuracy
        d_correct = sum(
            (base_trace[k] if select_fn(k) == "base" else rag_trace[k])["correct"]
            for k in disagree_keys
        )
        d_acc = d_correct / len(disagree_keys)

        chose_base = sum(1 for k in disagree_keys if select_fn(k) == "base")
        chose_rag = len(disagree_keys) - chose_base
        return acc, d_correct, d_acc, chose_base, chose_rag

    strategies = {}

    # S1: prob_shift_rag > 0 → RAG answer gained probability → trust RAG
    strategies["prob_shift_rag > 0"] = lambda k: "rag" if results[k]["prob_shift_rag"] > 0 else "base"

    # S2: prob_shift_base < 0 → base answer lost probability → trust RAG
    strategies["prob_shift_base < 0"] = lambda k: "rag" if results[k]["prob_shift_base"] < 0 else "base"

    # S3: entropy decreased → RAG more confident → trust RAG
    strategies["entropy_decrease"] = lambda k: "rag" if results[k]["entropy_change"] < 0 else "base"

    # S4: swing > 0 → net shift toward RAG
    strategies["swing > 0"] = lambda k: "rag" if results[k]["swing"] > 0 else "base"

    # S5: max_prob comparison → whose top choice is more confident
    strategies["max_prob"] = lambda k: "rag" if results[k]["max_prob_rag"] > results[k]["max_prob_base"] else "base"

    # S6: prob_shift_rag with threshold sweep
    for t in [0.01, 0.05, 0.1, 0.2, 0.3]:
        strategies[f"prob_shift_rag > {t}"] = (
            lambda k, thr=t: "rag" if results[k]["prob_shift_rag"] > thr else "base"
        )

    # S7: entropy_change with threshold sweep
    for t in [-0.5, -0.3, -0.1, -0.01]:
        strategies[f"entropy_change < {t}"] = (
            lambda k, thr=t: "rag" if results[k]["entropy_change"] < thr else "base"
        )

    # S8: Combined — prob_shift_rag > 0 AND entropy decreased
    strategies["shift>0 AND ent_decrease"] = (
        lambda k: "rag" if results[k]["prob_shift_rag"] > 0 and results[k]["entropy_change"] < 0 else "base"
    )

    # S9: Combined — prob_shift_rag > 0 OR entropy decreased significantly
    strategies["shift>0 OR ent<-0.3"] = (
        lambda k: "rag" if results[k]["prob_shift_rag"] > 0 or results[k]["entropy_change"] < -0.3 else "base"
    )

    print(f"\n{'Method':<35} {'Overall':>8} {'Δ vs RAG':>9} {'Disagree':>10} {'base':>5} {'rag':>5}")
    print("-" * 75)
    print(f"{'Always baseline':<35} {acc_base:>8.4f} {acc_base - acc_rag:>+9.4f} {d_bw:>7}/{len(disagree_keys):<3} {len(disagree_keys):>5} {0:>5}")
    print(f"{'Always RAG':<35} {acc_rag:>8.4f} {0:>+9.4f} {d_rw:>7}/{len(disagree_keys):<3} {0:>5} {len(disagree_keys):>5}")

    best_name, best_acc = None, 0
    for name, fn in strategies.items():
        acc, d_correct, d_acc, cb, cr = eval_method(fn, name)
        print(f"{name:<35} {acc:>8.4f} {acc - acc_rag:>+9.4f} {d_correct:>7}/{len(disagree_keys):<3} {cb:>5} {cr:>5}")
        if acc > best_acc:
            best_acc = acc
            best_name = name

    print(f"{'Oracle':<35} {oracle:>8.4f} {oracle - acc_rag:>+9.4f} {d_bw + d_rw:>7}/{len(disagree_keys):<3}")
    print(f"\nBest: {best_name} ({best_acc:.4f})")

    # --- Save detailed results ---
    if args.output:
        summary = {
            "baseline_acc": acc_base,
            "rag_acc": acc_rag,
            "oracle_acc": oracle,
            "n_total": n,
            "n_disagree": len(disagree_keys),
            "strategies": {},
            "decisions": {k: {sk: sv for sk, sv in v.items() if sk != "b_probs" and sk != "r_probs"}
                          for k, v in results.items()},
        }
        for name, fn in strategies.items():
            acc, d_correct, d_acc, cb, cr = eval_method(fn, name)
            summary["strategies"][name] = {
                "overall_acc": acc,
                "delta_vs_rag": acc - acc_rag,
                "disagree_correct": d_correct,
                "disagree_acc": d_acc,
                "chose_base": cb,
                "chose_rag": cr,
            }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Confidence Gain selection for RAG vs baseline")
    ap.add_argument("--baseline_trace", required=True)
    ap.add_argument("--rag_trace", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
