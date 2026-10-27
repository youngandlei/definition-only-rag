#!/usr/bin/env python3
"""
Inter-annotator agreement (IAA) between two LLM-based annotation runs of the
MMLU↔Investopedia term mapping.

Inputs: two audit JSONs from analysis/annotate_mmlu_terms.py (must share
        identical candidate pools per question — i.e., both runs used the
        same retrieve_candidates.py output).

Outputs:
  - Per-question Jaccard similarity, exact-match rate
  - Per-(question, candidate) Cohen's κ on binary in/out-of-gold decisions
  - F1, precision, recall (A as ref → B), and symmetric F1
  - Set-size agreement (Pearson r, mean abs diff)
  - Per-subject breakdown

Usage:
  python3 analysis/inter_annotator_agreement.py \\
    --a outputs/analysis/annotation/mmlu_finance_annotation_audit.json \\
    --b outputs/analysis/annotation/mmlu_finance_annotation_audit_gpt4o.json \\
    --a_label GPT-5.2 --b_label GPT-4o \\
    --output outputs/analysis/annotation/iaa_report.json
"""

import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy)


def cohens_kappa(observed_agree, n, p_yes_a, p_yes_b):
    """Cohen's kappa for a binary task.
    observed_agree = P(A=B), p_yes_a = P(A=1), p_yes_b = P(B=1)."""
    p_e = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    if 1 - p_e == 0:
        return 0.0
    return (observed_agree - p_e) / (1 - p_e)


def subject_from_row_key(rk):
    if not rk:
        return "unknown"
    try:
        d = json.loads(rk)
        return d.get("subject", "unknown") or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="First audit JSON")
    ap.add_argument("--b", required=True, help="Second audit JSON")
    ap.add_argument("--a_label", default="A")
    ap.add_argument("--b_label", default="B")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    a_list = json.load(open(args.a))
    b_list = json.load(open(args.b))

    # index by row_key
    a_by = {x["row_key"]: x for x in a_list}
    b_by = {x["row_key"]: x for x in b_list}
    shared = sorted(set(a_by) & set(b_by))
    print(f"Shared questions: {len(shared)} "
          f"(A only: {len(a_by)-len(shared)}, B only: {len(b_by)-len(shared)})")

    # ---- Per-question agreement ----
    jacc_scores = []
    exact_matches = 0
    a_sizes = []
    b_sizes = []
    set_diffs = []
    per_subject = defaultdict(list)

    # Per-(question, candidate) binary agreement
    n_total_pairs = 0
    n_both_yes = 0
    n_both_no = 0
    n_a_only = 0  # A picked, B didn't
    n_b_only = 0

    candidate_pool_mismatches = 0

    examples_disagree = []  # for inspection

    for rk in shared:
        ra = a_by[rk]
        rb = b_by[rk]
        a_pool = {c["doc_id"] for c in ra["candidates"]}
        b_pool = {c["doc_id"] for c in rb["candidates"]}
        if a_pool != b_pool:
            candidate_pool_mismatches += 1
            continue  # skip if pools differ
        pool = a_pool

        a_gold = set(ra.get("gold_inv_term_ids", []))
        b_gold = set(rb.get("gold_inv_term_ids", []))

        a_sizes.append(len(a_gold))
        b_sizes.append(len(b_gold))
        set_diffs.append(abs(len(a_gold) - len(b_gold)))

        j = jaccard(a_gold, b_gold)
        jacc_scores.append(j)
        if a_gold == b_gold:
            exact_matches += 1

        subj = subject_from_row_key(rk)
        per_subject[subj].append(j)

        # per-candidate binary
        for cid in pool:
            n_total_pairs += 1
            in_a = cid in a_gold
            in_b = cid in b_gold
            if in_a and in_b:
                n_both_yes += 1
            elif not in_a and not in_b:
                n_both_no += 1
            elif in_a and not in_b:
                n_a_only += 1
            else:
                n_b_only += 1

        # save a few disagreements
        if a_gold != b_gold and len(examples_disagree) < 30:
            cand_text = {c["doc_id"]: c["term"] for c in ra["candidates"]}
            examples_disagree.append({
                "row_key": rk,
                "question": ra["question"][:200],
                "subject": subj,
                f"{args.a_label}_picked": sorted([(cid, cand_text.get(cid, "?")) for cid in a_gold]),
                f"{args.b_label}_picked": sorted([(cid, cand_text.get(cid, "?")) for cid in b_gold]),
            })

    n_questions = len(jacc_scores)
    mean_jaccard = sum(jacc_scores) / n_questions if n_questions else 0
    exact_rate = exact_matches / n_questions if n_questions else 0

    # F1 with A as ref
    tp = n_both_yes
    fn = n_a_only
    fp = n_b_only
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Cohen's κ (per-candidate binary)
    observed_agree = (n_both_yes + n_both_no) / n_total_pairs if n_total_pairs else 0
    p_yes_a = (n_both_yes + n_a_only) / n_total_pairs if n_total_pairs else 0
    p_yes_b = (n_both_yes + n_b_only) / n_total_pairs if n_total_pairs else 0
    kappa = cohens_kappa(observed_agree, n_total_pairs, p_yes_a, p_yes_b)

    size_pearson = pearson(a_sizes, b_sizes)

    # Per-subject
    subject_stats = {
        s: {
            "n": len(scores),
            "mean_jaccard": sum(scores) / len(scores) if scores else 0,
            "exact_rate": sum(1 for j in scores if j == 1) / len(scores) if scores else 0,
        }
        for s, scores in per_subject.items()
    }

    report = {
        "a_label": args.a_label,
        "b_label": args.b_label,
        "n_questions": n_questions,
        "candidate_pool_mismatches": candidate_pool_mismatches,
        "exact_match_rate": exact_rate,
        "mean_jaccard": mean_jaccard,
        "f1_A_as_ref":           f1,
        "precision_A_as_ref":    precision,
        "recall_A_as_ref":       recall,
        "cohens_kappa_binary_per_candidate": kappa,
        "observed_per_candidate_agreement": observed_agree,
        "n_total_(question_candidate)_pairs": n_total_pairs,
        "n_both_yes": n_both_yes,
        "n_both_no": n_both_no,
        f"n_only_{args.a_label}": n_a_only,
        f"n_only_{args.b_label}": n_b_only,
        f"{args.a_label}_mean_set_size":   sum(a_sizes) / n_questions if n_questions else 0,
        f"{args.b_label}_mean_set_size":   sum(b_sizes) / n_questions if n_questions else 0,
        "set_size_pearson_r": size_pearson,
        "mean_set_size_abs_diff": sum(set_diffs) / n_questions if n_questions else 0,
        "subject_stats": subject_stats,
        "disagreement_examples_first30": examples_disagree,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== IAA Summary ({args.a_label} vs {args.b_label}) ===")
    print(f"Questions compared:              {n_questions}")
    print(f"Candidate-pool mismatches (skipped): {candidate_pool_mismatches}")
    print(f"Exact set match rate:            {exact_rate:.4f}  ({exact_matches}/{n_questions})")
    print(f"Mean Jaccard (per question):     {mean_jaccard:.4f}")
    print(f"Cohen's κ (per-candidate binary): {kappa:.4f}")
    print(f"   κ interpretation: <0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect")
    print(f"F1 (A as ref):                   {f1:.4f}  (P={precision:.4f}, R={recall:.4f})")
    print(f"Set-size Pearson r:              {size_pearson:.4f}")
    print(f"{args.a_label} mean set size:    {report[f'{args.a_label}_mean_set_size']:.3f}")
    print(f"{args.b_label} mean set size:    {report[f'{args.b_label}_mean_set_size']:.3f}")
    print(f"Mean |set_size_A - set_size_B|:  {report['mean_set_size_abs_diff']:.3f}")
    print(f"\nPer-(question, candidate) breakdown of {n_total_pairs} pairs:")
    print(f"  Both YES:        {n_both_yes:>6}  ({n_both_yes/n_total_pairs*100:.1f}%)")
    print(f"  Both NO:         {n_both_no:>6}  ({n_both_no/n_total_pairs*100:.1f}%)")
    print(f"  {args.a_label}-only:        {n_a_only:>6}  ({n_a_only/n_total_pairs*100:.1f}%)")
    print(f"  {args.b_label}-only:        {n_b_only:>6}  ({n_b_only/n_total_pairs*100:.1f}%)")
    print(f"\nPer-subject mean Jaccard:")
    for s, st in sorted(subject_stats.items()):
        print(f"  {s[:35]:35} n={st['n']:>4}  mean_jaccard={st['mean_jaccard']:.4f}  exact={st['exact_rate']:.4f}")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
