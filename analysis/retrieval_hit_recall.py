#!/usr/bin/env python3
"""
Supplementary experiment: retrieval hit-rate of the top-3 against the
human-annotated terms, and re-run of the per-question term-knowledge
correlations on the subset where retrieval actually injected the annotated
term.

Motivation: Section 5.3 correlates the TK score of a question's *annotated*
terms with per-question outcomes, but the prompt is fed the retrieval
pipeline's top-3, not the annotated terms. If recall@3 is low, the null
(|r| < 0.06) has a mundane alternative explanation (the pipeline did not
inject the term whose knowledge we measured). This script reports recall@3
and re-runs the correlations on the retrieval-hit subset.

Join: pure index alignment. The annotation audit and all 10 model traces
(5 models x {baseline, RAG}) are 1124 rows in the same MMLU-Finance question
order (verified 1124/1124). Retrieval is model-independent (query = question +
choices), so the top-3 and the hit mask are taken from one RAG trace.
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(ROOT)
AUDIT = ROOT / "outputs/analysis/annotation/mmlu_finance_annotation_audit.json"
TOPK = 3

PANEL = [  # (label, score_stub, trace_stub) -- sorted by size
    ("Qwen-2.5-7B",  "qwen7b",     "main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu"),
    ("Llama-3.1-8B", "llama31_8b", "archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu"),
    ("Qwen-2.5-14B", "qwen14b",    "main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu"),
    ("Mistral-S-24B","mistral24b", "archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu"),
    ("GPT-5.2",      "gpt52",      "archive/traces_gpt/openai_gpt_5_2__mmlu"),
]


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy)


def load_tk(path):
    d = json.load(open(path))
    out = {}
    for r in d.get("results") or d.get("scored") or []:
        did = r.get("doc_id")
        if did is None:
            continue
        if r.get("meteor") is not None:
            out[str(did)] = float(r["meteor"])
        elif r.get("score") is not None:
            try:
                s = int(r["score"])
            except (ValueError, TypeError):
                continue
            if s >= 0:
                out[str(did)] = s / 3.0
    return out


def load_trace_ordered(path):
    """Ordered list: [{correct, retrieved doc_ids}] by line."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "correct": int(r["correct"]) if r.get("correct") is not None else None,
                "retrieved": [str(d["doc_id"]) for d in r.get("retrieved_definitions", [])][:TOPK],
            })
    return rows


# ---- annotation as ordered list (index-aligned to every trace) ----
audit = [it for it in json.load(open(AUDIT)) if not it.get("error")]
anno = []
for it in audit:
    subj = ""
    try:
        subj = json.loads(it["row_key"]).get("subject", "")
    except Exception:
        pass
    anno.append({"gold": [str(x) for x in it.get("gold_inv_term_ids", [])],
                 "subject": subj or "unknown"})
N = len(anno)
print(f"annotation: {N} questions")

# ================= PART 1: recall / hit-rate (model-independent) =================
rag0 = load_trace_ordered(ROOT / f"outputs/traces/{PANEL[0][2]}__hybrid_rerank__defs_last.jsonl")
assert len(rag0) == N

tot = [0, 0.0, 0.0, 0, 0]          # n, sum_r1, sum_r3, hit1, hit3
by_subj = defaultdict(lambda: [0, 0, 0.0])  # subj -> [n, hit3, sum_r3]
for i in range(N):
    gold = set(anno[i]["gold"])
    if not gold:
        continue
    top = rag0[i]["retrieved"]
    top1, top3 = set(top[:1]), set(top[:3])
    r1, r3 = len(gold & top1) / len(gold), len(gold & top3) / len(gold)
    h1, h3 = int(bool(gold & top1)), int(bool(gold & top3))
    tot[0] += 1; tot[1] += r1; tot[2] += r3; tot[3] += h1; tot[4] += h3
    s = by_subj[anno[i]["subject"]]
    s[0] += 1; s[1] += h3; s[2] += r3

n = tot[0]
print("\n================ PART 1: RETRIEVAL RECALL vs ANNOTATED TERMS ================")
print(f"questions with >=1 annotated term: {n}")
print(f"  recall@1   = {tot[1]/n:.3f}    hit-rate@1 = {tot[3]/n:.3f}")
print(f"  recall@3   = {tot[2]/n:.3f}    hit-rate@3 = {tot[4]/n:.3f}")
print(f"  {'subject':<28} {'hit@3':>7} {'recall@3':>9} {'n':>5}")
for subj, (nn, h, rr) in sorted(by_subj.items(), key=lambda kv: -kv[1][0]):
    print(f"  {subj:<28} {h/nn:>7.3f} {rr/nn:>9.3f} {nn:>5}")

hit_mask = [bool(set(anno[i]["gold"]) & set(rag0[i]["retrieved"][:3])) for i in range(N)]

# ================= PART 2: correlations, full vs hit-subset =================
print("\n================ PART 2: KNOWLEDGE-OUTCOME CORRELATION, FULL vs HIT SUBSET ================")
print("Pearson r(mean term-knowledge over a question's annotated terms, outcome).")
print("gain = rag_correct - base_correct.   * = retrieval-hit subset (annotated term in top-3).")
hdr = (f"{'model':<14} {'metric':<7} | {'r(K,base)':>10} {'r(K,gain)':>10} {'n':>5} | "
       f"{'r(K,base)*':>11} {'r(K,gain)*':>11} {'n*':>5}")
print(hdr); print("-" * len(hdr))

for label, sstub, tstub in PANEL:
    base = load_trace_ordered(ROOT / f"outputs/traces/{tstub}__none__defs_first.jsonl")
    rag = load_trace_ordered(ROOT / f"outputs/traces/{tstub}__hybrid_rerank__defs_last.jsonl")
    for metric, suffix in [("METEOR", "_scored.json"), ("judge", "_judge_full.json")]:
        p = ROOT / f"outputs/analysis/term_eval/{sstub}{suffix}"
        if not os.path.exists(p):
            print(f"{label:<14} {metric:<7} | MISSING {p.name}")
            continue
        tk = load_tk(p)
        Kf, bf, gf, Kh, bh, gh = [], [], [], [], [], []
        for i in range(N):
            gold = anno[i]["gold"]
            vals = [tk[t] for t in gold if t in tk]
            if not vals:
                continue
            bc, rc = base[i]["correct"], rag[i]["correct"]
            if bc is None or rc is None:
                continue
            k, g = sum(vals) / len(vals), rc - bc
            Kf.append(k); bf.append(bc); gf.append(g)
            if hit_mask[i]:
                Kh.append(k); bh.append(bc); gh.append(g)
        print(f"{label:<14} {metric:<7} | "
              f"{pearson(Kf,bf):>+10.3f} {pearson(Kf,gf):>+10.3f} {len(Kf):>5} | "
              f"{pearson(Kh,bh):>+11.3f} {pearson(Kh,gh):>+11.3f} {len(Kh):>5}")
