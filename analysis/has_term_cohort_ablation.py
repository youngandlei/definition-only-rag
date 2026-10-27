#!/usr/bin/env python3
"""
Task 0: has-term / no-term cohort ablation (mirror of CLEF finding).

For each model, partition the MMLU 1124-question set by whether the
top-K retrieved glossary terms overlap lexically with the question:
  - has_term: at least one retrieved term is a substring of the question
              OR shares ≥2 content tokens with it
  - no_term:  no such overlap

Compute baseline acc, RAG acc, and Δ per cohort. Mirror prediction:
on-topic (has_term) is where the action is; off-topic (no_term) flat.

Inputs all local: 5 models' baseline + hybrid_rerank RAG traces.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(ROOT)

EN_STOP = set("""a an the of in on to from at by for with as is are was were be been
being have has had do does did will would can could should may might
this that these those it its their his her our your my they them us we
which what who whom whose where when why how if then else than not no
yes both either neither all any some each every more most less few many
into onto upon about above below over under between against across""".split())


def tokens(text):
    out = set()
    for w in re.findall(r"\w+", text.lower()):
        if len(w) >= 3 and w not in EN_STOP and not w.isdigit():
            out.add(w)
    return out


def has_term(question, retrieved_terms):
    q_lower = question.lower()
    q_tokens = tokens(question)
    for term in retrieved_terms:
        if not term:
            continue
        if term.lower() in q_lower:
            return True
        if len(tokens(term) & q_tokens) >= 2:
            return True
    return False


def load_per_question(trace_path):
    """Return question_text -> (correct, retrieved_terms)."""
    out = {}
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            q = r.get("question")
            c = r.get("correct")
            if q is None or c is None:
                continue
            rd = r.get("retrieved_definitions") or []
            terms = [d.get("term") or d.get("term_name") or "" for d in rd]
            out[q] = (int(c), terms)
    return out


PANEL = [
    {
        "label": "Qwen-7B",
        "base": ROOT / "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  ROOT / "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    {
        "label": "Qwen-14B",
        "base": ROOT / "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  ROOT / "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    {
        "label": "GPT-5.2",
        "base": ROOT / "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__none__defs_first.jsonl",
        "rag":  ROOT / "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    {
        "label": "Mistral-24B",
        "base": ROOT / "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__none__defs_first.jsonl",
        "rag":  ROOT / "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    {
        "label": "Llama-3.1-8B",
        "base": ROOT / "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  ROOT / "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
]


def analyse(label, base_path, rag_path):
    base = load_per_question(base_path)
    rag  = load_per_question(rag_path)
    shared = sorted(set(base) & set(rag))

    has, no = [], []
    for q in shared:
        b_correct, _ = base[q]
        r_correct, retrieved_terms = rag[q]  # use RAG's retrieval (it's the run we're analyzing)
        flag = has_term(q, retrieved_terms)
        rec = (b_correct, r_correct)
        (has if flag else no).append(rec)

    def stats(records):
        n = len(records)
        if n == 0:
            return {"n": 0, "base_acc": None, "rag_acc": None, "delta": None}
        b = sum(x[0] for x in records) / n
        r = sum(x[1] for x in records) / n
        return {"n": n, "base_acc": b, "rag_acc": r, "delta": r - b}

    h_stats = stats(has)
    n_stats = stats(no)

    full_n = len(has) + len(no)
    full_base = (sum(x[0] for x in has) + sum(x[0] for x in no)) / full_n
    full_rag  = (sum(x[1] for x in has) + sum(x[1] for x in no)) / full_n
    full_delta = full_rag - full_base

    # Sanity: weighted reconstruction
    if h_stats["n"] and n_stats["n"]:
        reweight = (h_stats["delta"] * h_stats["n"] + n_stats["delta"] * n_stats["n"]) / full_n
    else:
        reweight = None

    return {
        "label": label,
        "has_term": h_stats,
        "no_term":  n_stats,
        "full": {"n": full_n, "base_acc": full_base, "rag_acc": full_rag, "delta": full_delta},
        "reweight_check_delta": reweight,
    }


def main():
    out = []
    print(f"\n{'Model':<14} {'Cohort':<10} {'n':>5} {'base':>8} {'rag':>8} {'Δ':>8}")
    print("-" * 60)
    full_results = []
    for cfg in PANEL:
        a = analyse(cfg["label"], cfg["base"], cfg["rag"])
        out.append(a)
        for cohort, key in [("has_term", "has_term"), ("no_term", "no_term"), ("full", "full")]:
            s = a[key]
            print(f"{cfg['label']:<14} {cohort:<10} {s['n']:>5} "
                  f"{s['base_acc']*100:>7.2f}% {s['rag_acc']*100:>7.2f}% "
                  f"{s['delta']*100:>+7.2f}")
        print(f"{'  reweight check':<25} delta = {a['reweight_check_delta']*100 if a['reweight_check_delta'] is not None else 'NA':>+7.4f}  (should match 'full' Δ)")
        print()

    # Save
    out_path = ROOT / "outputs/analysis/annotation/has_term_cohort_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
