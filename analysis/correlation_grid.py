"""
Re-run the per-question Pearson correlation analysis on the human-validated
annotation, and compare to the original LLM-annotation numbers reported in
Table 2.

Output:
  review_bundle/correlation_on_validated.json
  + console summary

For each model, computes the 4-column row (TK-METEOR r_corr, TK-METEOR r_Delta,
TK-judge r_corr, TK-judge r_Delta) under mean-K / min-K aggregation, on the
n_validated questions that have at least one validated annotated term.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("DEFRAG_ROOT", "."))
VALIDATED = ROOT / "review_bundle/mmlu_finance_annotated_human_validated.csv"


MODELS = [
    ("Qwen-2.5-7B",       "qwen7b"),
    ("Qwen-2.5-14B",      "qwen14b"),
    ("Llama-3.1-8B",      "llama31_8b"),
    ("Mistral-S-24B",     "mistral24b"),
    ("GPT-5.2",           "gpt52"),
]

TRACES = {
    "qwen7b": {
        "base": "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    "qwen14b": {
        "base": "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    "llama31_8b": {
        "base": "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__none__defs_first.jsonl",
        "rag":  "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    "mistral24b": {
        "base": "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__none__defs_first.jsonl",
        "rag":  "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__hybrid_rerank__defs_last.jsonl",
    },
    "gpt52": {
        "base": "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__none__defs_first.jsonl",
        "rag":  "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__hybrid_rerank__defs_last.jsonl",
    },
}


def _norm(s):
    return " ".join(s.replace(" ", " ").split()).strip().lower()


def load_trace_correct(path):
    out = {}
    with open(ROOT / path) as f:
        for line in f:
            r = json.loads(line)
            key = _norm(r.get("question", ""))
            out[key] = bool(r.get("correct", False))
    return out


def load_tk(model_slug):
    """Returns dict doc_id -> {'meteor': float, 'judge': int}."""
    out = defaultdict(dict)
    with open(ROOT / f"outputs/analysis/term_eval/{model_slug}_scored.json") as f:
        s = json.load(f)
    for r in s.get("results", []):
        doc_id = str(r.get("doc_id", ""))
        m = r.get("meteor")
        if m is not None and doc_id:
            out[doc_id]["meteor"] = float(m)
    with open(ROOT / f"outputs/analysis/term_eval/{model_slug}_judge_full.json") as f:
        j = json.load(f)
    for r in j.get("results", []):
        doc_id = str(r.get("doc_id", ""))
        score = r.get("score")
        if score is not None and doc_id:
            out[doc_id]["judge"] = float(score)
    return out


def load_validated_annotation():
    """question (normalised) -> list of doc_ids."""
    out = {}
    for row in csv.DictReader(open(VALIDATED, encoding="utf-8")):
        try:
            q = json.loads(row["question_id"])["question"]
        except Exception:
            continue
        ids = [s.strip() for s in (row.get("gold_inv_term_id") or "").split(";") if s.strip()]
        out[_norm(q)] = ids
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fisher_ci_half_width(n, alpha=0.05):
    if n <= 3:
        return float("inf")
    return math.tanh(1.96 / math.sqrt(n - 3))


def main():
    annot = load_validated_annotation()
    print(f"Loaded validated annotation for {len(annot)} questions")
    n_term_bearing = sum(1 for ids in annot.values() if ids)
    print(f"  with at least one validated term: {n_term_bearing}")
    print(f"  zero-term: {sum(1 for ids in annot.values() if not ids)}")
    print()

    summary = {"models": {}, "n_term_bearing": n_term_bearing,
               "fisher_half_width": fisher_ci_half_width(n_term_bearing)}
    print(f"Fisher-z 95% CI half-width at r=0 for n={n_term_bearing}: ±{summary['fisher_half_width']:.3f}")
    print()

    print(f"{'Model':<16}{'TK-MET r_corr':>22}{'TK-MET r_Δ':>20}{'TK-judge r_corr':>22}{'TK-judge r_Δ':>20}")
    print("-" * 100)

    for label, slug in MODELS:
        base = load_trace_correct(TRACES[slug]["base"])
        rag = load_trace_correct(TRACES[slug]["rag"])
        tk = load_tk(slug)

        rows_full = []
        for q_norm, doc_ids in annot.items():
            if not doc_ids:
                continue
            if q_norm not in base or q_norm not in rag:
                continue
            meteors = [tk[d]["meteor"] for d in doc_ids if d in tk and "meteor" in tk[d]]
            judges = [tk[d]["judge"] for d in doc_ids if d in tk and "judge" in tk[d]]
            if not meteors or not judges:
                continue
            bc = 1 if base[q_norm] else 0
            rc = 1 if rag[q_norm] else 0
            delta = rc - bc
            rows_full.append({
                "bc": bc, "delta": delta,
                "meteor_mean": sum(meteors) / len(meteors),
                "meteor_min": min(meteors),
                "judge_mean": sum(judges) / len(judges),
                "judge_min": min(judges),
            })

        n = len(rows_full)
        def pcells():
            cells = []
            for key in ("meteor_mean", "meteor_min", "judge_mean", "judge_min"):
                xs_corr = [r[key] for r in rows_full]
                ys_corr = [r["bc"] for r in rows_full]
                ys_delta = [r["delta"] for r in rows_full]
                cells.append((pearson(xs_corr, ys_corr), pearson(xs_corr, ys_delta)))
            return cells

        cells = pcells()
        # Pretty print: TK-METEOR rcorr/rdelta (mean/min), TK-judge rcorr/rdelta (mean/min)
        mm_rc, mm_rd = cells[0]
        mi_rc, mi_rd = cells[1]
        jm_rc, jm_rd = cells[2]
        ji_rc, ji_rd = cells[3]
        out = {
            "n": n,
            "meteor_rcorr_mean": mm_rc, "meteor_rcorr_min": mi_rc,
            "meteor_rdelta_mean": mm_rd, "meteor_rdelta_min": mi_rd,
            "judge_rcorr_mean":  jm_rc, "judge_rcorr_min":  ji_rc,
            "judge_rdelta_mean": jm_rd, "judge_rdelta_min": ji_rd,
        }
        summary["models"][label] = out

        def fmt2(a, b):
            return f"{a:+.2f}/{b:+.2f}"
        print(f"{label:<16}"
              f"{fmt2(mm_rc, mi_rc):>22}"
              f"{fmt2(mm_rd, mi_rd):>20}"
              f"{fmt2(jm_rc, ji_rc):>22}"
              f"{fmt2(jm_rd, ji_rd):>20}")

    # Find max absolute r across all cells
    max_abs = 0.0
    for m in summary["models"].values():
        for k, v in m.items():
            if k == "n": continue
            max_abs = max(max_abs, abs(v))
    summary["max_abs_r"] = max_abs
    print()
    print(f"Max |r| across all 40 cells: {max_abs:.3f}")
    print(f"Fisher-z 95% CI half-width:  ±{summary['fisher_half_width']:.3f}")
    print(f"All cells within half-width? {max_abs < summary['fisher_half_width']}")

    out_path = ROOT / "review_bundle/correlation_on_validated.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
