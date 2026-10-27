"""
Paired bootstrap 95% CIs for Table 1 (Δ_RAG per model × benchmark).

For each (model, benchmark), load baseline and RAG trace JSONL,
match per question, compute paired bootstrap on (RAG_correct - base_correct).
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(ROOT)

CONFIGS = [
    # (label, baseline_path, rag_path)
    ("Qwen-2.5-7B / MMLU",
     "outputs/traces/archive/traces_qwen7/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl",
     "outputs/traces/archive/traces_qwen7/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl"),
    ("Qwen-2.5-14B / MMLU",
     "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__none__defs_first.jsonl",
     "outputs/traces/main/2026-04-08_qwen14_rerun_clean/vllm_qwen_qwen2_5_14b_instruct__mmlu__hybrid_rerank__defs_last.jsonl"),
    ("Llama-3.1-8B / MMLU",
     "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__none__defs_first.jsonl",
     "outputs/traces/archive/traces_0310_llama/vllm_meta_llama_llama_3_1_8b_instruct__mmlu__hybrid_rerank__defs_last.jsonl"),
    ("Mistral-S-24B / MMLU",
     "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__none__defs_first.jsonl",
     "outputs/traces/archive/traces_0310_mistral24b/vllm_mistralai_mistral_small_24b_instruct_2501__mmlu__hybrid_rerank__defs_last.jsonl"),
    ("GPT-5.2 / MMLU",
     "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__none__defs_first.jsonl",
     "outputs/traces/archive/traces_gpt/openai_gpt_5_2__mmlu__hybrid_rerank__defs_last.jsonl"),
    ("Qwen-2.5-7B / XFinBench TU",
     "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__none__defs_first.jsonl",
     None),
    ("Qwen-2.5-14B / XFinBench TU",
     "outputs/traces/main/2026-04-14_qwen14b_xfinbench/vllm_qwen_qwen2_5_14b_instruct__xfinbench__none__defs_first.jsonl",
     "outputs/traces/main/2026-04-14_qwen14b_xfinbench/vllm_qwen_qwen2_5_14b_instruct__xfinbench__hybrid_rerank__defs_last.jsonl"),
]


def _norm(s):
    # Aggressive normalisation so per-question matching survives whitespace / quote / unicode differences.
    return " ".join(s.replace(" ", " ").split()).strip().lower()


def load_correct(path):
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Prefer row_key if present, else normalised question.
            key = r.get("row_key") or _norm(r.get("question", ""))
            if not key:
                continue
            records[key] = bool(r.get("correct", False))
    return records


def paired_bootstrap(base, rag, n_boot=2000, seed=42, alpha=0.05):
    """
    base, rag: 1D arrays of {0,1} per matched question.
    Returns (mean_delta_pp, lo_pp, hi_pp).
    """
    rng = np.random.default_rng(seed)
    n = len(base)
    diff = rag.astype(np.int32) - base.astype(np.int32)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = diff[idx].mean()
    lo, hi = np.quantile(boot_deltas, [alpha / 2, 1 - alpha / 2])
    return diff.mean() * 100, lo * 100, hi * 100


# Look up Qwen-7B XFinBench RAG path
qwen7_xfin_dir = ROOT / "outputs/traces/main/2026-04-14_qwen7_xfinbench"
for p in qwen7_xfin_dir.glob("*hybrid_rerank*.jsonl"):
    CONFIGS[5] = (CONFIGS[5][0], CONFIGS[5][1], str(p.relative_to(ROOT)))
    break

print(f"{'config':<32} {'n':>6} {'base':>7} {'rag':>7} {'Δ (pp)':>9} {'95% CI':>20}")
print("-" * 85)

for label, base_path, rag_path in CONFIGS:
    if rag_path is None:
        print(f"{label:<32}  (no RAG trace found)")
        continue
    bp = ROOT / base_path
    rp = ROOT / rag_path
    if not bp.exists() or not rp.exists():
        print(f"{label:<32}  MISSING: {bp.exists()=} {rp.exists()=}")
        continue
    base_map = load_correct(bp)
    rag_map = load_correct(rp)
    keys = sorted(set(base_map) & set(rag_map))
    base_arr = np.array([base_map[k] for k in keys], dtype=np.int32)
    rag_arr = np.array([rag_map[k] for k in keys], dtype=np.int32)
    base_acc = base_arr.mean() * 100
    rag_acc = rag_arr.mean() * 100
    mean_d, lo, hi = paired_bootstrap(base_arr, rag_arr)
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"{label:<32} {len(keys):>6} {base_acc:>7.2f} {rag_acc:>7.2f} {mean_d:>+8.2f}{sig} [{lo:>+6.2f}, {hi:>+6.2f}]")

print()
print("* = 95% CI excludes zero (paired bootstrap, n_boot=2000, seed=42)")
