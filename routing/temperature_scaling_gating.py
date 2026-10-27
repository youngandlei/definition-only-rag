"""
Per-condition temperature scaling on the disagree slice, then re-do
confidence/entropy-based gating with calibrated probabilities.

Setup
-----
choice_scores in the trace files are raw logprobs over A/B/C/D (we verified that
softmax(choice_scores) reproduces score_confidence).

For each (setting, condition), we
  1. Fit T per-condition on the disagree slice using 5-fold CV: T* minimizes NLL
     of softmax(logits / T) against the gold label.
  2. Apply the held-out fold's T to compute calibrated max-prob and entropy.
  3. Gate baseline vs RAG per question using calibrated signals:
     - "low-entropy wins": pick the side whose calibrated entropy is lower
     - "high-confidence wins": pick the side whose calibrated max-prob is higher
     - "weighted aggregation": w_base * cal_conf_base vs w_rag * cal_conf_rag with
        w_rag > w_base, since RAG is the stronger default.
"""

import json
import math
from pathlib import Path
import random

ROOT = Path(ROOT)
OUT = ROOT / "outputs/analysis/calibration"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)


def load_jsonl(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["row_key"]] = r
    return out


def softmax_T(logits, T):
    mx = max(logits)
    exps = [math.exp((x - mx) / T) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def nll(records, T):
    """Negative log likelihood of the gold label under softmax(logits/T).
    records: list of (logits, gold_idx). Skips invalid."""
    total = 0.0
    n = 0
    for logits, gold in records:
        if logits is None or any(x is None for x in logits) or gold is None:
            continue
        p = softmax_T(logits, T)
        pg = max(p[gold], 1e-12)
        total -= math.log(pg)
        n += 1
    return total / n if n else float("inf")


def ece(records, T, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for logits, gold in records:
        if logits is None or any(x is None for x in logits) or gold is None:
            continue
        p = softmax_T(logits, T)
        pred = max(range(len(p)), key=lambda k: p[k])
        conf = p[pred]
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, int(pred == gold)))
    total = sum(len(b) for b in bins)
    if total == 0:
        return 0.0
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(x[0] for x in b) / len(b)
        acc = sum(x[1] for x in b) / len(b)
        e += (len(b) / total) * abs(avg_c - acc)
    return e


def fit_temperature(records, grid=None):
    """Grid search T minimizing NLL on the given records."""
    if grid is None:
        grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    best_T, best_l = 1.0, float("inf")
    for T in grid:
        l = nll(records, T)
        if l < best_l:
            best_T, best_l = T, l
    # Local refinement around best
    lo = max(0.05, best_T * 0.7)
    hi = best_T * 1.4
    fine = [lo + (hi - lo) * i / 30 for i in range(31)]
    for T in fine:
        l = nll(records, T)
        if l < best_l:
            best_T, best_l = T, l
    return best_T, best_l


def kfold_indices(n, k=5):
    idx = list(range(n))
    random.Random(42).shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    return folds


def evaluate_setting(label, base_path, rag_path, k_folds=5):
    base = load_jsonl(base_path)
    rag = load_jsonl(rag_path)
    keys = sorted(set(base) & set(rag))

    pairs = []
    for k in keys:
        b, r = base[k], rag[k]
        pairs.append({"key": k, "b": b, "r": r,
                      "agree": b["prediction"] == r["prediction"]})

    disagree = [p for p in pairs if not p["agree"]
                and p["b"].get("choice_scores") and p["r"].get("choice_scores")
                and all(x is not None for x in p["b"]["choice_scores"])
                and all(x is not None for x in p["r"]["choice_scores"])]
    print(f"[{label}] disagree={len(disagree)} valid pairs")

    folds = kfold_indices(len(disagree), k_folds)

    # We need per-side records for fitting T.
    def side_record(p, side):
        rec = p["b"] if side == "base" else p["r"]
        return (rec["choice_scores"], rec["gold"])

    # Hold out one fold at a time; fit T on the rest; compute calibrated metrics on held-out.
    cal_results = []  # one entry per disagree pair, in original order
    for _ in range(len(disagree)):
        cal_results.append(None)

    fit_Ts_base, fit_Ts_rag = [], []
    for f_idx in range(k_folds):
        train_idx = [i for fi, fold in enumerate(folds) if fi != f_idx for i in fold]
        test_idx = folds[f_idx]
        train_base = [side_record(disagree[i], "base") for i in train_idx]
        train_rag  = [side_record(disagree[i], "rag")  for i in train_idx]
        Tb, _ = fit_temperature(train_base)
        Tr, _ = fit_temperature(train_rag)
        fit_Ts_base.append(Tb)
        fit_Ts_rag.append(Tr)
        for i in test_idx:
            p = disagree[i]
            b_logits = p["b"]["choice_scores"]
            r_logits = p["r"]["choice_scores"]
            pb = softmax_T(b_logits, Tb)
            pr = softmax_T(r_logits, Tr)
            pred_b = max(range(len(pb)), key=lambda k: pb[k])
            pred_r = max(range(len(pr)), key=lambda k: pr[k])
            # entropy (natural log)
            eb = -sum(x * math.log(max(x, 1e-12)) for x in pb)
            er = -sum(x * math.log(max(x, 1e-12)) for x in pr)
            cal_results[i] = {
                "Tb": Tb, "Tr": Tr,
                "cal_conf_b": pb[pred_b], "cal_conf_r": pr[pred_r],
                "cal_ent_b": eb, "cal_ent_r": er,
                "pred_b": pred_b, "pred_r": pred_r,
                "gold": p["b"]["gold"],
                "b_correct": p["b"]["correct"],
                "r_correct": p["r"]["correct"],
                "raw_conf_b": p["b"]["score_confidence"],
                "raw_conf_r": p["r"]["score_confidence"],
                "raw_ent_b": p["b"]["score_entropy"],
                "raw_ent_r": p["r"]["score_entropy"],
            }

    # Baselines
    n = len(disagree)
    always_rag = sum(c["r_correct"] for c in cal_results)
    always_base = sum(c["b_correct"] for c in cal_results)
    oracle = sum(max(c["b_correct"], c["r_correct"]) for c in cal_results)

    def gating(pick_fn):
        correct = 0
        chose_b = 0
        for c in cal_results:
            side = pick_fn(c)
            chose_b += (side == "base")
            correct += c["b_correct"] if side == "base" else c["r_correct"]
        return correct, chose_b, n - chose_b

    # 1. Raw entropy gating (lower entropy wins)
    raw_ent = gating(lambda c: "base" if c["raw_ent_b"] < c["raw_ent_r"] else "rag")
    # 2. Calibrated entropy gating (lower calibrated entropy wins)
    cal_ent = gating(lambda c: "base" if c["cal_ent_b"] < c["cal_ent_r"] else "rag")
    # 3. Raw confidence gating (higher max-prob wins)
    raw_conf = gating(lambda c: "base" if c["raw_conf_b"] > c["raw_conf_r"] else "rag")
    # 4. Calibrated confidence gating (higher calibrated max-prob wins)
    cal_conf = gating(lambda c: "base" if c["cal_conf_b"] > c["cal_conf_r"] else "rag")
    # 5. Weighted aggregation: w_rag * cal_conf_r vs w_base * cal_conf_b
    weighted = {}
    for w_rag in [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]:
        weighted[w_rag] = gating(
            lambda c, wr=w_rag: "base" if c["cal_conf_b"] > wr * c["cal_conf_r"] else "rag"
        )

    # ECE on cal results
    def slice_ece(side):
        recs = []
        for c in cal_results:
            pred = c["pred_b"] if side == "base" else c["pred_r"]
            conf = c["cal_conf_b"] if side == "base" else c["cal_conf_r"]
            recs.append((conf, int(pred == c["gold"])))
        bins = [[] for _ in range(10)]
        for conf, ok in recs:
            idx = min(int(conf * 10), 9)
            bins[idx].append((conf, ok))
        total = sum(len(b) for b in bins)
        e = 0.0
        for b in bins:
            if not b:
                continue
            avg_c = sum(x[0] for x in b) / len(b)
            acc = sum(x[1] for x in b) / len(b)
            e += (len(b) / total) * abs(avg_c - acc)
        return e

    summary = {
        "n_disagree": n,
        "always_rag": always_rag,
        "always_base": always_base,
        "oracle": oracle,
        "fit_T_base_per_fold": fit_Ts_base,
        "fit_T_rag_per_fold":  fit_Ts_rag,
        "ECE_base_disagree_calibrated": slice_ece("base"),
        "ECE_rag_disagree_calibrated":  slice_ece("rag"),
        "avg_cal_conf_base": sum(c["cal_conf_b"] for c in cal_results) / n,
        "avg_cal_conf_rag":  sum(c["cal_conf_r"] for c in cal_results) / n,
        "avg_cal_ent_base":  sum(c["cal_ent_b"]  for c in cal_results) / n,
        "avg_cal_ent_rag":   sum(c["cal_ent_r"]  for c in cal_results) / n,
        "gating": {
            "raw_entropy":          {"correct": raw_ent[0], "chose_base": raw_ent[1], "chose_rag": raw_ent[2]},
            "calibrated_entropy":   {"correct": cal_ent[0], "chose_base": cal_ent[1], "chose_rag": cal_ent[2]},
            "raw_confidence":       {"correct": raw_conf[0], "chose_base": raw_conf[1], "chose_rag": raw_conf[2]},
            "calibrated_confidence":{"correct": cal_conf[0], "chose_base": cal_conf[1], "chose_rag": cal_conf[2]},
            "weighted": {str(w): {"correct": v[0], "chose_base": v[1], "chose_rag": v[2]}
                         for w, v in weighted.items()},
        },
    }
    return summary


def main():
    settings = [
        ("qwen7b_mmlu",
         ROOT / "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__none__defs_first.jsonl",
         ROOT / "outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl"),
        ("qwen7b_xfinbench",
         ROOT / "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__none__defs_first.jsonl",
         ROOT / "outputs/traces/main/2026-04-14_qwen7_xfinbench/vllm_qwen_qwen2_5_7b_instruct__xfinbench__hybrid_rerank__defs_last.jsonl"),
    ]
    all_results = {}
    for label, b, r in settings:
        print(f"\n=== {label} ===")
        s = evaluate_setting(label, b, r)
        all_results[label] = s
        n = s["n_disagree"]
        print(f"  n_disagree={n}  always_rag={s['always_rag']}  always_base={s['always_base']}  oracle={s['oracle']}")
        print(f"  fit T (base per-fold): {[round(t,3) for t in s['fit_T_base_per_fold']]}")
        print(f"  fit T (rag  per-fold): {[round(t,3) for t in s['fit_T_rag_per_fold']]}")
        print(f"  ECE (cal, disagree)  base={s['ECE_base_disagree_calibrated']:.4f}  rag={s['ECE_rag_disagree_calibrated']:.4f}")
        print(f"  avg cal conf  base={s['avg_cal_conf_base']:.4f}  rag={s['avg_cal_conf_rag']:.4f}")
        print(f"  avg cal ent   base={s['avg_cal_ent_base']:.4f}  rag={s['avg_cal_ent_rag']:.4f}")
        for k, v in s["gating"].items():
            if k == "weighted":
                continue
            print(f"  gating[{k:24s}]: {v['correct']}/{n}  base={v['chose_base']}  rag={v['chose_rag']}")
        print(f"  weighted (w_rag * cal_conf_r vs cal_conf_b):")
        for w, v in s["gating"]["weighted"].items():
            print(f"    w_rag={w}: {v['correct']}/{n}  base={v['chose_base']}  rag={v['chose_rag']}")

    out_path = OUT / "temperature_scaling_2026-05-13.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
