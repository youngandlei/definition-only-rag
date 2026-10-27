#!/usr/bin/env python3
"""
Hidden-State Probe for RAG necessity prediction.

Extract hidden states from intermediate transformer layers for each question
(both baseline and RAG prompts), then train a linear probe to predict whether
RAG helps.

Usage:
  Step 1: Extract hidden states (requires GPU + model)
    python hidden_state_probe.py extract \
      --baseline_trace ... --rag_trace ... \
      --model Qwen/Qwen2.5-7B-Instruct \
      --layers 0,8,16,24,28 \
      --output_dir /path/to/hidden_states/

  Step 2: Train probe (can run on CPU)
    python hidden_state_probe.py probe \
      --baseline_trace ... --rag_trace ... \
      --hidden_dir /path/to/hidden_states/ \
      --output /path/to/results.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Trace loading (shared)
# ---------------------------------------------------------------------------

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


def build_labels(base_trace, rag_trace):
    """Build labels and splits from traces."""
    keys = sorted(set(base_trace) & set(rag_trace))
    records = []
    for k in keys:
        b = base_trace[k]
        r = rag_trace[k]
        agree = b["prediction"] == r["prediction"]
        if agree:
            label = 0
            cat = "agree"
        elif r["correct"] and not b["correct"]:
            label = 1
            cat = "rag_wins"
        elif b["correct"] and not r["correct"]:
            label = 0
            cat = "base_wins"
        else:
            label = -1
            cat = "both_wrong"
        records.append({"key": k, "label": label, "category": cat, "agree": agree})
    return keys, records


# ---------------------------------------------------------------------------
# Step 1: Extract hidden states
# ---------------------------------------------------------------------------

def extract_hidden_states(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base_trace = load_trace(args.baseline_trace)
    rag_trace = load_trace(args.rag_trace)
    keys, records = build_labels(base_trace, rag_trace)

    # Parse layers
    if args.layers == "all":
        layers_to_extract = None  # will be set after model loads
    else:
        layers_to_extract = [int(x) for x in args.layers.split(",")]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if layers_to_extract is None:
        # Sample evenly: first, 1/4, 1/2, 3/4, last
        layers_to_extract = sorted(set([
            0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1
        ]))

    print(f"Model has {n_layers} layers. Extracting layers: {layers_to_extract}")
    print(f"Total questions: {len(keys)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooling_modes = [p.strip() for p in args.pooling.split(",")]
    print(f"Pooling modes: {pooling_modes}")

    def get_hidden(prompt: str) -> dict:
        """Get hidden states with multiple pooling strategies."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_length).to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
        hs = {}
        for layer_idx in layers_to_extract:
            h_full = outputs.hidden_states[layer_idx + 1][0]  # [seq_len, hidden_dim]
            for pool in pooling_modes:
                if pool == "last":
                    h = h_full[-1, :].cpu().float().numpy()
                elif pool == "mean":
                    h = h_full.mean(dim=0).cpu().float().numpy()
                elif pool == "max":
                    h = h_full.max(dim=0).values.cpu().float().numpy()
                elif pool == "first":
                    h = h_full[0, :].cpu().float().numpy()
                else:
                    continue
                hs[(layer_idx, pool)] = h
        return hs

    # Extract for both conditions — accumulate per (layer, pool) arrays
    # Pre-allocate arrays for memory efficiency
    array_keys_list = [(li, pool) for li in layers_to_extract for pool in pooling_modes]
    base_arrays = {k: [] for k in array_keys_list}
    rag_arrays = {k: [] for k in array_keys_list}

    for i, k in enumerate(keys):
        b_prompt = base_trace[k]["prompt"]
        r_prompt = rag_trace[k]["prompt"]

        b_hs = get_hidden(b_prompt)
        r_hs = get_hidden(r_prompt)

        for ak in array_keys_list:
            base_arrays[ak].append(b_hs[ak])
            rag_arrays[ak].append(r_hs[ak])

        if (i + 1) % 50 == 0 or (i + 1) == len(keys):
            print(f"  {i+1}/{len(keys)} done")

    # Save
    meta = {
        "keys": keys,
        "layers": layers_to_extract,
        "pooling_modes": pooling_modes,
        "n_layers_total": n_layers,
        "model": args.model,
        "hidden_dim": model.config.hidden_size,
    }

    for (li, pool) in array_keys_list:
        b_arr = np.array(base_arrays[(li, pool)], dtype=np.float32)
        r_arr = np.array(rag_arrays[(li, pool)], dtype=np.float32)
        np.save(str(out_dir / f"base_layer{li}_{pool}.npy"), b_arr)
        np.save(str(out_dir / f"rag_layer{li}_{pool}.npy"), r_arr)
        print(f"Saved layer {li} {pool}: base {b_arr.shape}, rag {r_arr.shape}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Extraction complete. Saved to {out_dir}")


# ---------------------------------------------------------------------------
# Step 2: Train probe
# ---------------------------------------------------------------------------

def train_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    base_trace = load_trace(args.baseline_trace)
    rag_trace = load_trace(args.rag_trace)
    keys, records = build_labels(base_trace, rag_trace)

    n = len(keys)
    acc_base = sum(base_trace[k]["correct"] for k in keys) / n
    acc_rag = sum(rag_trace[k]["correct"] for k in keys) / n
    oracle = sum(max(base_trace[k]["correct"], rag_trace[k]["correct"]) for k in keys) / n

    disagree_keys = [k for k in keys if base_trace[k]["prediction"] != rag_trace[k]["prediction"]]
    print(f"Total: {n}, disagree: {len(disagree_keys)}")
    print(f"Baseline: {acc_base:.4f} | RAG: {acc_rag:.4f} | Oracle: {oracle:.4f}")

    # Load meta
    hidden_dir = Path(args.hidden_dir)
    with open(hidden_dir / "meta.json") as f:
        meta = json.load(f)

    layers = meta["layers"]
    stored_keys = meta["keys"]

    # Build key -> index mapping
    key_to_idx = {k: i for i, k in enumerate(stored_keys)}

    # Build label arrays
    rec_map = {r["key"]: r for r in records}

    # Indices for train (exclude both_wrong) and all
    train_mask = []
    train_labels = []
    all_labels = []
    for k in stored_keys:
        r = rec_map[k]
        all_labels.append(r["label"])
        if r["label"] != -1:
            train_mask.append(key_to_idx[k])
            train_labels.append(r["label"])

    train_mask = np.array(train_mask)
    train_labels = np.array(train_labels)

    # Disagree-only
    disagree_mask = []
    disagree_labels = []
    for k in stored_keys:
        r = rec_map[k]
        if not r["agree"] and r["label"] != -1:
            disagree_mask.append(key_to_idx[k])
            disagree_labels.append(r["label"])
    disagree_mask = np.array(disagree_mask)
    disagree_labels = np.array(disagree_labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_predictions(y_pred_map):
        """Given key->prediction map, compute overall + disagree accuracy."""
        correct = 0
        d_correct = 0
        d_total = 0
        chose_base = 0
        chose_rag = 0
        for k in keys:
            b = base_trace[k]
            r = rag_trace[k]
            agree = b["prediction"] == r["prediction"]
            use_rag = y_pred_map.get(k, 1) == 1  # default RAG for both_wrong
            chosen = r if use_rag else b
            correct += chosen["correct"]
            if not agree:
                d_total += 1
                d_correct += chosen["correct"]
                if use_rag:
                    chose_rag += 1
                else:
                    chose_base += 1
        return correct / n, d_correct, d_total, chose_base, chose_rag

    print(f"\n{'Config':<45} {'Overall':>8} {'Δ RAG':>7} {'Disagree':>10} {'base':>5} {'rag':>5}")
    print("-" * 80)
    print(f"{'Always RAG':<45} {acc_rag:>8.4f} {0:>+7.4f} {sum(rag_trace[k]['correct'] for k in disagree_keys):>7}/{len(disagree_keys):<3}")

    # --- Probe each layer and combination ---
    representations = {
        "base": {},    # layer -> [n, hidden_dim]
        "rag": {},
        "delta": {},   # rag - base
        "concat": {},  # [base; rag]
    }

    for li in layers:
        # Support both old format (base_layer{li}.npy) and new format (base_layer{li}_last.npy)
        base_path = hidden_dir / f"base_layer{li}.npy"
        if not base_path.exists():
            base_path = hidden_dir / f"base_layer{li}_last.npy"
        rag_path_h = hidden_dir / f"rag_layer{li}.npy"
        if not rag_path_h.exists():
            rag_path_h = hidden_dir / f"rag_layer{li}_last.npy"
        base_h = np.load(str(base_path))
        rag_h = np.load(str(rag_path_h))
        representations["base"][li] = base_h
        representations["rag"][li] = rag_h
        representations["delta"][li] = rag_h - base_h
        representations["concat"][li] = np.concatenate([base_h, rag_h], axis=1)

    classifiers = {
        "LogReg": lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "SVM": lambda: LinearSVC(C=1.0, max_iter=5000, class_weight="balanced"),
    }

    best_name, best_acc = None, 0
    all_results = {}

    for rep_name in ["base", "rag", "delta", "concat"]:
        for li in layers:
            X_full = representations[rep_name][li]

            for train_mode in ["all", "disagree_only"]:
                if train_mode == "all":
                    X_tr = X_full[train_mask]
                    y_tr = train_labels
                    suffix = ""
                else:
                    X_tr = X_full[disagree_mask]
                    y_tr = disagree_labels
                    suffix = "(d)"

                for clf_name, clf_fn in classifiers.items():
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)

                    clf = clf_fn()
                    try:
                        y_cv = cross_val_predict(clf, X_tr_s, y_tr, cv=cv)
                    except Exception:
                        continue

                    # Map predictions back
                    pred_map = {}
                    if train_mode == "all":
                        for idx, pred in zip(train_mask, y_cv):
                            pred_map[stored_keys[idx]] = int(pred)
                    else:
                        for idx, pred in zip(disagree_mask, y_cv):
                            pred_map[stored_keys[idx]] = int(pred)

                    acc, d_correct, d_total, cb, cr = evaluate_predictions(pred_map)

                    label = f"{rep_name}_L{li}{suffix}"
                    print(f"{label:<45} {clf_name:<6} {acc:>8.4f} "
                          f"{acc - acc_rag:>+7.4f} "
                          f"{d_correct:>7}/{d_total:<3} {cb:>5} {cr:>5}")

                    key = f"{label}__{clf_name}"
                    all_results[key] = {
                        "overall_acc": acc,
                        "delta_vs_rag": acc - acc_rag,
                        "disagree_correct": d_correct,
                        "chose_base": cb,
                        "chose_rag": cr,
                    }
                    if acc > best_acc:
                        best_acc = acc
                        best_name = key

    print(f"\n{'Oracle':<45} {'':<6} {oracle:>8.4f} {oracle-acc_rag:>+7.4f}")
    print(f"\nBest: {best_name} → {best_acc:.4f} (Δ vs RAG: {best_acc-acc_rag:+.4f})")

    if args.output:
        summary = {
            "baseline_acc": acc_base,
            "rag_acc": acc_rag,
            "oracle_acc": oracle,
            "n_total": n,
            "n_disagree": len(disagree_keys),
            "layers": layers,
            "best_config": best_name,
            "best_acc": best_acc,
            "results": all_results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"Saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")

    # Extract
    ext = sub.add_parser("extract", help="Extract hidden states from model")
    ext.add_argument("--baseline_trace", required=True)
    ext.add_argument("--rag_trace", required=True)
    ext.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ext.add_argument("--layers", default="auto",
                     help="Comma-separated layer indices, 'all', or 'auto'")
    ext.add_argument("--max_length", type=int, default=2048)
    ext.add_argument("--pooling", default="last",
                     help="Comma-separated pooling modes: last,mean,max,first")
    ext.add_argument("--output_dir", required=True)

    # Probe
    prb = sub.add_parser("probe", help="Train probe on extracted hidden states")
    prb.add_argument("--baseline_trace", required=True)
    prb.add_argument("--rag_trace", required=True)
    prb.add_argument("--hidden_dir", required=True)
    prb.add_argument("--output", default="")

    args = ap.parse_args()
    if args.command == "extract":
        if args.layers == "auto":
            args.layers = "all"  # will be handled inside
        extract_hidden_states(args)
    elif args.command == "probe":
        train_probe(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
