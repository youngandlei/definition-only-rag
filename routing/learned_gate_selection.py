#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def load_trace(path: str):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["row_key"]] = obj
    return rows


def parse_row_key(row_key: str):
    try:
        return json.loads(row_key)
    except Exception:
        return {}


def tokenize(text):
    return {t.lower() for t in TOKEN_RE.findall(text or "")}


def safe_float(x):
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return None
    return v


def stats(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return vals[0], (vals[0] - vals[1]) if len(vals) > 1 else vals[0], mean, math.sqrt(var)


def retrieval_features(row):
    defs = row.get("retrieved_definitions") or []
    scores = [safe_float(d.get("score")) for d in defs if isinstance(d, dict)]
    top_score, gap12, mean_score, std_score = stats(scores)
    question_toks = tokenize(row.get("question", ""))
    term_toks = set()
    text_toks = set()
    for d in defs:
        if isinstance(d, dict):
            term_toks |= tokenize(d.get("term", ""))
            text_toks |= tokenize(d.get("text", ""))
    term_overlap = len(question_toks & term_toks)
    text_overlap = len(question_toks & text_toks)
    return {
        "rag_num_defs": float(len(defs)),
        "rag_top_score": top_score,
        "rag_score_gap12": gap12,
        "rag_score_mean": mean_score,
        "rag_score_std": std_score,
        "rag_term_overlap": float(term_overlap),
        "rag_text_overlap": float(text_overlap),
    }


def question_features(row, meta):
    question = row.get("question", "")
    q_lower = question.lower()
    q_tokens = tokenize(question)
    choices = meta.get("choices") or []
    choice_lens = [len(tokenize(c)) for c in choices]
    return {
        "q_token_count": float(len(q_tokens)),
        "q_char_count": float(len(question)),
        "choice_count": float(len(choices)),
        "choice_avg_tokens": float(sum(choice_lens) / len(choice_lens)) if choice_lens else 0.0,
        "has_number": float(bool(re.search(r"\d", question))),
        "has_percent": float("%" in question or "percent" in q_lower),
        "asks_definition": float("what is" in q_lower or "which of the following is true" in q_lower),
        "asks_incorrect": float("incorrect" in q_lower or "not true" in q_lower or "except" in q_lower),
        "asks_calculation": float(any(
            k in q_lower for k in ["calculate", "what amount", "what is the value", "maximum amount", "how much"]
        )),
        "asks_hypothesis": float(any(
            k in q_lower for k in ["hypothesis", "confidence interval", "significance level", "regression"]
        )),
    }


def build_feature_row(base_row, rag_row, meta):
    bm = safe_float(base_row.get("score_margin"))
    rm = safe_float(rag_row.get("score_margin"))
    be = safe_float(base_row.get("score_entropy"))
    re_ = safe_float(rag_row.get("score_entropy"))
    bc = safe_float(base_row.get("score_confidence"))
    rc = safe_float(rag_row.get("score_confidence"))

    feats = {
        "baseline_margin": bm,
        "rag_margin": rm,
        "delta_margin": None if bm is None or rm is None else rm - bm,
        "baseline_entropy": be,
        "rag_entropy": re_,
        "delta_entropy": None if be is None or re_ is None else be - re_,
        "baseline_confidence": bc,
        "rag_confidence": rc,
        "delta_confidence": None if bc is None or rc is None else rc - bc,
        "same_answer": float(base_row.get("prediction") == rag_row.get("prediction")),
    }
    feats.update(question_features(base_row, meta))
    feats.update(retrieval_features(rag_row))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_trace", required=True)
    ap.add_argument("--rag_trace", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group_by", choices=["none", "subject"], default="none")
    args = ap.parse_args()

    base = load_trace(args.baseline_trace)
    rag = load_trace(args.rag_trace)
    keys = sorted(set(base) & set(rag))
    if not keys:
        raise SystemExit("No overlapping row_key values found.")

    feature_names = None
    all_features = []
    all_keys = []
    all_labels = []
    disagreement_mask = []
    all_groups = []

    for k in keys:
        meta = parse_row_key(k)
        feats = build_feature_row(base[k], rag[k], meta)
        if feature_names is None:
            feature_names = list(feats.keys())
        all_features.append([feats[name] for name in feature_names])
        all_keys.append(k)
        b = int(base[k]["correct"])
        r = int(rag[k]["correct"])
        all_labels.append(1 if r > b else 0)
        disagreement_mask.append(base[k].get("prediction") != rag[k].get("prediction"))
        all_groups.append(meta.get("subject", "__unknown__"))

    train_indices = [i for i, d in enumerate(disagreement_mask) if d]
    X_train = [all_features[i] for i in train_indices]
    y_train = [all_labels[i] for i in train_indices]
    groups_train = [all_groups[i] for i in train_indices]

    positives = sum(y_train)
    negatives = len(y_train) - positives
    if len(train_indices) < 10 or positives == 0 or negatives == 0:
        raise SystemExit("Not enough disagreement examples with both classes to train a learned gate.")

    if args.group_by == "subject":
        unique_groups = sorted(set(groups_train))
        n_splits = min(args.folds, len(unique_groups))
        if n_splits < 2:
            raise SystemExit("Not enough subject groups for grouped cross-validation.")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_train, y_train, groups_train)
        cv_scheme = f"grouped_by_subject_{n_splits}fold"
    else:
        n_splits = min(args.folds, positives, negatives)
        if n_splits < 2:
            raise SystemExit("Not enough class balance for cross-validation.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X_train, y_train)
        cv_scheme = f"stratified_{n_splits}fold"

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=args.seed,
            solver="liblinear",
        )),
    ])

    proba_rag = [0.0] * len(keys)
    for train_pos, test_pos in split_iter:
        train_idx = [train_indices[i] for i in train_pos]
        test_idx = [train_indices[i] for i in test_pos]
        X_fit = [all_features[i] for i in train_idx]
        y_fit = [all_labels[i] for i in train_idx]
        X_eval = [all_features[i] for i in test_idx]
        model.fit(X_fit, y_fit)
        probs = model.predict_proba(X_eval)[:, 1]
        for idx, prob in zip(test_idx, probs):
            proba_rag[idx] = float(prob)

    selected_correct = 0
    choice_counts = {"baseline": 0, "rag": 0, "same_answer": 0}
    improved_vs_baseline = 0
    regressed_vs_baseline = 0
    changed_vs_baseline = 0

    disagreement_probs = []
    disagreement_y = []
    examples = []

    for i, k in enumerate(keys):
        b = int(base[k]["correct"])
        r = int(rag[k]["correct"])
        if not disagreement_mask[i]:
            chosen = "same_answer"
            sel = b
        else:
            disagreement_probs.append(proba_rag[i])
            disagreement_y.append(all_labels[i])
            if proba_rag[i] > 0.5:
                chosen = "rag"
                sel = r
            else:
                chosen = "baseline"
                sel = b

        choice_counts[chosen] += 1
        selected_correct += sel
        if sel != b:
            changed_vs_baseline += 1
            if sel > b:
                improved_vs_baseline += 1
            else:
                regressed_vs_baseline += 1

        if disagreement_mask[i] and len(examples) < 20:
            examples.append({
                "row_key": k,
                "question": base[k].get("question", ""),
                "baseline_correct": b,
                "rag_correct": r,
                "proba_choose_rag": proba_rag[i],
                "chosen": chosen,
                "baseline_margin": base[k].get("score_margin"),
                "rag_margin": rag[k].get("score_margin"),
                "baseline_entropy": base[k].get("score_entropy"),
                "rag_entropy": rag[k].get("score_entropy"),
            })

    base_correct = sum(int(base[k]["correct"]) for k in keys)
    rag_correct = sum(int(rag[k]["correct"]) for k in keys)
    oracle_correct = sum(max(int(base[k]["correct"]), int(rag[k]["correct"])) for k in keys)

    out = {
        "n": len(keys),
        "train_on": "disagreement_only",
        "cv_scheme": cv_scheme,
        "disagreement_n": len(train_indices),
        "baseline_acc": base_correct / len(keys),
        "rag_acc": rag_correct / len(keys),
        "oracle_acc": oracle_correct / len(keys),
        "selected_acc": selected_correct / len(keys),
        "selected_delta_vs_baseline": (selected_correct - base_correct) / len(keys),
        "selected_delta_vs_rag": (selected_correct - rag_correct) / len(keys),
        "unused_headroom_after_selection": (oracle_correct - selected_correct) / len(keys),
        "choice_counts": choice_counts,
        "changed_vs_baseline": changed_vs_baseline,
        "improved_vs_baseline": improved_vs_baseline,
        "regressed_vs_baseline": regressed_vs_baseline,
        "disagreement_auc": (
            None if len(set(disagreement_y)) < 2 else roc_auc_score(disagreement_y, disagreement_probs)
        ),
        "feature_names": feature_names,
        "sample_examples": examples,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
