#!/usr/bin/env python3
"""
LLM-as-Judge selection: for each question where baseline and RAG disagree,
ask a judge LLM to pick the better answer. When they agree, use the consensus.
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# API call (minimal, no external deps beyond stdlib)
# ---------------------------------------------------------------------------

def call_openai(prompt: str, model: str, api_key: str, base_url: str = None, max_tokens: int = 16) -> str:
    # GPT-5 family uses the Responses API
    if model.startswith("gpt-5"):
        url = base_url or "https://api.openai.com/v1/responses"
        payload = json.dumps({
            "model": model,
            "input": prompt,
            "max_output_tokens": max(16, max_tokens),
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                if "output_text" in data and data["output_text"]:
                    return data["output_text"].strip()
                out = []
                for item in data.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") == "output_text" and c.get("text"):
                            out.append(c["text"])
                return "\n".join(out).strip()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    # Standard chat completions for gpt-4o etc.
    url = base_url or "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def build_judge_prompt(question: str, choices: list, ans1: str, ans2: str,
                       definitions: list = None, with_defs: bool = True) -> str:
    choices_text = "\n".join(f"{CHOICE_LABELS[i]}) {c}" for i, c in enumerate(choices))

    defs_block = ""
    if with_defs and definitions:
        defs_lines = [f"- {d['term']}: {d['text']}" for d in definitions if isinstance(d, dict) and d.get("term")]
        if defs_lines:
            defs_block = "\n\nRelevant financial definitions:\n" + "\n".join(defs_lines[:3])

    prompt = f"""You are evaluating two candidate answers to a multiple-choice question. Pick the one more likely to be correct.

Question: {question}

Options:
{choices_text}{defs_block}

Candidate Answer 1: {ans1}
Candidate Answer 2: {ans2}

Which candidate is more likely correct? Reply with exactly "1" or "2"."""
    return prompt


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["row_key"]] = r
    return rows


def idx_to_label(idx: int) -> str:
    return CHOICE_LABELS[idx] if 0 <= idx < len(CHOICE_LABELS) else str(idx)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_judge(
    baseline_trace: str,
    rag_trace: str,
    judge_model: str,
    api_key: str,
    base_url: str,
    output: str,
    with_defs: bool,
    workers: int,
    seed: int,
    dry_run: bool,
):
    rng = random.Random(seed)

    base = load_trace(baseline_trace)
    rag = load_trace(rag_trace)
    keys = sorted(set(base) & set(rag))
    print(f"Overlapping questions: {len(keys)}")

    acc_base = sum(base[k]["correct"] for k in keys) / len(keys)
    acc_rag  = sum(rag[k]["correct"]  for k in keys) / len(keys)
    oracle   = sum(max(base[k]["correct"], rag[k]["correct"]) for k in keys) / len(keys)
    print(f"Baseline acc:  {acc_base:.4f}")
    print(f"RAG acc:       {acc_rag:.4f}")
    print(f"Oracle acc:    {oracle:.4f}")

    agree_keys    = [k for k in keys if base[k]["prediction"] == rag[k]["prediction"]]
    disagree_keys = [k for k in keys if base[k]["prediction"] != rag[k]["prediction"]]
    print(f"Agree:    {len(agree_keys)}  ({len(agree_keys)/len(keys):.1%})")
    print(f"Disagree: {len(disagree_keys)}  ({len(disagree_keys)/len(keys):.1%})")

    # Among disagreements, what's the oracle?
    d_base_wins  = sum(1 for k in disagree_keys if base[k]["correct"] and not rag[k]["correct"])
    d_rag_wins   = sum(1 for k in disagree_keys if rag[k]["correct"]  and not base[k]["correct"])
    d_both_wrong = sum(1 for k in disagree_keys if not base[k]["correct"] and not rag[k]["correct"])
    print(f"  Disagree breakdown: baseline_wins={d_base_wins}, rag_wins={d_rag_wins}, both_wrong={d_both_wrong}")
    print(f"  Max gain from perfect judging: +{(d_base_wins+d_rag_wins)/len(keys):.4f}")

    if dry_run:
        print("\n[dry-run] Stopping before API calls.")
        return

    # Run judge on disagreement cases
    results = {}  # row_key -> {"chosen": "base"/"rag", "judge_raw": str, "flipped": bool}

    def judge_one(k):
        b = base[k]
        r = rag[k]
        choices = b["choices"]
        b_label = idx_to_label(b["prediction"])
        r_label = idx_to_label(r["prediction"])

        # Randomly assign to position 1 or 2 to reduce position bias
        flip = rng.random() < 0.5
        if flip:
            ans1, ans2 = r_label, b_label
        else:
            ans1, ans2 = b_label, r_label

        defs = r.get("retrieved_definitions") or []
        prompt = build_judge_prompt(b["question"], choices, ans1, ans2,
                                    definitions=defs, with_defs=with_defs)

        raw = call_openai(prompt, judge_model, api_key, base_url)

        # Parse "1" or "2"
        choice_num = None
        for tok in raw.strip().split():
            if tok.strip(".,!\"'") in ("1", "2"):
                choice_num = tok.strip(".,!\"'")
                break
        if choice_num is None:
            # Fallback: look for 1 or 2 anywhere
            if "1" in raw and "2" not in raw:
                choice_num = "1"
            elif "2" in raw and "1" not in raw:
                choice_num = "2"
            else:
                choice_num = "1"  # default

        # Map back to base/rag
        if flip:
            chosen = "rag" if choice_num == "1" else "base"
        else:
            chosen = "base" if choice_num == "1" else "rag"

        return k, {"chosen": chosen, "judge_raw": raw, "flipped": flip,
                   "ans1": ans1, "ans2": ans2, "choice_num": choice_num}

    print(f"\nCalling judge ({judge_model}) on {len(disagree_keys)} disagreement cases ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, k): k for k in disagree_keys}
        for i, fut in enumerate(as_completed(futures), 1):
            k, res = fut.result()
            results[k] = res
            if i % 20 == 0 or i == len(disagree_keys):
                print(f"  {i}/{len(disagree_keys)} done")

    # Compute final accuracy
    correct_judge = 0
    judge_chose_base = 0
    judge_chose_rag  = 0
    for k in keys:
        if k in results:
            chosen_side = results[k]["chosen"]
            chosen_rec  = base[k] if chosen_side == "base" else rag[k]
            if chosen_side == "base":
                judge_chose_base += 1
            else:
                judge_chose_rag += 1
        else:
            # Agreement case: both give same answer, use either
            chosen_rec = base[k]
        correct_judge += chosen_rec["correct"]

    acc_judge = correct_judge / len(keys)

    # Among disagreements only
    correct_judge_disagree = sum(
        (base[k] if results[k]["chosen"] == "base" else rag[k])["correct"]
        for k in disagree_keys
    )
    acc_judge_disagree = correct_judge_disagree / len(disagree_keys)

    print(f"\n=== Results ===")
    print(f"Baseline acc:        {acc_base:.4f}")
    print(f"RAG acc:             {acc_rag:.4f}")
    print(f"Oracle acc:          {oracle:.4f}")
    print(f"Judge acc (overall): {acc_judge:.4f}  (delta vs RAG: {acc_judge-acc_rag:+.4f})")
    print(f"Judge acc on disagree subset: {acc_judge_disagree:.4f}  ({correct_judge_disagree}/{len(disagree_keys)})")
    print(f"Judge chose base: {judge_chose_base}, chose RAG: {judge_chose_rag}")

    summary = {
        "baseline_acc":  acc_base,
        "rag_acc":        acc_rag,
        "oracle_acc":     oracle,
        "judge_acc":      acc_judge,
        "judge_delta_vs_rag": acc_judge - acc_rag,
        "judge_acc_disagree": acc_judge_disagree,
        "n_total":        len(keys),
        "n_agree":        len(agree_keys),
        "n_disagree":     len(disagree_keys),
        "judge_chose_base": judge_chose_base,
        "judge_chose_rag":  judge_chose_rag,
        "baseline_trace": baseline_trace,
        "rag_trace":      rag_trace,
        "judge_model":    judge_model,
        "with_defs":      with_defs,
        "decisions":      results,
    }

    if output:
        Path(output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved to {output}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge selector for RAG vs baseline")
    ap.add_argument("--baseline_trace", required=True, help="JSONL trace for baseline (no RAG)")
    ap.add_argument("--rag_trace",      required=True, help="JSONL trace for RAG run")
    ap.add_argument("--judge_model",    default="gpt-4o-mini", help="OpenAI model to use as judge")
    ap.add_argument("--api_key",        default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base_url",       default="", help="Custom OpenAI-compatible base URL")
    ap.add_argument("--output",         default="", help="Path to save JSON results")
    ap.add_argument("--with_defs",      action="store_true", default=True,
                    help="Include retrieved definitions in judge prompt")
    ap.add_argument("--no_defs",        action="store_true", help="Do NOT include definitions")
    ap.add_argument("--workers",        type=int, default=8, help="Parallel API call threads")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--dry_run",        action="store_true", help="Print stats but skip API calls")
    args = ap.parse_args()

    if args.no_defs:
        args.with_defs = False

    if not args.api_key and not args.dry_run:
        raise SystemExit("No API key found. Set OPENAI_API_KEY or pass --api_key.")

    run_judge(
        baseline_trace=args.baseline_trace,
        rag_trace=args.rag_trace,
        judge_model=args.judge_model,
        api_key=args.api_key,
        base_url=args.base_url or None,
        output=args.output,
        with_defs=args.with_defs,
        workers=args.workers,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
