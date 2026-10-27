#!/usr/bin/env python3
"""
Semantic entropy selection for MC questions.

For each disagreement case (baseline vs RAG predict different answers):
  - Sample K answers from baseline condition (temperature > 0)
  - Sample K answers from RAG condition (temperature > 0)
  - Compute entropy over vote distribution for each
  - Select the condition with lower entropy (more confident)

For agreement cases: use the consensus answer directly.
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

CHOICE_LABELS = ["A", "B", "C", "D", "E"]
CHOICE_RE = re.compile(r'\b([A-Ea-e])\b')


# ---------------------------------------------------------------------------
# vLLM API call
# ---------------------------------------------------------------------------

def call_vllm(prompt: str, model: str, host: str, port: int,
              api_key: str, temperature: float, max_tokens: int = 64) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def parse_choice(text: str) -> str | None:
    """Extract A/B/C/D from model response."""
    text = text.strip()
    # First token is often the answer
    first = text.split()[0].strip(".,)(:") if text else ""
    if first.upper() in CHOICE_LABELS:
        return first.upper()
    # Search anywhere
    matches = CHOICE_RE.findall(text)
    if matches:
        return matches[0].upper()
    return None


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def vote_entropy(votes: list[str]) -> float:
    """Entropy of a vote distribution. Lower = more confident."""
    if not votes:
        return float('inf')
    counts = Counter(votes)
    n = len(votes)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def plurality_vote(votes: list[str]) -> str | None:
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Load traces
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


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def sample_condition(prompt: str, model: str, host: str, port: int,
                     api_key: str, temperature: float, k: int,
                     max_tokens: int = 64) -> list[str]:
    """Sample K answers for one prompt, return list of parsed choices."""
    answers = []
    for _ in range(k):
        raw = call_vllm(prompt, model, host, port, api_key, temperature, max_tokens)
        choice = parse_choice(raw)
        if choice:
            answers.append(choice)
    return answers


def run(args):
    base_trace = load_trace(args.baseline_trace)
    rag_trace  = load_trace(args.rag_trace)

    keys = sorted(set(base_trace) & set(rag_trace))
    agree_keys    = [k for k in keys if base_trace[k]["prediction"] == rag_trace[k]["prediction"]]
    disagree_keys = [k for k in keys if base_trace[k]["prediction"] != rag_trace[k]["prediction"]]

    print(f"Total: {len(keys)}, agree: {len(agree_keys)}, disagree: {len(disagree_keys)}")

    acc_base   = sum(base_trace[k]["correct"] for k in keys) / len(keys)
    acc_rag    = sum(rag_trace[k]["correct"]  for k in keys) / len(keys)
    oracle     = sum(max(base_trace[k]["correct"], rag_trace[k]["correct"]) for k in keys) / len(keys)
    print(f"Baseline: {acc_base:.4f} | RAG: {acc_rag:.4f} | Oracle: {oracle:.4f}")

    d_bw = sum(1 for k in disagree_keys if base_trace[k]["correct"] and not rag_trace[k]["correct"])
    d_rw = sum(1 for k in disagree_keys if rag_trace[k]["correct"]  and not base_trace[k]["correct"])
    d_xx = sum(1 for k in disagree_keys if not base_trace[k]["correct"] and not rag_trace[k]["correct"])
    print(f"Disagree: base_wins={d_bw}, rag_wins={d_rw}, both_wrong={d_xx}")

    if args.dry_run:
        print("[dry-run] done.")
        return

    # Sample disagreement cases in parallel
    results = {}  # row_key -> dict

    def process_one(k):
        b = base_trace[k]
        r = rag_trace[k]

        b_votes = sample_condition(b["prompt"], args.model, args.host, args.port,
                                   args.api_key, args.temperature, args.k, args.max_tokens)
        r_votes = sample_condition(r["prompt"], args.model, args.host, args.port,
                                   args.api_key, args.temperature, args.k, args.max_tokens)

        b_entropy = vote_entropy(b_votes)
        r_entropy = vote_entropy(r_votes)
        b_plurality = plurality_vote(b_votes)
        r_plurality = plurality_vote(r_votes)

        # Selection: lower entropy wins
        chosen = "base" if b_entropy <= r_entropy else "rag"

        return k, {
            "chosen": chosen,
            "base_entropy": b_entropy,
            "rag_entropy": r_entropy,
            "base_votes": b_votes,
            "rag_votes": r_votes,
            "base_plurality": b_plurality,
            "rag_plurality": r_plurality,
        }

    print(f"\nSampling {args.k} times per condition on {len(disagree_keys)} disagree cases "
          f"(temperature={args.temperature}, workers={args.workers}) ...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, k): k for k in disagree_keys}
        for i, fut in enumerate(as_completed(futures), 1):
            k, res = fut.result()
            results[k] = res
            if i % 20 == 0 or i == len(disagree_keys):
                print(f"  {i}/{len(disagree_keys)} done")

    # Compute accuracy
    def eval_method(select_fn):
        """select_fn(k) -> 'base' or 'rag'"""
        correct = 0
        for k in keys:
            if k in results:
                chosen = select_fn(k)
            else:
                chosen = "base"  # agree case, both same
            rec = base_trace[k] if chosen == "base" else rag_trace[k]
            correct += rec["correct"]
        return correct / len(keys)

    # Method 1: entropy selection
    acc_entropy = eval_method(lambda k: results[k]["chosen"])

    # Method 2: plurality vote selection (pick whichever vote majority is more confident)
    def plurality_select(k):
        res = results[k]
        b_cnt = Counter(res["base_votes"]).most_common(1)[0][1] if res["base_votes"] else 0
        r_cnt = Counter(res["rag_votes"]).most_common(1)[0][1] if res["rag_votes"] else 0
        return "base" if b_cnt >= r_cnt else "rag"

    acc_plurality = eval_method(plurality_select)

    # Method 3: always rag / always base (sanity check)
    acc_always_rag  = eval_method(lambda k: "rag")
    acc_always_base = eval_method(lambda k: "base")

    print(f"\n=== Results ===")
    print(f"Baseline (original greedy):  {acc_base:.4f}")
    print(f"RAG (original greedy):       {acc_rag:.4f}")
    print(f"Always RAG on disagree:      {acc_always_rag:.4f}")
    print(f"Always Base on disagree:     {acc_always_base:.4f}")
    print(f"Entropy selection:           {acc_entropy:.4f}  (delta vs RAG: {acc_entropy - acc_rag:+.4f})")
    print(f"Plurality vote selection:    {acc_plurality:.4f}  (delta vs RAG: {acc_plurality - acc_rag:+.4f})")
    print(f"Oracle:                      {oracle:.4f}")

    # Per-decision stats
    chose_base = sum(1 for k in disagree_keys if results[k]["chosen"] == "base")
    chose_rag  = sum(1 for k in disagree_keys if results[k]["chosen"] == "rag")
    print(f"\nChose base: {chose_base}, chose RAG: {chose_rag}")

    finite_b = [results[k]["base_entropy"] for k in disagree_keys if results[k]["base_entropy"] != float('inf')]
    finite_r = [results[k]["rag_entropy"]  for k in disagree_keys if results[k]["rag_entropy"] != float('inf')]
    avg_b_ent = sum(finite_b) / len(finite_b) if finite_b else float('inf')
    avg_r_ent = sum(finite_r) / len(finite_r) if finite_r else float('inf')
    b_empty = len(disagree_keys) - len(finite_b)
    r_empty = len(disagree_keys) - len(finite_r)
    print(f"Avg base entropy: {avg_b_ent:.4f} (inf/empty: {b_empty}/{len(disagree_keys)}) | "
          f"Avg RAG entropy: {avg_r_ent:.4f} (inf/empty: {r_empty}/{len(disagree_keys)})")

    summary = {
        "baseline_acc": acc_base,
        "rag_acc": acc_rag,
        "oracle_acc": oracle,
        "entropy_selection_acc": acc_entropy,
        "plurality_selection_acc": acc_plurality,
        "delta_entropy_vs_rag": acc_entropy - acc_rag,
        "n_total": len(keys),
        "n_disagree": len(disagree_keys),
        "chose_base": chose_base,
        "chose_rag": chose_rag,
        "k_samples": args.k,
        "temperature": args.temperature,
        "decisions": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_trace", required=True)
    ap.add_argument("--rag_trace",      required=True)
    ap.add_argument("--model",    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--host",     default="127.0.0.1")
    ap.add_argument("--port",     type=int, default=8000)
    ap.add_argument("--api_key",  default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--k",        type=int, default=5,   help="Samples per condition per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--output",   default="")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--dry_run",  action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
