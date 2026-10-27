#!/usr/bin/env python3
"""
Filter candidate Investopedia terms per MCQ via GPT-5.2 (or any OpenAI model) to
produce gold annotation matching XFinBench's `gold_fin_term_id` schema.

Input:  JSON from retrieve_candidates.py (per-question top-K candidates)
Output:
  - CSV in XFinBench-compatible format
  - JSON with raw model responses (audit trail)

Usage:
  export OPENAI_API_KEY=sk-xxx
  python3 analysis/annotate_mmlu_terms.py \\
    --candidates outputs/analysis/annotation/mmlu_candidates_top10.json \\
    --model gpt-5.2 \\
    --workers 8 \\
    --output_csv outputs/analysis/annotation/mmlu_finance_annotated.csv \\
    --output_json outputs/analysis/annotation/mmlu_finance_annotation_audit.json
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

PROMPT = """You are an expert in financial terminology. Given a multiple-choice question and a list of candidate financial terms with definitions, identify which terms must be understood in order to correctly answer the question.

A term is REQUIRED if its definition is essential — not merely topically related — to choosing the correct answer.

QUESTION:
{question}

CHOICES:
{choices}

CANDIDATE TERMS (id, term, definition):
{candidates}

Rules:
- Return ONLY a JSON object: {{"required_term_ids": ["doc_id_a", "doc_id_b", ...]}}
- It is acceptable to return an empty list if the question does not depend on any specific term (e.g., pure arithmetic, general knowledge).
- It is acceptable to return multiple term IDs if the question requires understanding several concepts.
- Use ONLY IDs that appear in the candidate list above. Do not invent IDs."""


def call_openai(prompt, model, api_key, max_tokens=200):
    if model.startswith("gpt-5"):
        url = "https://api.openai.com/v1/responses"
        payload = json.dumps({
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }).encode()
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            if "output_text" in data:
                return data["output_text"].strip()
            out = []
            for item in data.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        out.append(c["text"])
            return "\n".join(out).strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def parse_response(raw, valid_ids):
    """Extract required_term_ids from raw model output, intersected with valid candidate set."""
    if not raw:
        return []
    # First try strict JSON parse
    try:
        d = json.loads(raw)
        ids = d.get("required_term_ids", [])
        return [str(x) for x in ids if str(x) in valid_ids]
    except json.JSONDecodeError:
        pass
    # Try to find JSON block inside the response
    m = re.search(r'\{[^{}]*"required_term_ids"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return [str(x) for x in d.get("required_term_ids", []) if str(x) in valid_ids]
        except json.JSONDecodeError:
            pass
    # Last resort: scan for quoted doc_ids that match valid candidates
    found = []
    for tid in re.findall(r'"([^"]+)"', raw):
        if tid in valid_ids and tid not in found:
            found.append(tid)
    return found


def format_choices(choices):
    if isinstance(choices, list):
        return "\n".join(f"{chr(65+i)}. {ch}" for i, ch in enumerate(choices))
    return str(choices)


def format_candidates(candidates, max_def_chars=300):
    lines = []
    for c in candidates:
        defn = c["definition"][:max_def_chars]
        if len(c["definition"]) > max_def_chars:
            defn += "..."
        lines.append(f"  {c['doc_id']} | {c['term']} | {defn}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_def_chars", type=int, default=300)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY missing (env var or --api_key)")

    items = json.load(open(args.candidates))
    print(f"Annotating {len(items)} questions with {args.model} (workers={args.workers})...")

    def annotate(item):
        valid_ids = {c["doc_id"] for c in item["candidates"]}
        prompt = PROMPT.format(
            question=item["question"],
            choices=format_choices(item["choices"]),
            candidates=format_candidates(item["candidates"], args.max_def_chars),
        )
        try:
            raw = call_openai(prompt, args.model, args.api_key)
            term_ids = parse_response(raw, valid_ids)
            return {**item, "raw_response": raw, "gold_inv_term_ids": term_ids, "error": None}
        except Exception as e:
            return {**item, "raw_response": "", "gold_inv_term_ids": [], "error": str(e)}

    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {pool.submit(annotate, item): i for i, item in enumerate(items)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"  {done}/{len(items)} done")

    # Save audit JSON
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save XFinBench-style CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "benchmark", "subject", "task",
                    "question", "choices", "gold_answer", "gold_inv_term_id"])
        for r in results:
            qid = r.get("row_key", "")
            ch_list = r.get("choices", [])
            choices_str = json.dumps(ch_list, ensure_ascii=False) if isinstance(ch_list, list) else str(ch_list)
            term_ids_str = ";".join(r.get("gold_inv_term_ids", []))
            w.writerow([
                qid,
                r.get("benchmark", ""),
                r.get("subject", ""),
                r.get("type", "mcq"),
                r["question"],
                choices_str,
                r.get("gold", ""),
                term_ids_str,
            ])

    # Summary
    n_terms = [len(r["gold_inv_term_ids"]) for r in results if not r.get("error")]
    n_err = sum(1 for r in results if r.get("error"))
    dist = Counter(n_terms)
    print(f"\n=== Annotation summary ===")
    print(f"Total questions:        {len(results)}")
    print(f"Errors:                 {n_err}")
    print(f"Avg terms per question: {sum(n_terms)/len(n_terms):.2f}" if n_terms else "—")
    print(f"Term-count distribution:")
    for k in sorted(dist):
        print(f"   {k} term(s): {dist[k]:>5} questions")
    print(f"\nSaved CSV:  {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
