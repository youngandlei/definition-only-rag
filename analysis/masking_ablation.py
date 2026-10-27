#!/usr/bin/env python3
"""Leakage masking ablation.

Operates by string surgery on the ORIGINAL prompts stored in the trace, so the
template stays byte-identical to the main runs; only the Definitions block changes.
Scoring replicates rag_benchmarks._score_mc_logprob_vllm exactly (echo=True,
max_tokens=0, sum of continuation token logprobs over the full choice text).

Conditions:
  vanilla    — original prompt untouched (sanity check; should reproduce 64.06)
  mask_term  — inside the Definitions block, every snippet's term name (and its
               occurrences in the snippet text) replaced with "[TERM]"
  drop_match — snippets whose text contains any full answer-option string (len>=4)
               are removed; if all are removed the Definitions block is dropped

Usage (vLLM serving Qwen-2.5-7B-Instruct):
  python3 analysis/masking_ablation.py \
    --trace outputs/traces/main/2026-04-09_qwen7_mmlu/vllm_qwen_qwen2_5_7b_instruct__mmlu__hybrid_rerank__defs_last.jsonl \
    --base_url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
    --output outputs/analysis/masking_ablation_qwen7_mmlu.json
"""
import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEF_RE = re.compile(r"\nDefinitions:\n(.*?)\n\nAnswer:", re.S)


def get_def_block(prompt):
    m = DEF_RE.search(prompt)
    return m.group(1) if m else None


def set_def_block(prompt, new_block):
    if new_block is None:  # drop the whole block
        return DEF_RE.sub("\n\nAnswer:", prompt)
    return DEF_RE.sub("\nDefinitions:\n" + new_block.replace("\\", "\\\\") + "\n\nAnswer:", prompt)


def parse_items(block):
    """Definitions block lines look like '- Term: text...' (text may wrap? stored single-line per doc)."""
    items = []
    for line in block.split("\n- "):
        line = line.lstrip("- ").strip()
        if not line:
            continue
        term, sep, text = line.partition(": ")
        if not sep:
            term, text = "", line
        items.append((term.strip(), text.strip()))
    return items


def render_items(items):
    return "\n- ".join(f"{t}: {x}" if t else x for t, x in items) if items else None


def cond_mask_term(prompt, rec):
    block = get_def_block(prompt)
    if block is None:
        return prompt
    items = parse_items(block)
    out = []
    for term, text in items:
        if term:
            text = re.sub(re.escape(term), "[TERM]", text, flags=re.IGNORECASE)
        out.append(("[TERM]", text))
    rendered = render_items(out)
    return set_def_block(prompt, "- " + rendered if rendered else None)


def cond_drop_match(prompt, rec):
    block = get_def_block(prompt)
    if block is None:
        return prompt
    items = parse_items(block)
    lowers = [c.lower().strip() for c in rec["choices"] if c and len(c.strip()) >= 4]
    keep = [(t, x) for t, x in items if not any(c in (t + " " + x).lower() for c in lowers)]
    if not keep:
        return set_def_block(prompt, None)
    return set_def_block(prompt, "- " + render_items(keep))


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def score_mc(base_url, model, prompt, choices):
    """Byte-for-byte port of rag_benchmarks._score_mc_logprob_vllm."""
    url = base_url.rstrip("/") + "/completions"
    scores = []
    for choice in choices:
        full = prompt + " " + choice
        data = post(url, {"model": model, "prompt": full, "max_tokens": 0,
                          "temperature": 0, "echo": True, "logprobs": 1})
        ch = (data.get("choices") or [{}])[0]
        lp = ch.get("logprobs") or {}
        token_lps = lp.get("token_logprobs") or []
        offsets = lp.get("text_offset") or []
        if not token_lps or len(token_lps) != len(offsets):
            raise RuntimeError("no usable logprobs from vLLM")
        start = len(prompt)
        total = sum(float(t) for off, t in zip(offsets, token_lps)
                    if off >= start and t is not None)
        scores.append(total)
    return max(range(len(scores)), key=lambda i: scores[i]), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default="vanilla,mask_term,drop_match")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.trace)]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} questions loaded", flush=True)

    builders = {"vanilla": lambda p, r: p, "mask_term": cond_mask_term, "drop_match": cond_drop_match}
    summary = {}
    for name in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        build = builders[name]

        def run(rec):
            prompt = build(rec["prompt"], rec)
            try:
                pred, _ = score_mc(args.base_url, args.model, prompt, rec["choices"])
            except Exception as exc:
                print("ERR", rec.get("row_key"), exc, file=sys.stderr, flush=True)
                return None
            nd = 0 if get_def_block(prompt) is None else len(parse_items(get_def_block(prompt)))
            return {"row_key": rec.get("row_key"), "correct": pred == int(rec["gold"]), "n_defs": nd}

        with ThreadPoolExecutor(args.workers) as ex:
            results = [r for r in ex.map(run, recs) if r]
        acc = sum(r["correct"] for r in results) / len(results) * 100
        zero = sum(1 for r in results if r["n_defs"] == 0)
        summary[name] = {"accuracy": round(acc, 2), "n": len(results), "zero_def_prompts": zero}
        print(f"{name}: acc={acc:.2f}  n={len(results)}  zero-def prompts={zero}", flush=True)

    json.dump(summary, open(args.output, "w"), indent=1)
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
