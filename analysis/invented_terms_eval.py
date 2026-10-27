#!/usr/bin/env python3
"""Invented-terms control experiment.

50 synthetic financial terms with author-written definitions, each containing one
arbitrary discrete fact. Prompt template is byte-compatible with the paper's
defs_last format. Two conditions per model:

  baseline — question + options only (term unseen in any training data by construction)
  withdef  — question + options + Definitions: block containing the custom definition

Scoring matches the main runs per provider:
  vLLM (Qwen-7B): echo logprob over full choice text (port of _score_mc_logprob_vllm)
  OpenAI (GPT-5.2): generate mode, "reply with only the letter", Responses API, T=0
    (the same fallback path OpenAIModel.score_mc used in the main experiments)

Usage:
  python3 analysis/invented_terms_eval.py --model openai:gpt-5.2 \
      --output outputs/analysis/invented_terms_gpt52.json
  python3 analysis/invented_terms_eval.py --model vllm:Qwen/Qwen2.5-7B-Instruct \
      --base_url http://127.0.0.1:8000/v1 \
      --output outputs/analysis/invented_terms_qwen7.json
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LETTERS = ["A", "B", "C", "D"]


def build_prompt(rec, with_def):
    lines = [f"Question: {rec['question']}", "", "Options:"]
    for i, c in enumerate(rec["choices"]):
        lines.append(f"{LETTERS[i]}) {c}")
    if with_def:
        lines += ["", "Definitions:", f"- {rec['term']}: {rec['definition']}"]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def score_vllm(base_url, model, prompt, choices):
    url = base_url.rstrip("/") + "/completions"
    scores = []
    for choice in choices:
        full = prompt + " " + choice
        data = post(url, {"model": model, "prompt": full, "max_tokens": 0,
                          "temperature": 0, "echo": True, "logprobs": 1},
                    {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        lp = (data.get("choices") or [{}])[0].get("logprobs") or {}
        tls, offs = lp.get("token_logprobs") or [], lp.get("text_offset") or []
        start = len(prompt)
        scores.append(sum(float(t) for o, t in zip(offs, tls) if o >= start and t is not None))
    return max(range(len(scores)), key=lambda i: scores[i])


def extract_letter(text, n):
    m = re.search(r"\b([A-D])\b", text.upper())
    if m:
        idx = LETTERS.index(m.group(1))
        if idx < n:
            return idx
    return None


def score_openai(model, prompt, choices, api_key):
    ask = prompt + "\n\nChoose the single best option and reply with only the letter."
    data = post("https://api.openai.com/v1/responses",
                {"model": model, "input": ask, "temperature": 0, "max_output_tokens": 16},
                {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                text += c.get("text", "")
    idx = extract_letter(text, len(choices))
    return (idx if idx is not None else 0), text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/invented_terms/invented_terms_v1.json")
    ap.add_argument("--model", required=True, help="openai:gpt-5.2 or vllm:<hf-id>")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    provider, model = args.model.split(":", 1)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if provider == "openai" and not api_key:
        sys.exit("OPENAI_API_KEY not set")

    recs = json.load(open(args.data))
    out = {"model": args.model, "n": len(recs), "conditions": {}}
    for cond in ["baseline", "withdef"]:
        def run(rec):
            prompt = build_prompt(rec, cond == "withdef")
            if provider == "vllm":
                pred = score_vllm(args.base_url, model, prompt, rec["choices"])
                raw = ""
            else:
                pred, raw = score_openai(model, prompt, rec["choices"], api_key)
            return {"id": rec["id"], "pred": pred, "correct": pred == rec["answer"], "raw": raw}

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(run, recs))
        acc = sum(r["correct"] for r in results) / len(results) * 100
        out["conditions"][cond] = {"accuracy": round(acc, 1), "results": results}
        print(f"{args.model} {cond}: {acc:.1f}", flush=True)

    json.dump(out, open(args.output, "w"), indent=1)
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
