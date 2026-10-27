#!/usr/bin/env python3
"""
Financial Terminology Knowledge Evaluation.

For each Investopedia term, ask a model to define it, then score the
response against the ground-truth definition. This produces:
  1. A per-term knowledge score for each model
  2. A "blind spot" list — terms the model doesn't know well
  3. Cross-model comparison of financial terminology coverage

Supports both vLLM (local) and OpenAI API models.

Usage:
  # Local vLLM model
  python term_knowledge_eval.py \
    --investopedia_path InvestopediaDatasetFinal.json \
    --model vllm:Qwen/Qwen2.5-7B-Instruct \
    --vllm_base_url http://127.0.0.1:8000/v1 \
    --output outputs/analysis/term_eval/qwen7b.json

  # OpenAI model
  python term_knowledge_eval.py \
    --investopedia_path InvestopediaDatasetFinal.json \
    --model openai:gpt-4o-mini \
    --api_key $OPENAI_API_KEY \
    --output outputs/analysis/term_eval/gpt4o_mini.json

  # Score only (if generation already done)
  python term_knowledge_eval.py \
    --score_only \
    --generated outputs/analysis/term_eval/qwen7b.json \
    --scorer openai:gpt-4o-mini \
    --api_key $OPENAI_API_KEY
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request


# ---------------------------------------------------------------------------
# Investopedia loader
# ---------------------------------------------------------------------------

def load_investopedia(path: str, min_def_len: int = 30) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    types = data.get("Type", {})
    terms = data.get("Term Name", {})
    ktaws = data.get("KTAW", {})

    all_terms = []
    for key, typ in types.items():
        if typ != "terms":
            continue
        term = str(terms.get(key, "")).strip()
        if not term:
            continue
        ktaw_list = ktaws.get(key, []) or []
        if isinstance(ktaw_list, list):
            definition = " ".join(s.strip() for s in ktaw_list if s and s.strip())
        else:
            definition = str(ktaw_list).strip()
        if not definition or len(definition) < min_def_len:
            continue
        all_terms.append({
            "doc_id": str(key),
            "term": term,
            "definition": definition,
        })
    return all_terms


# ---------------------------------------------------------------------------
# XFinBench knowledge_bank loader
# ---------------------------------------------------------------------------

def load_xfinbench_kb(path: str, min_def_len: int = 30) -> list:
    """Load XFinBench knowledge_bank.csv as a list of {doc_id, term, definition}.

    Each row has columns id, term_name, term_definition. term_name is sometimes
    comma-separated ("term, context_topic_1, context_topic_2"); we take the
    first comma piece as the canonical term, consistent with rag_benchmarks.py
    read_xfinbench_kb().
    """
    import csv as _csv
    all_terms = []
    with open(path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            term_name = (row.get("term_name") or "").strip()
            definition = (row.get("term_definition") or "").strip()
            doc_id = (row.get("id") or "").strip()
            if not term_name or not doc_id:
                continue
            primary_term = term_name.split(",")[0].strip()
            if not primary_term:
                continue
            if not definition or len(definition) < min_def_len:
                continue
            all_terms.append({
                "doc_id": doc_id,
                "term": primary_term,
                "definition": definition,
            })
    return all_terms


def load_caregiver(path: str, min_def_len: int = 30) -> list:
    """Load Caregiver Glossary (same column-major schema as Investopedia)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    types = data.get("Type", {})
    terms = data.get("Term Name", {})
    ktaws = data.get("KTAW", {})
    out = []
    for key, typ in types.items():
        if typ != "terms":
            continue
        term = str(terms.get(key, "")).strip()
        if not term:
            continue
        ktaw_list = ktaws.get(key, []) or []
        if isinstance(ktaw_list, list):
            definition = " ".join(s.strip() for s in ktaw_list if s and s.strip())
        else:
            definition = str(ktaw_list).strip()
        if not definition or len(definition) < min_def_len:
            continue
        out.append({"doc_id": str(key), "term": term, "definition": definition})
    return out


def load_corpus(corpus_type: str, path: str, min_def_len: int = 30) -> list:
    if corpus_type == "investopedia":
        return load_investopedia(path, min_def_len=min_def_len)
    if corpus_type == "xfinbench":
        return load_xfinbench_kb(path, min_def_len=min_def_len)
    if corpus_type == "caregiver":
        return load_caregiver(path, min_def_len=min_def_len)
    raise ValueError(f"Unknown corpus_type: {corpus_type}")


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_vllm(prompt: str, base_url: str, model: str, max_tokens: int = 256,
              temperature: float = 0.0) -> str:
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def call_openai(prompt: str, model: str, api_key: str, max_tokens: int = 256,
                temperature: float = 0.0) -> str:
    if model.startswith("gpt-5"):
        url = "https://api.openai.com/v1/responses"
        payload = json.dumps({
            "model": model,
            "input": prompt,
            "max_output_tokens": max(16, max_tokens),
        }).encode()
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
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


def call_model(prompt: str, model_spec: str, **kwargs) -> str:
    """Dispatch to vllm or openai based on model_spec prefix."""
    if model_spec.startswith("vllm:"):
        model_name = model_spec[5:]
        return call_vllm(prompt, kwargs["vllm_base_url"], model_name,
                         max_tokens=kwargs.get("max_tokens", 256))
    elif model_spec.startswith("openai:"):
        model_name = model_spec[7:]
        return call_openai(prompt, model_name, kwargs["api_key"],
                           max_tokens=kwargs.get("max_tokens", 256))
    else:
        raise ValueError(f"Unknown model spec: {model_spec}. Use vllm: or openai: prefix.")


# ---------------------------------------------------------------------------
# Step 1: Generate definitions
# ---------------------------------------------------------------------------

DEFINE_PROMPT = """Define the following financial term in 2-3 sentences. Be specific and precise.

Term: {term}

Definition:"""


def generate_definitions(terms: list, model_spec: str, workers: int, **kwargs) -> list:
    results = []

    def process_one(term_info):
        prompt = DEFINE_PROMPT.format(term=term_info["term"])
        try:
            response = call_model(prompt, model_spec, max_tokens=200, **kwargs)
            return {**term_info, "model_definition": response, "error": None}
        except Exception as e:
            return {**term_info, "model_definition": "", "error": str(e)}

    print(f"Generating definitions for {len(terms)} terms with {model_spec}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, t): t for t in terms}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(terms):
                print(f"  {i}/{len(terms)} done")

    return results


# ---------------------------------------------------------------------------
# Step 2: Score definitions
# ---------------------------------------------------------------------------

SCORE_PROMPT = """You are evaluating whether a model's definition of a financial term is correct.

TERM: {term}

GROUND TRUTH DEFINITION:
{ground_truth}

MODEL'S DEFINITION:
{model_definition}

Score the model's definition on a scale of 0-3:
  0 = Completely wrong, irrelevant, or no meaningful content
  1 = Partially correct but has significant errors or missing key aspects
  2 = Mostly correct with minor inaccuracies or missing details
  3 = Fully correct, captures the essential meaning

Reply with ONLY a JSON object: {{"score": <0-3>, "reason": "<brief explanation>"}}"""


def score_definitions(results: list, scorer_spec: str, workers: int, **kwargs) -> list:
    scored = []

    def score_one(item):
        if not item.get("model_definition") or item.get("error"):
            return {**item, "score": 0, "score_reason": "no definition generated"}

        prompt = SCORE_PROMPT.format(
            term=item["term"],
            ground_truth=item["definition"][:500],
            model_definition=item["model_definition"][:500],
        )
        try:
            raw = call_model(prompt, scorer_spec, max_tokens=100, **kwargs)
            # Parse score
            try:
                parsed = json.loads(raw)
                score = int(parsed.get("score", 0))
                reason = parsed.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                # Try to extract score from text
                match = re.search(r'"score"\s*:\s*(\d)', raw)
                if match:
                    score = int(match.group(1))
                    reason = raw
                else:
                    score = -1
                    reason = f"parse error: {raw[:100]}"
            return {**item, "score": score, "score_reason": reason}
        except Exception as e:
            return {**item, "score": -1, "score_reason": str(e)}

    print(f"Scoring {len(results)} definitions with {scorer_spec}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_one, r): r for r in results}
        for i, fut in enumerate(as_completed(futures), 1):
            scored.append(fut.result())
            if i % 100 == 0 or i == len(results):
                print(f"  {i}/{len(results)} done")

    return scored


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(scored: list, model_name: str):
    valid = [s for s in scored if s["score"] >= 0]
    scores = [s["score"] for s in valid]

    if not scores:
        print("No valid scores!")
        return

    from collections import Counter
    dist = Counter(scores)

    print(f"\n{'='*60}")
    print(f"=== Terminology Knowledge Evaluation: {model_name} ===")
    print(f"{'='*60}")
    print(f"Total terms evaluated: {len(valid)}")
    print(f"Average score: {sum(scores)/len(scores):.2f} / 3.00")
    print(f"\nScore distribution:")
    for s in [0, 1, 2, 3]:
        count = dist.get(s, 0)
        pct = count / len(valid) * 100
        bar = "█" * int(pct / 2)
        print(f"  {s}: {count:>5} ({pct:>5.1f}%) {bar}")

    # Blind spots = score 0 or 1
    blind_spots = [s for s in valid if s["score"] <= 1]
    well_known = [s for s in valid if s["score"] >= 2]
    print(f"\nBlind spots (score 0-1): {len(blind_spots)} ({len(blind_spots)/len(valid)*100:.1f}%)")
    print(f"Well known (score 2-3):  {len(well_known)} ({len(well_known)/len(valid)*100:.1f}%)")

    if blind_spots:
        print(f"\nSample blind spots:")
        for s in blind_spots[:5]:
            print(f"  - {s['term']}: score={s['score']}")
            print(f"    Model said: {s['model_definition'][:100]}...")
            print(f"    Reason: {s['score_reason'][:100]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investopedia_path", default="InvestopediaDatasetFinal.json",
                    help="Path to corpus (Investopedia JSON or XFinBench knowledge_bank.csv); "
                         "interpreted according to --corpus_type")
    ap.add_argument("--corpus_type", choices=["investopedia", "xfinbench", "caregiver"],
                    default="investopedia",
                    help="Format of the corpus file at --investopedia_path")
    ap.add_argument("--model", default="vllm:Qwen/Qwen2.5-7B-Instruct",
                    help="Model to evaluate (vllm:name or openai:name)")
    ap.add_argument("--scorer", default="openai:gpt-4o-mini",
                    help="Model to score definitions (openai:name)")
    ap.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--max_terms", type=int, default=0,
                    help="Max terms to evaluate (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--score_only", action="store_true",
                    help="Skip generation, score existing output file")
    ap.add_argument("--generated", default="",
                    help="Path to existing generated definitions (for --score_only)")
    ap.add_argument("--skip_scoring", action="store_true",
                    help="Only generate definitions, skip scoring")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        # Load existing results and score them
        gen_path = args.generated or args.output
        with open(gen_path) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        scored = score_definitions(results, args.scorer, args.workers,
                                   api_key=args.api_key)
        analyze_results(scored, data.get("model", "unknown"))
        # Save
        summary = {
            "model": data.get("model", "unknown"),
            "scorer": args.scorer,
            "n_terms": len(scored),
            "results": scored,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")
        return

    # Full pipeline: generate + optionally score
    terms = load_corpus(args.corpus_type, args.investopedia_path)
    if args.max_terms:
        terms = terms[:args.max_terms]
    print(f"Loaded {len(terms)} terms from {args.corpus_type}")

    # Generate
    model_kwargs = {"vllm_base_url": args.vllm_base_url, "api_key": args.api_key}
    results = generate_definitions(terms, args.model, args.workers, **model_kwargs)

    errors = sum(1 for r in results if r.get("error"))
    print(f"Generation done: {len(results) - errors} ok, {errors} errors")

    if not args.skip_scoring and args.api_key:
        # Score
        scored = score_definitions(results, args.scorer, args.workers,
                                   api_key=args.api_key)
        analyze_results(scored, args.model)

        summary = {
            "model": args.model,
            "scorer": args.scorer,
            "n_terms": len(scored),
            "avg_score": sum(s["score"] for s in scored if s["score"] >= 0) /
                         max(1, sum(1 for s in scored if s["score"] >= 0)),
            "blind_spots": [s["term"] for s in scored if s["score"] <= 1],
            "well_known": [s["term"] for s in scored if s["score"] >= 2],
            "results": scored,
        }
    else:
        summary = {
            "model": args.model,
            "n_terms": len(results),
            "results": results,
        }
        if args.skip_scoring:
            print("Scoring skipped (--skip_scoring). Run with --score_only later.")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
