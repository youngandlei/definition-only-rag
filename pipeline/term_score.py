#!/usr/bin/env python3
"""
Score model-generated financial term definitions against Investopedia ground truth.

Uses COMET, METEOR, and BLEU — the same metrics as:
  "Evaluating LLMs' Understanding of Financial Terminology via Definition Modeling"
  (Jhirad, Marrese-Taylor, Matsuo, 2023)

No LLM-as-judge. All metrics are deterministic and locally computed.

Usage:
  python term_score.py \
    --generated outputs/analysis/term_eval/qwen7b_defs.json \
    --investopedia_path InvestopediaDatasetFinal.json \
    --output outputs/analysis/term_eval/qwen7b_scored.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# BLEU (local implementation, no deps)
# ---------------------------------------------------------------------------

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Corpus-level BLEU between a single reference and hypothesis."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(ngrams(ref_tokens, n))
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(clipped / total)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Geometric mean of precisions
    log_avg = 0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += math.log(p) / max_n
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# METEOR (simplified version using nltk)
# ---------------------------------------------------------------------------

def meteor_score_single(reference: str, hypothesis: str) -> float:
    """Compute METEOR score for a single ref-hyp pair."""
    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        return _meteor([ref_tokens], hyp_tokens)
    except Exception:
        # Fallback: F1 of unigram overlap
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        if not ref_set or not hyp_set:
            return 0.0
        overlap = ref_set & hyp_set
        p = len(overlap) / len(hyp_set)
        r = len(overlap) / len(ref_set)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# COMET
# ---------------------------------------------------------------------------

_COMET_MODEL = None


def load_comet():
    global _COMET_MODEL
    if _COMET_MODEL is None:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(model_path)
    return _COMET_MODEL


def comet_score_batch(sources: list, references: list, hypotheses: list, gpus: int = 1) -> list:
    """Compute COMET scores for a batch."""
    model = load_comet()
    data = [{"src": s, "mt": h, "ref": r}
            for s, r, h in zip(sources, references, hypotheses)]
    output = model.predict(data, batch_size=64, gpus=gpus)
    return output.scores


# ---------------------------------------------------------------------------
# BERTScore (Jhirad et al. 2023 also report this; uses roberta-large for en)
# ---------------------------------------------------------------------------

def bertscore_batch(references: list, hypotheses: list, batch_size: int = 64) -> list:
    """Return BERTScore F1 for each (reference, hypothesis) pair."""
    from bert_score import score as _bscore
    _, _, f1 = _bscore(
        hypotheses, references, lang="en",
        batch_size=batch_size, rescale_with_baseline=True, verbose=True,
    )
    return [float(x) for x in f1]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_investopedia_lookup(path: str) -> dict:
    """Load Investopedia as term -> definition lookup."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    types = data.get("Type", {})
    terms = data.get("Term Name", {})
    ktaws = data.get("KTAW", {})

    lookup = {}
    for key, typ in types.items():
        if typ != "terms":
            continue
        term = str(terms.get(key, "")).strip()
        if not term:
            continue
        ktaw_list = ktaws.get(key, []) or []
        if isinstance(ktaw_list, list):
            takeaways = [s.strip() for s in ktaw_list if s and s.strip()]
        else:
            takeaways = [str(ktaw_list).strip()]
        if not takeaways:
            continue

        lookup[term] = {
            "first_takeaway": takeaways[0],
            "all_takeaways": takeaways,
            "gold_candidates": gold_candidates(takeaways),
        }
    return lookup


def load_xfinbench_lookup(path: str) -> dict:
    """Load XFinBench knowledge_bank.csv as term -> definition lookup.

    XFinBench's term_definition is a single textbook passage (no
    multiple-takeaway structure like Investopedia), so gold_candidates is a
    single-element list. term_name may be comma-separated; first piece is used.
    """
    import csv as _csv
    lookup = {}
    with open(path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            term_name = (row.get("term_name") or "").strip()
            definition = (row.get("term_definition") or "").strip()
            if not term_name or not definition:
                continue
            primary_term = term_name.split(",")[0].strip()
            if not primary_term:
                continue
            lookup[primary_term] = {
                "first_takeaway": definition,
                "all_takeaways": [definition],
                "gold_candidates": [definition],
            }
    return lookup


def load_corpus_lookup(corpus_type: str, path: str) -> dict:
    if corpus_type == "investopedia":
        return load_investopedia_lookup(path)
    if corpus_type == "xfinbench":
        return load_xfinbench_lookup(path)
    raise ValueError(f"Unknown corpus_type: {corpus_type}")


def gold_candidates(ktaw: list) -> list:
    """Build all gold reference candidates from Key Takeaways.

    Following Jhirad et al. (2023): first KTAW is always included,
    combined with all subsets of remaining KTAWs. This ensures evaluation
    covers both concise and comprehensive definitions.
    """
    from itertools import combinations as _combinations
    if not ktaw:
        return []
    first = ktaw[0].strip()
    rest = [s.strip() for s in ktaw[1:] if s.strip()]
    candidates = [first]
    for r in range(1, len(rest) + 1):
        for combo in _combinations(range(len(rest)), r):
            parts = [first] + [rest[i] for i in combo]
            candidates.append(" ".join(parts))
    return candidates


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="JSON with model definitions")
    ap.add_argument("--investopedia_path", default="InvestopediaDatasetFinal.json",
                    help="Path to corpus reference (Investopedia JSON or XFinBench knowledge_bank.csv)")
    ap.add_argument("--corpus_type", choices=["investopedia", "xfinbench"],
                    default="investopedia",
                    help="Format of --investopedia_path")
    ap.add_argument("--output", required=True)
    ap.add_argument("--use_comet", action="store_true",
                    help="Include COMET scoring (slower, needs GPU or CPU time)")
    ap.add_argument("--comet_gpus", type=int, default=1,
                    help="GPUs for COMET (1=use the RTX8000, 0=CPU)")
    ap.add_argument("--use_bertscore", action="store_true",
                    help="Include BERTScore F1 (roberta-large; needs GPU for speed)")
    ap.add_argument("--max_terms", type=int, default=0, help="Limit for testing")
    args = ap.parse_args()

    # Load model definitions
    with open(args.generated) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    model_name = data.get("model", "unknown") if isinstance(data, dict) else "unknown"

    # Load ground truth
    lookup = load_corpus_lookup(args.corpus_type, args.investopedia_path)

    print(f"Model: {model_name}")
    print(f"Generated definitions: {len(results)}")
    print(f"Investopedia terms: {len(lookup)}")

    # Match and score — following Jhirad et al.: max over all gold candidates
    scored = []
    skipped = 0
    for r in results:
        term = r.get("term", "")
        model_def = r.get("model_definition", "")

        if not model_def or term not in lookup:
            skipped += 1
            continue

        gt = lookup[term]
        candidates = gt["gold_candidates"]

        # Max BLEU over all candidates
        b = max((bleu_score(ref, model_def) for ref in candidates), default=0)
        # Max METEOR over all candidates
        m = max((meteor_score_single(ref, model_def) for ref in candidates), default=0)

        scored.append({
            "term": term,
            "model_definition": model_def,
            "reference": gt["first_takeaway"],
            "n_candidates": len(candidates),
            "bleu": round(b, 4),
            "meteor": round(m, 4),
            "doc_id": r.get("doc_id", ""),
        })

    if args.max_terms and len(scored) > args.max_terms:
        scored = scored[:args.max_terms]

    print(f"Scored: {len(scored)}, Skipped: {skipped}")

    # COMET (optional, batch)
    if args.use_comet and scored:
        print("Computing COMET scores...")
        sources = [s["term"] for s in scored]
        references = [s["reference"] for s in scored]
        hypotheses = [s["model_definition"] for s in scored]
        comet_scores = comet_score_batch(sources, references, hypotheses, gpus=args.comet_gpus)
        for s, c in zip(scored, comet_scores):
            s["comet"] = round(c, 4)

    # BERTScore (optional, batch)
    if args.use_bertscore and scored:
        print("Computing BERTScore (F1)...")
        references = [s["reference"] for s in scored]
        hypotheses = [s["model_definition"] for s in scored]
        bert_f1 = bertscore_batch(references, hypotheses)
        for s, f in zip(scored, bert_f1):
            s["bertscore"] = round(f, 4)

    # Aggregate
    bleu_scores = [s["bleu"] for s in scored]
    meteor_scores = [s["meteor"] for s in scored]
    comet_scores_list = [s["comet"] for s in scored if "comet" in s]
    bert_scores_list = [s["bertscore"] for s in scored if "bertscore" in s]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_comet = sum(comet_scores_list) / len(comet_scores_list) if comet_scores_list else None
    avg_bert = sum(bert_scores_list) / len(bert_scores_list) if bert_scores_list else None

    print(f"\n{'='*50}")
    print(f"=== Results: {model_name} ===")
    print(f"{'='*50}")
    print(f"Terms scored: {len(scored)}")
    print(f"Avg BLEU:   {avg_bleu:.4f}")
    print(f"Avg METEOR: {avg_meteor:.4f}")
    if avg_comet is not None:
        print(f"Avg COMET:  {avg_comet:.4f}")
    if avg_bert is not None:
        print(f"Avg BERTScore-F1: {avg_bert:.4f}")

    # Score distribution for blind spot detection
    # Low METEOR = model doesn't know the term well
    blind_spots = [s for s in scored if s["meteor"] < 0.15]
    well_known = [s for s in scored if s["meteor"] >= 0.30]
    mid = [s for s in scored if 0.15 <= s["meteor"] < 0.30]

    print(f"\nBlind spots (METEOR < 0.15): {len(blind_spots)} ({len(blind_spots)/len(scored)*100:.1f}%)")
    print(f"Mid range (0.15-0.30):       {len(mid)} ({len(mid)/len(scored)*100:.1f}%)")
    print(f"Well known (METEOR >= 0.30): {len(well_known)} ({len(well_known)/len(scored)*100:.1f}%)")

    if blind_spots:
        print(f"\nSample blind spots:")
        for s in sorted(blind_spots, key=lambda x: x["meteor"])[:5]:
            print(f"  {s['term']}: BLEU={s['bleu']:.3f} METEOR={s['meteor']:.3f}")
            print(f"    Model: {s['model_definition'][:80]}...")
            print(f"    Truth: {s['reference'][:80]}...")

    # Save
    summary = {
        "model": model_name,
        "n_scored": len(scored),
        "avg_bleu": avg_bleu,
        "avg_meteor": avg_meteor,
        "avg_comet": avg_comet,
        "avg_bertscore": avg_bert,
        "n_blind_spots": len(blind_spots),
        "n_well_known": len(well_known),
        "blind_spot_terms": [s["term"] for s in blind_spots],
        "well_known_terms": [s["term"] for s in well_known],
        "results": scored,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
