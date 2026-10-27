"""Re-score Investopedia definition modeling under Jhirad 2023's
1st-KTAW only gold reference (no combinations).

Reuses the already-generated model definitions stored in
outputs/analysis/term_eval/<model>_scored.json; only the METEOR
computation changes (single reference = first key takeaway).
"""

import json
import argparse
from pathlib import Path

import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.translate.meteor_score import meteor_score as nltk_meteor
import os

ROOT = os.environ.get("DEFRAG_ROOT", ".")


def score_one(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    try:
        return float(nltk_meteor([ref_tokens], hyp_tokens))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored_dir', default=os.path.join(ROOT, "outputs/analysis/term_eval"))
    ap.add_argument('--output', default=os.path.join(ROOT, "outputs/analysis/term_eval/meteor_1st_ktaw_only.json"))
    args = ap.parse_args()

    targets = [
        ('Qwen-2.5-7B',       'qwen7b_scored.json'),
        ('Qwen-2.5-14B',      'qwen14b_scored.json'),
        ('Llama-3.1-8B',      'llama31_8b_scored.json'),
        ('Mistral-Small-24B', 'mistral24b_scored.json'),
        ('GPT-5.2',           'gpt52_scored.json'),
    ]

    out = {}
    for label, fname in targets:
        p = Path(args.scored_dir) / fname
        if not p.exists():
            print(f'MISSING {p}')
            continue
        data = json.load(open(p))
        results = data['results']
        meteors = []
        for r in results:
            ref = r.get('reference', '')
            hyp = r.get('model_definition', '')
            if not ref or not hyp:
                continue
            meteors.append(score_one(ref, hyp))
        avg = 100 * sum(meteors) / len(meteors) if meteors else 0
        out[label] = {
            'n_scored': len(meteors),
            'avg_meteor_1st_ktaw_only': round(avg, 2),
            'avg_meteor_all_combinations': round(100 * data.get('avg_meteor', 0), 2),
        }
        print(f'{label}: 1st-KTAW only = {avg:.2f}  (vs all-combinations = {100*data["avg_meteor"]:.2f}) n={len(meteors)}')

    json.dump(out, open(args.output, 'w'), indent=2)
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
