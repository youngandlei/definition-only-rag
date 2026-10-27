# Per-question term annotation

For each MMLU Finance question, the Investopedia terms whose definitions are
load-bearing for the correct answer.

| File | Contents |
|---|---|
| `mmlu_finance_terms.csv` | the human-validated annotation; this is the one used in the paper |
| `mmlu_finance_terms_gpt4o.csv` | the second LLM annotator's pass, before adjudication |
| `mmlu_finance_terms_intersection.csv` | the stricter variant: pairs proposed by both annotators |
| `rubric.md` | the written rubric the adjudication followed |
| `agreement.json` | inter-annotator agreement between the two LLM passes |
| `resolve_terms.py` | turns term identifiers into term names, given your own copy of the glossary |
| `summarise.py` | reproduces the retention and coverage numbers below |

## Format

| column | |
|---|---|
| `question_id` | stable 12-hex digest of the item, for joining |
| `subject` | MMLU subject the question comes from |
| `question`, `choices`, `answer` | the MMLU item; `answer` indexes `choices` |
| `gold_inv_term_id` | the validated Investopedia term identifiers |

`gold_inv_term_id` is semicolon-separated when a question depends on more than
one term, and empty for a zero-term question: one whose answer needs no glossary
term, kept as a control.

The MMLU items are carried so the annotation can be joined without ambiguity;
MMLU is distributed under the MIT licence, see Hendrycks et al. (2020).
`gold_inv_term_id` is an identifier into the Investopedia term bank, which the
next section covers.

## How it was built

Dense retrieval proposed the top-30 candidate terms per question. Two LLM
annotators filtered them independently. The first author then reviewed every
pair in the union against `rubric.md`:

| | |
|---|---|
| pairs reviewed | 2,233 |
| kept | 84.7% |
| removed as misalignments | 15.3% |
| added (missed by both annotators) | 20 |

## Coverage

The file has 1,124 rows, one per MMLU Finance question, of which 1,006 carry at
least one validated term and 118 are zero-term controls, for 1,926
(question, term) pairs in all. `summarise.py` prints the breakdown.

## Resolving term identifiers

`gold_inv_term_id` indexes the curated Investopedia term bank of Jhirad et al.
(2023), which its authors distribute on request. With a copy in hand:

```bash
python resolve_terms.py --glossary /path/to/InvestopediaDataset.json \
                        --in mmlu_finance_terms.csv \
                        --out mmlu_finance_terms_named.csv
```
