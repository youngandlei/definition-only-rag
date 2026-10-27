# Analysis

## The per-question correlation grid

`correlation_grid.json` holds the forty cells behind the central null: five
models, two term-knowledge measurements (surface METEOR and an LLM judge), two
aggregations over a question's terms (`mean`, `min`), and two outcomes
(`rcorr`, against baseline correctness; `rdelta`, against per-question RAG
gain). `correlation_grid_intersection.json` is the same grid on the stricter
both-annotator subset.

| grid | questions | max abs r |
|---|---|---|
| human-validated | 995 | 0.056 |
| both-annotator intersection | 959 | 0.050 |

Every cell lies inside an absolute r of 0.06 in both, which is the bound the
paper draws its conclusion from.

`correlation_grid.py` recomputes the grid. It needs the per-model term-knowledge
scores and paired per-question outcomes, which are not in this repository; set
`DEFRAG_ROOT` to a tree that holds them.

## The invented-terms control

`invented_terms_eval.py` scores the fifty synthetic items in
`../invented_terms/` with and without the definition in the prompt. It needs API
or vLLM access to the models under test.

## Reported numbers

| script | produces |
|---|---|
| `bootstrap_table1_cis.py` | the paired-bootstrap intervals on the main panel |
| `masking_ablation.py` | the leakage controls: dropping answer-matching snippets, masking term names |
| `invented_terms_eval.py` | the invented-terms control |
| `rescore_1st_ktaw_only.py` | the null under the stricter first-KTAW scoring protocol |
| `retrieval_hit_recall.py` | recall@3 and hit-rate@3 against the validated annotation |
| `has_term_cohort_ablation.py` | the zero-term control cohort |
