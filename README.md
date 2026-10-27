# Definition-Only RAG for Financial QA

Artifacts for **"Definition-Only RAG for Financial QA: Average Gains and a Diagnostic
Boundary"** (Findings of EMNLP 2026).

Prior work reports that a language model's ability to define domain terms tracks its
benchmark performance in that domain, which invites using that ability to decide when
retrieval should fire. We test the definition in both roles, as retrieved payload and as
control signal. Retrieving a single glossary definition produces a structured gain on
financial multiple-choice QA, concentrated in a mid-capability band rather than scaling
with model size. The same term-knowledge signal carries no per-question information. It
does not locate the questions where a definition changes the answer, and no internal
routing signal approaches the per-question oracle.

## Contents

```
pipeline/          retrieval, term-knowledge measurement, annotation build
annotation/        per-question term annotation, rubric, agreement report
definition_probe/  194 reviewed items whose answer the definition carries
invented_terms/    50 synthetic terms, for separating reading from recall
analysis/          the correlation grid and the reported numbers
routing/           the eight-signal sweep against the oracle
talk/              the slides as presented
poster/            the poster as presented
```

| | |
|---|---|
| `annotation/mmlu_finance_terms.csv` | 1,124 questions; 1,006 carry at least one validated term |
| `definition_probe/definition_probe.json` | 194 items |
| `invented_terms/invented_terms.json` | 50 items |
| `analysis/correlation_grid.json` | 40 cells, all inside an absolute r of 0.06 |

Each directory has a README covering format, construction, and the numbers it
reproduces.

## Using the annotation

`gold_inv_term_id` indexes the curated Investopedia term bank of Jhirad et al.
(2023), which its authors distribute on request. With a copy in hand:

```bash
python annotation/resolve_terms.py --glossary InvestopediaDataset.json \
                                   --in  annotation/mmlu_finance_terms.csv \
                                   --out mmlu_finance_terms_named.csv
```

`annotation/summarise.py` prints the coverage breakdown from the CSV alone.

## Running the code

The scripts read the glossary, the benchmarks, and the model traces from a tree
you point `DEFRAG_ROOT` at.

```bash
pip install -r requirements.txt
export DEFRAG_ROOT=/path/to/your/tree
python analysis/bootstrap_table1_cis.py
```

## Citation

```bibtex
@inproceedings{ouyang2026definition,
  title     = {Definition-Only {RAG} for Financial {QA}: Average Gains and a Diagnostic Boundary},
  author    = {Ouyang, Leyi and Marrese-Taylor, Edison and Matsuo, Yutaka},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

Code is released under the MIT License. The annotation, the probe set and the
invented-terms set are released under CC BY 4.0.
