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

## Release status

The artifacts are prepared and will be released here. They comprise:

- **Per-question term annotation.** For each of 1,124 MMLU Finance questions, the
  Investopedia terms whose definitions are load-bearing for the correct answer, with the
  annotation rubric, the second-annotator pass, and the agreement report.
- **Controlled probe set.** 194 hand-reviewed items whose answer is a value carried by the
  source term's definition, with source terms and construction notes.
- **Invented terms.** 50 synthetic financial terms with hand-written definitions, used to
  separate reading an injected definition from recalling a memorised one.
- **Evaluation harness and analysis code.** The retrieval and scoring pipeline together
  with the scripts behind every reported number, including the per-model term-knowledge
  scores and the paired per-question outcomes that reproduce the correlation analysis.

The Investopedia glossary of Jhirad et al. (2023) is not publicly hosted and will not be
redistributed here. Instructions for obtaining it from its authors, and for resolving the
term identifiers used in the annotation, will accompany the release.

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

Code will be released under the MIT License. The annotation, the probe set, and the
invented-terms set will be released under CC BY 4.0. Neither covers the Investopedia
glossary, which is not redistributed here.
