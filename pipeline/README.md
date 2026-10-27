# Pipeline

Retrieval, term-knowledge measurement, and the construction of the annotation.
Each script expects `DEFRAG_ROOT` to point at a tree holding the glossary, the
benchmarks and the model traces; none of those are redistributed here.

| script | |
|---|---|
| `retrieve_candidates.py` | hybrid retrieval over the glossary: lexical and dense lists, reciprocal-rank fusion, cross-encoder rerank |
| `term_knowledge_eval.py` | prompts each model for a definition of every term and scores the generations |
| `term_score.py` | the two scoring protocols: surface METEOR and the LLM sense-correctness judge |
| `annotate_mmlu_terms.py` | stages 1 and 2 of the annotation: dense top-30 candidates, then LLM filtering |
| `inter_annotator_agreement.py` | agreement between the two LLM annotators, producing `../annotation/agreement.json` |
