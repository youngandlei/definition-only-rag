# Routing signals

The eight signals swept against the per-question oracle, each on
Qwen-2.5-7B / MMLU Finance, each threshold swept in its own favour.

| script | signal |
|---|---|
| `semantic_entropy_selection.py` | semantic entropy over sampled generations |
| `confidence_gain_selection.py` | predictive entropy and the top-1 minus top-2 margin |
| `term_knowledge_gating.py` | the model's own term knowledge, the signal the paper is about |
| `retrieval_threshold.py` | the reranker's top-1 score |
| `learned_gate_selection.py` | a logistic gate over cheap logit, entropy and retrieval features |
| `hidden_state_probe.py` | a probe on the RAG-minus-baseline hidden state |
| `llm_judge_selection.py` | the external judge, the one method that beats always-on retrieval |
| `temperature_scaling_gating.py` | the calibration check on the confidence margin |
