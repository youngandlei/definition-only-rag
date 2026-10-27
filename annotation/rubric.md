# MMLU Finance Term Annotation — Review Rubric

**Purpose.** Convert the existing LLM-only annotation into a human-validated annotation, with a documented edit log that supports the paper's contribution claim.

**Reviewer.** First author (BS Finance background).

**Input.** `review_bundle/annotation_review.csv` — one row per (question, candidate Investopedia term). Disagreement cases (where GPT-5.2 and GPT-4o differ on whether to include a term) are sorted to the top of the file.

**Output.** Same CSV with the `decision` and `notes` columns filled.

---

## Decision values

Use exactly one of these in the `decision` column:

| Value | Meaning |
|---|---|
| `keep`   | The (question, term) pair is correct. The term's Investopedia definition is required to answer the question correctly. |
| `remove` | The pair is incorrect. The term is not needed to answer this question (either irrelevant, or only superficially mentioned). |
| `add`    | (Only for rows you create yourself; see "Adding missed pairs" below.) |

Leave `decision` blank for any row you have not yet reviewed — this makes it easy to resume.

---

## Three retention criteria

A (question, term) pair is `keep` **if and only if all three** hold:

1. **Definitional support.** The term's Investopedia definition directly supports the correct answer. A domain-trained reader looking only at the term's definition could pick the correct option, or eliminate enough wrong options to land there.
2. **Necessity.** Removing the term would make the answer genuinely ambiguous to a domain-trained reader. If a domain-trained reader would still confidently answer the question without ever knowing the term's definition, the term is not necessary.
3. **Semantic role.** The term contributes to the *answer's* semantic construction, not just to the question's surface form. Terms that appear in the question stem but are not load-bearing for distinguishing the correct option from the distractors fail this criterion.

If any one criterion fails, the pair is `remove`.

### Examples (illustrative)

- *Question:* "Under state law, Boca Co. may reimburse the state directly for actual unemployment claims or it may pay 3% of eligible gross wages. ..." *Term:* `Unemployment Insurance` → **keep** (all three criteria hold).
- *Question:* "The first phase of an audit engagement is to ..." *Term:* `Audit` → **remove** (criterion 2 fails: the answer follows from general audit-procedure knowledge; the term's specific definition is not needed).
- *Question:* "What is the expected return of a portfolio with weights 0.5 and 0.5 on assets returning 8% and 12%?" *Term:* `Portfolio` → **remove** (criterion 3 fails: the term appears in the question but the answer is a numerical computation, not a definitional appeal).

---

## Adding missed pairs

If you encounter a question whose answer clearly depends on a term that neither LLM annotator picked, you may add a row at the bottom of the CSV with:
- `subject`, `question`, `choices`, `gold_letter`, `gold_text` copied from a row of the same question
- `doc_id` and `term` filled in (look up the Investopedia entry separately if needed)
- `investopedia_reference` filled in with the Investopedia text
- `annotator_source` = `human added`
- `decision` = `add`
- `notes` explaining why

Only do this when you are confident the missed term is load-bearing. Do not stretch.

---

## Workflow

1. **Open `annotation_review.csv` in a spreadsheet app** (Numbers / Excel / Google Sheets).
2. **Disagreement cases (`annotator_source` = `GPT-5.2 only` or `GPT-4o only`) come first.** These are the highest-yield rows to review — they are the cases where the LLMs themselves were uncertain.
3. **Then proceed through the `both` rows.** These are cases the two LLMs agreed on; most should be `keep`, but spot-check is required.
4. **Aim for ~1–2 minutes per row.** Read the question, glance at the term + Investopedia reference, apply the three criteria, decide.
5. **Drop a brief `notes` entry only when the decision is non-obvious.** Most `keep` decisions need no note; `remove` and ambiguous cases should briefly say which criterion failed.

---

## Reporting after completion

After all rows are decided, run:
```
python3 review_bundle/summarise_review.py
```
which will print:
- Total pairs reviewed
- `keep` count and percentage
- `remove` count and percentage
- `add` count
- Per-source-LLM breakdown (kept-from-GPT-5.2-only, kept-from-GPT-4o-only, etc.)
- Final per-question average term count

These numbers will then fill the `XX.X` placeholders in the paper.

---

## Estimated time

- 2,233 (q, term) pairs at ~1 min each ≈ **37 hours**
- Disagreement-first ordering means the first ~12 hours covers the highest-uncertainty third of the data; after that the pace accelerates as the "both LLMs agree" rows mostly confirm.
- Spread across ~6–8 working days at 4–6 hours per day is realistic.
