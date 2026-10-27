# Scenario Bench V6 — Review Rubric

**Purpose.** Audit each of the 194 generated V6 MCQs against a written rubric to support the paper's "human-reviewed V6 bundle" claim.

**Reviewer.** First author (BS Finance background).

**Input.** `review_bundle/v6_review.csv` — one row per generated MCQ, including the source Investopedia term and definition used by the generator.

**Output.** Same CSV with `decision` and `notes` columns filled.

---

## Decision values

| Value | Meaning |
|---|---|
| `keep`   | The MCQ passes all four criteria as-is. |
| `revise` | The MCQ has a fixable distractor issue (e.g., one distractor is too obviously wrong). Use `notes` to record the specific edit needed; the edit can be applied afterward. |
| `drop`   | The MCQ has a fundamental problem (ambiguous correct answer, leaked term, multiple valid answers) that cannot be patched by editing one distractor. |

---

## Four retention criteria

An MCQ is `keep` **if and only if all four** hold:

1. **Definitional answer.** The correct answer is verbatim or directly derivable from the source term's Investopedia definition. A reader with the Investopedia entry in front of them must be able to reach the marked correct option deterministically.
2. **Type-consistent choices.** All four choices are values of the same type (all percentages, all years, all currency amounts, all organisation names, etc.). Mixed-type choice sets fail.
3. **No alternative correct answer.** None of the three distractors is also a valid answer under any reasonable reading of the question and the term's definition.
4. **No surface leak.** The source term's exact name does not appear in any choice; the question does not phrase itself so that the correct option is recoverable from word matching without the term's content (e.g., the question must not say "Under the X rule" while choice C is the only option mentioning "X").

If any one criterion fails:
- Criterion 2 alone (mixed types) is often patchable → `revise` with a note like "rewrite choice C to match type".
- Criteria 1, 3, 4 failures are usually structural → `drop`.

### Examples (illustrative)

- *Source term:* Overall Liquidity Ratio. *Question:* "An insurer reports Total Assets = \$120M, Total Liabilities = \$80M, Conditional Reserves = \$20M. What is its Overall Liquidity Ratio?" *Choices:* 1.20, 1.40, 1.71, 0.86. *Correct:* 1.20. → **keep** (formula applied directly from definition; all choices are same-type ratios; no distractor is also correct; no surface leak).
- *Source term:* Wholesale Insurance. *Question:* "Wholesale insurance is typically sold to employer groups with what employee size threshold?" *Choices:* Fewer than 10 / Fewer than 50 / Fewer than 100 / Fewer than 500. *Correct:* Fewer than 10. → **keep** (definition specifies "usually with fewer than 10 employees"; type-consistent; no alternative answer).
- Hypothetical: a question where one distractor is "$1B" while another distractor is "the year 1985" → **revise** (criterion 2 failed; choose a type and rewrite the off-type distractor).
- Hypothetical: a question where the term's definition admits two equally valid threshold values → **drop** (criterion 3 failed; not patchable by single-distractor edit).

---

## Workflow

1. **Open `v6_review.csv` in a spreadsheet app.**
2. **The CSV is grouped by source term** — adjacent rows often share the same term family. Use this to keep context across the review.
3. **For each row**: read source definition → read question → read 4 choices → check correctness against the explanation → apply 4 criteria → decide.
4. **~2 minutes per row.** Total: 194 × 2 min ≈ **6.5 hours**. Single afternoon.
5. **For `revise` decisions**, write the proposed edit in `notes` (e.g., "choice C → '1.40' to make type-consistent with others").

---

## Reporting after completion

After all rows are decided, run:
```
python3 review_bundle/summarise_v6.py
```
which will print:
- Total items
- `keep` count and percentage
- `revise` count and percentage (with note inventory)
- `drop` count and percentage
- Reason breakdown

These numbers will fill the `XXX` placeholders in §5 and Appendix B of the paper.
