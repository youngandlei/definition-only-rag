# Definition Probe

194 hand-reviewed multiple-choice items whose answer is a value carried by the
source term's definition. The set exists to bound the opposite regime from the
natural benchmarks: the one in which a retrieved definition is decisive.

Each item has:

| field | |
|---|---|
| `question`, `choices`, `answer`, `answer_text` | the item; `answer` indexes `choices` |
| `source_term`, `source_term_id` | the Investopedia term the answer depends on |
| `explanation` | why the answer follows from that term's definition |
| `subject`, `type`, `q_idx` | bookkeeping |

Source terms were drawn from a robust-mid knowledge band: terms every model in
the panel half-knows, so neither ceiling nor floor effects dominate. Distractors
were written to be plausible as prose, so that prose plausibility alone cannot
answer the item. `rubric.md` is the retention rubric every item was reviewed
against.

Definitions themselves are not included here; see `../annotation/README.md` on
obtaining the glossary.
