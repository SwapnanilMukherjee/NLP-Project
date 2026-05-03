# Part 2 QA Evaluation

Task: RAG question answering over the r/gradadmissions repository.
Metrics: ROUGE-L, BERTScore F1, and faithfulness flags. Faithfulness can be manually overridden in the review CSV.

## Overall Summary

| provider | model | n | rouge_l | bertscore_f1 | faithfulness_manual_or_auto_pct |
| --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash | 18 | 0.1652 | 0.9583 | 88.8889 |

## Type Breakdown

| provider | type | n | rouge_l | bertscore_f1 | faithfulness_manual_or_auto_pct |
| --- | --- | --- | --- | --- | --- |
| gemini | adversarial_absent | 2 | 0.2167 | 0.9563 | 50.0000 |
| gemini | factual_community | 6 | 0.1596 | 0.9555 | 83.3333 |
| gemini | opinion_summary | 10 | 0.1582 | 0.9603 | 100.0000 |

## Qualitative Analysis

- `gemini` is the only evaluated provider for `rouge_l` in this table.
- `gemini` is the only evaluated provider for `bertscore_f1` in this table.
- `gemini` is the only evaluated provider for `faithfulness_manual_or_auto_pct` in this table.
- The hardest questions by mean ROUGE-L are listed below; these are usually broad opinion summaries or adversarial items where the model must avoid hallucination.

| example_id | mean_rouge_l | question | type |
| --- | --- | --- | --- |
| qa01 | 0.0839 | What is the main theme of r/gradadmissions in this corpus? | factual_community |
| qa08 | 0.1034 | What advice do users give for graduate interviews? | opinion_summary |
| qa10 | 0.1169 | How do users describe the emotional experience of rejection or waiting? | opinion_summary |

Manual faithfulness review file: `/home/swapnanil_mukherjee/nlp_project/data/part2/reports/qa_manual_faithfulness_review.csv`

Faithful answers should stay grounded in retrieved Reddit evidence and explicitly refuse adversarial questions whose answer is absent from the corpus.
