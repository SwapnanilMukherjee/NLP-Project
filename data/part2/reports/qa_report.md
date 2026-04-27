# Part 2 QA Evaluation

Task: RAG question answering over the r/gradadmissions repository.
Metrics: ROUGE-L, BERTScore F1, and faithfulness flags. Faithfulness can be manually overridden in the review CSV.

## Overall Summary

| provider | model | n | rouge_l | bertscore_f1 | faithfulness_manual_or_auto_pct |
| --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash | 18 | 0.1814 | 0.9641 | 88.8889 |
| groq | llama-3.3-70b-versatile | 18 | 0.1563 | 0.9567 | 94.4444 |

## Type Breakdown

| provider | type | n | rouge_l | bertscore_f1 | faithfulness_manual_or_auto_pct |
| --- | --- | --- | --- | --- | --- |
| gemini | adversarial_absent | 2 | 0.2059 | 0.9605 | 50.0000 |
| gemini | factual_community | 6 | 0.1876 | 0.9645 | 83.3333 |
| gemini | opinion_summary | 10 | 0.1728 | 0.9646 | 100.0000 |
| groq | adversarial_absent | 2 | 0.2958 | 0.9631 | 100.0000 |
| groq | factual_community | 6 | 0.1524 | 0.9555 | 83.3333 |
| groq | opinion_summary | 10 | 0.1307 | 0.9561 | 100.0000 |

## Qualitative Analysis

- `gemini` scores higher than `groq` on `rouge_l` in this table.
- `gemini` scores higher than `groq` on `bertscore_f1` in this table.
- `groq` scores higher than `gemini` on `faithfulness_manual_or_auto_pct` in this table.
- The hardest questions by mean ROUGE-L are listed below; these are usually broad opinion summaries or adversarial items where the model must avoid hallucination.

| example_id | mean_rouge_l | question | type |
| --- | --- | --- | --- |
| qa12 | 0.0977 | What do users say about contacting professors before applying? | opinion_summary |
| qa08 | 0.1025 | What advice do users give for graduate interviews? | opinion_summary |
| qa01 | 0.1274 | What is the main theme of r/gradadmissions in this corpus? | factual_community |

Manual faithfulness review file: `/home/swapnanil_mukherjee/nlp_project/data/part2/reports/qa_manual_faithfulness_review.csv`

Faithful answers should stay grounded in retrieved Reddit evidence and explicitly refuse adversarial questions whose answer is absent from the corpus.
