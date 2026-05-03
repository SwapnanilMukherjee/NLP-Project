# Hindi Language Evaluation

Chosen Indian language: Hindi.
Implemented formats:
- Translation
- Cross-lingual QA
- Summarisation
- Extra experiment: Hinglish to clean Hindi normalization

## Task Sizes

| task_label | examples |
| --- | --- |
| Cross-lingual QA | 20 |
| Extra: Hinglish to Clean Hindi | 20 |
| Summarisation | 20 |
| Translation | 24 |

## Overall Provider Summary

| provider | model | n | chrf | multilingual_bertscore_f1 | manual_fluency_1_to_5 | manual_adequacy_1_to_5 |
| --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash | 84 | 0.2027 | 0.9859 | 1.0000 | 1.0000 |

## Per-Task Summary

| provider | model | task | task_label | n | chrf | multilingual_bertscore_f1 | manual_fluency_1_to_5 | manual_adequacy_1_to_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash | code_mixed_normalization | Extra: Hinglish to Clean Hindi | 20 | 0.2578 | 0.9878 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | cross_lingual_qa | Cross-lingual QA | 20 | 0.1398 | 0.9835 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | summarization | Summarisation | 20 | 0.0855 | 0.9831 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | translation | Translation | 24 | 0.3067 | 0.9886 | 1.0000 | 1.0000 |

## Edge-case Analysis

The suite deliberately includes code-mixed Hinglish, Reddit slang, named entities, admissions acronyms, funding talk, interviews, portals, and GradCafe-style anxiety language.

| tag | mean_chrf |
| --- | --- |
| unfunded_masters | 0.0320 |
| missing_materials | 0.0514 |
| low_gpa | 0.0598 |
| waiting | 0.0632 |
| community | 0.0645 |

## Qualitative Analysis

- `gemini` is the only evaluated provider for `chrf` in this table.
- `gemini` is the only evaluated provider for `multilingual_bertscore_f1` in this table.
- Cross-lingual QA and summarisation are the most demanding formats because they require the model to read English Reddit evidence and produce a fluent Hindi answer.
- The extra Hinglish-normalization experiment focuses on transforming code-mixed Reddit-style admissions text into cleaner Hindi while preserving acronyms and named entities.

Manual fluency/adequacy review file: `/home/swapnanil_mukherjee/nlp_project/data/part2/reports/hindi_suite_manual_review.csv`
