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
| gemini | gemini-2.5-flash | 84 | 0.2122 | 0.9862 | 1.0000 | 1.0000 |
| groq | llama-3.3-70b-versatile | 84 | 0.4735 | 0.9900 | 4.0000 | 3.4000 |

## Per-Task Summary

| provider | model | task | task_label | n | chrf | multilingual_bertscore_f1 | manual_fluency_1_to_5 | manual_adequacy_1_to_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash | code_mixed_normalization | Extra: Hinglish to Clean Hindi | 20 | 0.2751 | 0.9877 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | cross_lingual_qa | Cross-lingual QA | 20 | 0.1477 | 0.9841 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | summarization | Summarisation | 20 | 0.1116 | 0.9838 | 1.0000 | 1.0000 |
| gemini | gemini-2.5-flash | translation | Translation | 24 | 0.2972 | 0.9886 | 1.0000 | 1.0000 |
| groq | llama-3.3-70b-versatile | code_mixed_normalization | Extra: Hinglish to Clean Hindi | 20 | 0.6216 | 0.9934 | 4.0000 | 3.0000 |
| groq | llama-3.3-70b-versatile | cross_lingual_qa | Cross-lingual QA | 20 | 0.3167 | 0.9860 | 4.0000 | 3.0000 |
| groq | llama-3.3-70b-versatile | summarization | Summarisation | 20 | 0.2949 | 0.9856 | 4.0000 | 2.0000 |
| groq | llama-3.3-70b-versatile | translation | Translation | 24 | 0.6296 | 0.9939 | 4.0000 | 4.5000 |

## Edge-case Analysis

The suite deliberately includes code-mixed Hinglish, Reddit slang, named entities, admissions acronyms, funding talk, interviews, portals, and GradCafe-style anxiety language.

| tag | mean_chrf |
| --- | --- |
| scholarship | 0.1610 |
| program_selection | 0.1737 |
| mscs | 0.1931 |
| multiple_admits | 0.1940 |
| low_gpa | 0.1971 |

## Qualitative Analysis

- `groq` scores higher than `gemini` on `chrf` in this table.
- `groq` scores higher than `gemini` on `multilingual_bertscore_f1` in this table.
- Cross-lingual QA and summarisation are the most demanding formats because they require the model to read English Reddit evidence and produce a fluent Hindi answer.
- The extra Hinglish-normalization experiment focuses on transforming code-mixed Reddit-style admissions text into cleaner Hindi while preserving acronyms and named entities.

Manual fluency/adequacy review file: `/home/swapnanil_mukherjee/nlp_project/data/part2/reports/hindi_suite_manual_review.csv`
