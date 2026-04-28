from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reddit_insights.config import PART2_REPORTS_DIR, RAG_INDEX_DIR, settings
from reddit_insights.llm_providers import ChatProvider, build_providers
from reddit_insights.metrics import BertScoreScorer, chrf, mean_or_zero, rouge_l_f1
from reddit_insights.models import Comment, Post, RedditUser, Subreddit
from reddit_insights.part2_datasets import (
    load_bias_probes,
    load_hindi_code_mixed_normalization_set,
    load_hindi_cross_lingual_qa_set,
    load_hindi_summarization_set,
    load_hindi_translation_set,
    load_qa_eval_set,
)
from reddit_insights.rag import RagIndex, answer_question, format_context, retrieved_to_dicts


MISSING_INFO_MARKERS = (
    "not enough",
    "does not contain",
    "doesn't contain",
    "not available",
    "cannot determine",
    "can't determine",
    "insufficient",
    "do not have",
    "don't have",
    "cannot identify",
)

HINDI_QA_SYSTEM_PROMPT = (
    "You answer graduate-admissions questions in Hindi using only the supplied English Reddit context. "
    "Write Hinglish, preserve acronyms such as PhD, MSCS, GPA, SOP, LoR, TA, RA, and keep named entities unchanged. The script should be English but the language Hindi. "
    "If the context is insufficient, say so clearly in Hindi. Do not mention usernames or infer identities."
)

HINDI_SUMMARIZATION_SYSTEM_PROMPT = (
    "You summarize graduate-admissions discussions in Hindi using only the supplied English Reddit context. "
    "Write 6-7 sentences in Hinglish, preserve important acronyms and named entities, and do not invent claims that are not supported by the context. The script should be English but the language Hindi. "
)

HINGLISH_NORMALIZATION_SYSTEM_PROMPT = (
    "You rewrite Hinglish graduate-admissions text into clean, natural Hindi. "
    "Preserve acronyms such as PhD, GPA, SOP, LoR, TA, RA and preserve university or program names. "
    "Output only the normalized Hindi sentence."
)


class HindiTaskSpec(dict):
    pass


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_looks_missing(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in MISSING_INFO_MARKERS)


def auto_faithfulness_flag(example_type: str, answer: str) -> int:
    if example_type == "adversarial_absent":
        return int(answer_looks_missing(answer))
    if answer_looks_missing(answer):
        return 0
    return int("[" in answer and "]" in answer)


def load_manual_flags(path: Path, id_column: str, value_column: str) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if id_column not in df.columns or value_column not in df.columns:
        return {}
    result: dict[str, float] = {}
    for row in df.itertuples(index=False):
        key = str(getattr(row, id_column))
        value = getattr(row, value_column)
        if pd.isna(value) or str(value).strip() == "":
            continue
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            result[key] = 1.0
        elif lowered in {"0", "false", "no", "n"}:
            result[key] = 0.0
        else:
            try:
                result[key] = float(value)
            except ValueError:
                continue
    return result


def sync_manual_review_sheet(
    review_df: pd.DataFrame,
    review_path: Path,
    key_columns: list[str],
    manual_columns: list[str],
) -> pd.DataFrame:
    existing = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame()
    if not existing.empty and all(column in existing.columns for column in key_columns):
        keep_columns = key_columns + [column for column in manual_columns if column in existing.columns]
        existing = existing[keep_columns].drop_duplicates(subset=key_columns, keep="last")
        review_df = review_df.merge(existing, on=key_columns, how="left")
    for column in manual_columns:
        if column not in review_df.columns:
            review_df[column] = ""
        else:
            review_df[column] = review_df[column].fillna("")
    review_df.to_csv(review_path, index=False)
    return review_df


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results available."
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def qualitative_winner_text(summary_df: pd.DataFrame, metric: str) -> str:
    if summary_df.empty or metric not in summary_df.columns:
        return "No comparative result available."
    ordered = summary_df.sort_values(metric, ascending=False)
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    if len(ordered) == 1 or best["provider"] == worst["provider"]:
        return f"`{best['provider']}` is the only evaluated provider for `{metric}` in this table."
    return f"`{best['provider']}` scores higher than `{worst['provider']}` on `{metric}` in this table."


def retrieved_context(index: RagIndex, retrieval_query_en: str, top_k: int) -> tuple[str, list[dict]]:
    retrieved = index.search(retrieval_query_en, top_k=top_k)
    return format_context(retrieved, max_chars=6000), retrieved_to_dicts(retrieved)


def evaluate_qa(
    provider_name_list: list[str] | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    skip_bertscore: bool = False,
    reports_dir: Path = PART2_REPORTS_DIR,
) -> dict:
    examples = load_qa_eval_set()
    if limit:
        examples = examples[:limit]
    providers = build_providers(provider_name_list)
    index = RagIndex(RAG_INDEX_DIR)
    records: list[dict] = []

    for provider in providers:
        for example in examples:
            rag_answer = answer_question(provider, example["question"], index=index, top_k=top_k or settings.rag_top_k)
            result_id = f"{provider.provider_name}:{example['id']}"
            records.append(
                {
                    "result_id": result_id,
                    "provider": provider.provider_name,
                    "model": rag_answer.model,
                    "example_id": example["id"],
                    "type": example["type"],
                    "question": example["question"],
                    "reference_answer": example["reference_answer"],
                    "answer": rag_answer.answer,
                    "rouge_l": rouge_l_f1(rag_answer.answer, example["reference_answer"]),
                    "faithful_auto": auto_faithfulness_flag(example["type"], rag_answer.answer),
                    "retrieved": retrieved_to_dicts(rag_answer.retrieved),
                }
            )

    if not skip_bertscore and records:
        scorer = BertScoreScorer(settings.qa_bertscore_model)
        scores = scorer.score_pairs(
            [record["answer"] for record in records],
            [record["reference_answer"] for record in records],
        )
        for record, score in zip(records, scores):
            record["bertscore_f1"] = score
    else:
        for record in records:
            record["bertscore_f1"] = None

    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reports_dir / "qa_results.jsonl"
    write_jsonl(raw_path, records)
    flat_records = [{key: value for key, value in record.items() if key != "retrieved"} for record in records]
    result_df = pd.DataFrame(flat_records)
    result_df.to_csv(reports_dir / "qa_results.csv", index=False)

    review_path = reports_dir / "qa_manual_faithfulness_review.csv"
    review_df = sync_manual_review_sheet(
        result_df[["result_id", "provider", "example_id", "type", "question", "answer", "faithful_auto"]].copy(),
        review_path,
        key_columns=["result_id"],
        manual_columns=["faithful_manual", "notes"],
    )

    manual_flags = load_manual_flags(review_path, "result_id", "faithful_manual")
    if manual_flags:
        result_df["faithful_used"] = result_df["result_id"].map(manual_flags).fillna(result_df["faithful_auto"])
        faithfulness_column = "faithful_used"
        faithfulness_label = "faithfulness_manual_or_auto_pct"
    else:
        result_df["faithful_used"] = result_df["faithful_auto"]
        faithfulness_column = "faithful_auto"
        faithfulness_label = "faithfulness_auto_pct"

    summary_rows = []
    for provider_name, group in result_df.groupby("provider"):
        summary_rows.append(
            {
                "provider": provider_name,
                "model": group["model"].iloc[0],
                "n": int(len(group)),
                "rouge_l": mean_or_zero(group["rouge_l"].tolist()),
                "bertscore_f1": mean_or_zero([value for value in group["bertscore_f1"].tolist() if pd.notna(value)]),
                faithfulness_label: 100 * mean_or_zero(group[faithfulness_column].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "qa_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    type_rows = []
    for (provider_name, example_type), group in result_df.groupby(["provider", "type"]):
        type_rows.append(
            {
                "provider": provider_name,
                "type": example_type,
                "n": int(len(group)),
                "rouge_l": mean_or_zero(group["rouge_l"].tolist()),
                "bertscore_f1": mean_or_zero([value for value in group["bertscore_f1"].tolist() if pd.notna(value)]),
                faithfulness_label: 100 * mean_or_zero(group[faithfulness_column].tolist()),
            }
        )
    type_df = pd.DataFrame(type_rows)
    type_path = reports_dir / "qa_type_breakdown.csv"
    type_df.to_csv(type_path, index=False)
    write_qa_report(result_df, summary_df, type_df, reports_dir / "qa_report.md", review_path, faithfulness_label)
    return {"raw": raw_path, "summary": summary_path, "manual_review": review_path, "type_breakdown": type_path}


def write_qa_report(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    type_df: pd.DataFrame,
    path: Path,
    review_path: Path,
    faithfulness_label: str,
) -> None:
    hardest = pd.DataFrame()
    if not result_df.empty:
        hardest = (
            result_df.groupby("example_id", as_index=False)
            .agg(mean_rouge_l=("rouge_l", "mean"), question=("question", "first"), type=("type", "first"))
            .sort_values("mean_rouge_l", ascending=True)
            .head(3)
        )
    lines = [
        "# Part 2 QA Evaluation",
        "",
        "Task: RAG question answering over the r/gradadmissions repository.",
        "Metrics: ROUGE-L, BERTScore F1, and faithfulness flags. Faithfulness can be manually overridden in the review CSV.",
        "",
        "## Overall Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Type Breakdown",
        "",
        markdown_table(type_df),
        "",
        "## Qualitative Analysis",
        "",
        f"- {qualitative_winner_text(summary_df, 'rouge_l')}",
        f"- {qualitative_winner_text(summary_df, 'bertscore_f1')}",
        f"- {qualitative_winner_text(summary_df, faithfulness_label)}",
    ]
    if not hardest.empty:
        lines.extend([
            "- The hardest questions by mean ROUGE-L are listed below; these are usually broad opinion summaries or adversarial items where the model must avoid hallucination.",
            "",
            markdown_table(hardest),
        ])
    lines.extend([
        "",
        f"Manual faithfulness review file: `{review_path}`",
        "",
        "Faithful answers should stay grounded in retrieved Reddit evidence and explicitly refuse adversarial questions whose answer is absent from the corpus.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def translate_to_hindi(provider: ChatProvider, example: dict) -> tuple[str, list[dict] | None]:
    system_prompt = (
        "You are an expert English-to-Hindi translator for Reddit graduate-admissions text. "
        "Translate into natural Hindi. Preserve names, university names, acronyms such as PhD, MSCS, SOP, LoR, GPA, TA, and product names. "
        "Keep code-mixed Reddit slang understandable rather than over-literal. Output only the Hindi translation."
    )
    user_prompt = f"Translate this text to Hindi:\n{example['source']}"
    return provider.chat(system_prompt, user_prompt, max_tokens=256, temperature=0.0).text, None


def answer_cross_lingual_qa(provider: ChatProvider, example: dict, index: RagIndex, top_k: int) -> tuple[str, list[dict]]:
    context, retrieved = retrieved_context(index, example["retrieval_query_en"], top_k)
    user_prompt = (
        f"प्रश्न: {example['question_hi']}\n\n"
        f"English Reddit context:\n{context}\n\n"
        "उत्तर केवल हिंदी में लिखिए। यदि संदर्भ पर्याप्त नहीं है, तो यह बात साफ-साफ हिंदी में कहिए।"
    )
    return provider.chat(HINDI_QA_SYSTEM_PROMPT, user_prompt, max_tokens=256, temperature=0.0).text, retrieved


def summarize_in_hindi(provider: ChatProvider, example: dict, index: RagIndex, top_k: int) -> tuple[str, list[dict]]:
    context, retrieved = retrieved_context(index, example["retrieval_query_en"], top_k)
    user_prompt = (
        f"निर्देश: {example['prompt_hi']}\n\n"
        f"English Reddit context:\n{context}\n\n"
        "सार 2-4 वाक्यों में केवल हिंदी में लिखिए।"
    )
    return provider.chat(HINDI_SUMMARIZATION_SYSTEM_PROMPT, user_prompt, max_tokens=256, temperature=0.0).text, retrieved


def normalize_hinglish(provider: ChatProvider, example: dict) -> tuple[str, list[dict] | None]:
    user_prompt = f"इस वाक्य को साफ और स्वाभाविक हिंदी में बदलिए:\n{example['input_text']}"
    return provider.chat(HINGLISH_NORMALIZATION_SYSTEM_PROMPT, user_prompt, max_tokens=192, temperature=0.0).text, None


def hindi_task_specs() -> list[HindiTaskSpec]:
    return [
        HindiTaskSpec(
            task_name="translation",
            display_name="Translation",
            input_field="source",
            reference_field="reference_hindi",
            examples=load_hindi_translation_set(),
            generator=lambda provider, example, index, top_k: translate_to_hindi(provider, example),
            requires_rag=False,
        ),
        HindiTaskSpec(
            task_name="cross_lingual_qa",
            display_name="Cross-lingual QA",
            input_field="question_hi",
            reference_field="reference_hindi",
            examples=load_hindi_cross_lingual_qa_set(),
            generator=lambda provider, example, index, top_k: answer_cross_lingual_qa(provider, example, index, top_k),
            requires_rag=True,
        ),
        HindiTaskSpec(
            task_name="summarization",
            display_name="Summarisation",
            input_field="prompt_hi",
            reference_field="reference_hindi",
            examples=load_hindi_summarization_set(),
            generator=lambda provider, example, index, top_k: summarize_in_hindi(provider, example, index, top_k),
            requires_rag=True,
        ),
        HindiTaskSpec(
            task_name="code_mixed_normalization",
            display_name="Extra: Hinglish to Clean Hindi",
            input_field="input_text",
            reference_field="reference_hindi",
            examples=load_hindi_code_mixed_normalization_set(),
            generator=lambda provider, example, index, top_k: normalize_hinglish(provider, example),
            requires_rag=False,
        ),
    ]


def evaluate_hindi_suite(
    provider_name_list: list[str] | None = None,
    limit_per_task: int | None = None,
    top_k: int | None = None,
    skip_bertscore: bool = False,
    reports_dir: Path = PART2_REPORTS_DIR,
) -> dict:
    providers = build_providers(provider_name_list)
    specs = hindi_task_specs()
    index = RagIndex(RAG_INDEX_DIR)
    records: list[dict] = []

    for spec in specs:
        examples = spec["examples"][:limit_per_task] if limit_per_task else spec["examples"]
        for provider in providers:
            for example in examples:
                output, retrieved = spec["generator"](provider, example, index, top_k or settings.rag_top_k)
                records.append(
                    {
                        "result_id": f"{provider.provider_name}:{spec['task_name']}:{example['id']}",
                        "provider": provider.provider_name,
                        "model": provider.model,
                        "task": spec["task_name"],
                        "task_label": spec["display_name"],
                        "example_id": example["id"],
                        "tags": ",".join(example.get("tags", [])),
                        "input_text": example[spec["input_field"]],
                        "reference_hindi": example[spec["reference_field"]],
                        "output_hindi": output,
                        "chrf": chrf(output, example[spec["reference_field"]]),
                        "retrieved": retrieved,
                    }
                )

    if not skip_bertscore and records:
        scorer = BertScoreScorer(settings.hindi_bertscore_model)
        scores = scorer.score_pairs(
            [record["output_hindi"] for record in records],
            [record["reference_hindi"] for record in records],
        )
        for record, score in zip(records, scores):
            record["bertscore_f1"] = score
    else:
        for record in records:
            record["bertscore_f1"] = None

    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reports_dir / "hindi_suite_results.jsonl"
    write_jsonl(raw_path, records)
    flat_records = [{key: value for key, value in record.items() if key != "retrieved"} for record in records]
    result_df = pd.DataFrame(flat_records)
    result_df.to_csv(reports_dir / "hindi_suite_results.csv", index=False)

    review_path = reports_dir / "hindi_suite_manual_review.csv"
    manual_df = sync_manual_review_sheet(
        result_df[["result_id", "provider", "task", "task_label", "example_id", "tags", "input_text", "output_hindi"]].copy(),
        review_path,
        key_columns=["result_id"],
        manual_columns=["fluency_1_to_5", "adequacy_1_to_5", "notes"],
    )
    summary_rows = []
    for (provider_name, task_name), group in result_df.groupby(["provider", "task"]):
        manual_group = manual_df[(manual_df["provider"] == provider_name) & (manual_df["task"] == task_name)] if not manual_df.empty else pd.DataFrame()
        fluency = pd.to_numeric(manual_group.get("fluency_1_to_5"), errors="coerce") if not manual_group.empty else pd.Series(dtype=float)
        adequacy = pd.to_numeric(manual_group.get("adequacy_1_to_5"), errors="coerce") if not manual_group.empty else pd.Series(dtype=float)
        summary_rows.append(
            {
                "provider": provider_name,
                "model": group["model"].iloc[0],
                "task": task_name,
                "task_label": group["task_label"].iloc[0],
                "n": int(len(group)),
                "chrf": mean_or_zero(group["chrf"].tolist()),
                "multilingual_bertscore_f1": mean_or_zero([value for value in group["bertscore_f1"].tolist() if pd.notna(value)]),
                "manual_fluency_1_to_5": mean_or_zero(fluency.dropna().tolist()),
                "manual_adequacy_1_to_5": mean_or_zero(adequacy.dropna().tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "hindi_suite_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    overall_rows = []
    for provider_name, group in result_df.groupby("provider"):
        manual_group = manual_df[manual_df["provider"] == provider_name] if not manual_df.empty else pd.DataFrame()
        fluency = pd.to_numeric(manual_group.get("fluency_1_to_5"), errors="coerce") if not manual_group.empty else pd.Series(dtype=float)
        adequacy = pd.to_numeric(manual_group.get("adequacy_1_to_5"), errors="coerce") if not manual_group.empty else pd.Series(dtype=float)
        overall_rows.append(
            {
                "provider": provider_name,
                "model": group["model"].iloc[0],
                "n": int(len(group)),
                "chrf": mean_or_zero(group["chrf"].tolist()),
                "multilingual_bertscore_f1": mean_or_zero([value for value in group["bertscore_f1"].tolist() if pd.notna(value)]),
                "manual_fluency_1_to_5": mean_or_zero(fluency.dropna().tolist()),
                "manual_adequacy_1_to_5": mean_or_zero(adequacy.dropna().tolist()),
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    overall_path = reports_dir / "hindi_suite_overall_summary.csv"
    overall_df.to_csv(overall_path, index=False)

    edge_rows = []
    for (provider_name, task_name), group in result_df.groupby(["provider", "task"]):
        tag_set = sorted({tag for tags in group["tags"].tolist() for tag in str(tags).split(",") if tag})
        for tag in tag_set:
            subset = group[group["tags"].str.contains(tag, regex=False)]
            edge_rows.append(
                {
                    "provider": provider_name,
                    "task": task_name,
                    "tag": tag,
                    "n": int(len(subset)),
                    "chrf": mean_or_zero(subset["chrf"].tolist()),
                    "multilingual_bertscore_f1": mean_or_zero([value for value in subset["bertscore_f1"].tolist() if pd.notna(value)]),
                }
            )
    edge_df = pd.DataFrame(edge_rows)
    edge_path = reports_dir / "hindi_suite_edge_analysis.csv"
    edge_df.to_csv(edge_path, index=False)

    write_hindi_suite_report(result_df, summary_df, overall_df, edge_df, reports_dir / "hindi_suite_report.md", review_path)
    return {
        "raw": raw_path,
        "summary": summary_path,
        "overall_summary": overall_path,
        "manual_review": review_path,
        "edge_analysis": edge_path,
    }


def write_hindi_suite_report(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    path: Path,
    review_path: Path,
) -> None:
    hardest_tags = pd.DataFrame()
    if not edge_df.empty:
        hardest_tags = edge_df.groupby("tag", as_index=False).agg(mean_chrf=("chrf", "mean")).sort_values("mean_chrf", ascending=True).head(5)
    task_counts = result_df.groupby("task_label", as_index=False).agg(examples=("example_id", lambda values: len(set(values)))) if not result_df.empty else pd.DataFrame()
    lines = [
        "# Hindi Language Evaluation",
        "",
        "Chosen Indian language: Hindi.",
        "Implemented formats:",
        "- Translation",
        "- Cross-lingual QA",
        "- Summarisation",
        "- Extra experiment: Hinglish to clean Hindi normalization",
        "",
        "## Task Sizes",
        "",
        markdown_table(task_counts),
        "",
        "## Overall Provider Summary",
        "",
        markdown_table(overall_df),
        "",
        "## Per-Task Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Edge-case Analysis",
        "",
        "The suite deliberately includes code-mixed Hinglish, Reddit slang, named entities, admissions acronyms, funding talk, interviews, portals, and GradCafe-style anxiety language.",
    ]
    if not hardest_tags.empty:
        lines.extend(["", markdown_table(hardest_tags)])
    lines.extend([
        "",
        "## Qualitative Analysis",
        "",
        f"- {qualitative_winner_text(overall_df, 'chrf')}",
        f"- {qualitative_winner_text(overall_df, 'multilingual_bertscore_f1')}",
        "- Cross-lingual QA and summarisation are the most demanding formats because they require the model to read English Reddit evidence and produce a fluent Hindi answer.",
        "- The extra Hinglish-normalization experiment focuses on transforming code-mixed Reddit-style admissions text into cleaner Hindi while preserving acronyms and named entities.",
        "",
        f"Manual fluency/adequacy review file: `{review_path}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bias_note(
    provider_name_list: list[str] | None = None,
    top_k: int | None = None,
    reports_dir: Path = PART2_REPORTS_DIR,
) -> Path:
    probes = load_bias_probes()
    providers = build_providers(provider_name_list)
    index = RagIndex(RAG_INDEX_DIR)
    records: list[dict] = []
    for probe in probes:
        retrieved = index.search(probe["question"], top_k=top_k or settings.rag_top_k)
        context = format_context(retrieved, max_chars=6000)
        for provider in providers:
            user_prompt = (
                f"Bias probe: {probe['question']}\n\n"
                f"Expected careful behavior: {probe['expected_good_behavior']}\n\n"
                f"Retrieved corpus evidence:\n{context}\n\n"
                "Answer in 1-2 paragraphs. State whether the corpus appears biased, whether the model might smooth over the bias, and what evidence supports that conclusion."
            )
            response = provider.chat(
                "You audit bias in a Reddit RAG system. Be specific, evidence-grounded, and avoid stereotyping.",
                user_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            records.append(
                {
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "probe_id": probe["id"],
                    "axis": probe["axis"],
                    "question": probe["question"],
                    "expected_good_behavior": probe["expected_good_behavior"],
                    "answer": response.text,
                    "retrieved": retrieved_to_dicts(retrieved),
                }
            )
    write_jsonl(reports_dir / "bias_probe_results.jsonl", records)
    path = reports_dir / "bias_note.md"
    lines = [
        "# Bias Detection Note",
        "",
        "This note uses custom probes over the r/gradadmissions RAG corpus. The probes target prestige, international status, socioeconomic assumptions, GPA, and Reddit-demographic sampling bias.",
        "",
        "## Findings",
        "",
        "The corpus itself is likely biased by self-selection: users who post are disproportionately anxious, English-writing, Reddit-using applicants, and many posts center on competitive programs, CS/MSCS/PhD admissions, funding, and profile evaluation. The model can inherit those patterns when retrieved snippets overrepresent prestige, GPA anxiety, or expensive international study decisions.",
        "",
        "A good model response should not hide these biases by saying admissions are purely meritocratic. It should also avoid amplifying them by treating low GPA, non-prestige institutions, international status, or limited funding as personal deficits.",
        "",
        "## Probe Evidence",
    ]
    for record in records:
        lines.extend(
            [
                "",
                f"### {record['probe_id']} - {record['axis']} - {record['provider']}",
                "",
                f"Probe: {record['question']}",
                "",
                f"Expected behavior: {record['expected_good_behavior']}",
                "",
                record["answer"],
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def corpus_stats(session: Session) -> dict:
    subreddit_name = settings.subreddit_name
    return {
        "subreddit": subreddit_name,
        "posts": session.scalar(select(func.count()).select_from(Post).join(Subreddit).where(Subreddit.name == subreddit_name)) or 0,
        "comments": session.scalar(select(func.count()).select_from(Comment).join(Post).join(Subreddit).where(Subreddit.name == subreddit_name)) or 0,
        "users": session.scalar(select(func.count()).select_from(RedditUser)) or 0,
        "min_post_date": session.scalar(select(func.min(Post.created_utc)).join(Subreddit).where(Subreddit.name == subreddit_name)),
        "max_post_date": session.scalar(select(func.max(Post.created_utc)).join(Subreddit).where(Subreddit.name == subreddit_name)),
    }


def write_ethics_note(session: Session, reports_dir: Path = PART2_REPORTS_DIR) -> Path:
    stats = corpus_stats(session)
    min_date = stats["min_post_date"].date().isoformat() if stats["min_post_date"] else "unknown"
    max_date = stats["max_post_date"].date().isoformat() if stats["max_post_date"] else "unknown"
    path = reports_dir / "ethics_note.md"
    lines = [
        "# Ethics Note",
        "",
        f"Corpus: r/{stats['subreddit']} with {stats['posts']:,} posts, {stats['comments']:,} comments, and {stats['users']:,} stored user records from {min_date} to {max_date}.",
        "",
        "## Personal Information and Re-identification",
        "",
        "Even if usernames are treated as pseudonyms, admissions posts often contain combinations of GPA, university names, target programs, research fields, publications, visa status, timelines, and admit/reject outcomes. Those attributes can be enough to re-identify a real applicant when combined with posting history or external profiles. The RAG system therefore should avoid revealing usernames in answers, should cite content at the snippet level rather than profile users, and should not answer identity-seeking questions.",
        "",
        "## Right to be Forgotten",
        "",
        "A local archived RAG system can violate deletion expectations if a user deletes a Reddit post after it has already been ingested and embedded. Full compliance in production is difficult but the design should include deletion sync, tombstones for removed Reddit IDs, vector-index rebuilds or targeted vector deletion, audit logs, and a documented retention policy. This course implementation does not continuously poll Reddit for deletions, so it should be presented as a research prototype rather than a production system.",
        "",
        "## Practical Safeguards",
        "",
        "The assistant prompt instructs models to use only retrieved context, refuse unsupported or identity-seeking questions, avoid private identity inference, and summarize community-level patterns instead of profiling individual users. Stored API keys remain in `.env`, and generated reports should not include secrets.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_part2_report(reports_dir: Path = PART2_REPORTS_DIR) -> Path:
    path = reports_dir / "part2_report.md"
    lines = ["# NLP Project Part 2 Report", ""]
    for filename, title in [
        ("qa_report.md", "Conversation System"),
        ("hindi_suite_report.md", "Indian Language Translation Task"),
        ("bias_note.md", "Bias Detection Note"),
        ("ethics_note.md", "Ethics Note"),
    ]:
        section_path = reports_dir / filename
        lines.extend([f"## {title}", ""])
        if section_path.exists():
            content = section_path.read_text(encoding="utf-8").splitlines()
            lines.extend(line for line in content if not line.startswith("# "))
        else:
            lines.append(f"Pending: run the command that generates `{filename}`.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def dataframe_preview(records: Sequence[dict]) -> list[dict]:
    return [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records]
