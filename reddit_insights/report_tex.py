from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from reddit_insights.config import ARTIFACTS_DIR, PART2_EVAL_DIR, PART2_REPORTS_DIR, PROJECT_ROOT, RAG_INDEX_DIR, settings


FINAL_REPORT_TEX_PATH = PART2_REPORTS_DIR / "final_project_report.tex"
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def latex_escape(value: Any) -> str:
    text_value = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text_value = text_value.replace(old, new)
    return text_value


def tex_path(path: Path | str) -> str:
    raw = Path(path)
    try:
        display = raw.relative_to(PROJECT_ROOT)
    except ValueError:
        display = raw
    return rf"\texttt{{{latex_escape(display.as_posix())}}}"


def has_devanagari(text_value: Any) -> bool:
    return bool(DEVANAGARI_RE.search(str(text_value or "")))


def tex_text(value: Any) -> str:
    escaped = latex_escape(value)
    if has_devanagari(value):
        return rf"{{\hindifont {escaped}}}"
    return escaped


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def dataframe_to_latex(df: pd.DataFrame, columns: list[str] | None = None, float_digits: int = 4) -> str:
    if df.empty:
        return "No rows available."
    if columns is None:
        columns = list(df.columns)
    width = max(0.09, round(0.96 / max(len(columns), 1), 2))
    colspec = '|'.join([f"p{{{width}\\textwidth}}"] * len(columns))
    rows = [r"\begin{center}", rf"\small\begin{{tabular}}{{{colspec}}}", r"\hline"]
    header = " & ".join(latex_escape(column) for column in columns) + r" \\ \hline"
    rows.append(header)
    for record in df[columns].to_dict(orient="records"):
        values: list[str] = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                rendered = format_float(value, float_digits)
            else:
                rendered = str(value)
            values.append(tex_text(rendered))
        rows.append(" & ".join(values) + r" \\ \hline")
    rows.extend([r"\end{tabular}", r"\end{center}"])
    return "\n".join(rows)


def build_db_snapshot() -> tuple[dict[str, Any], pd.DataFrame]:
    engine = create_engine(settings.database_url)
    counts_query = text(
        """
        select
          (select count(*) from posts) as posts,
          (select count(*) from comments) as comments,
          (select count(*) from users) as users,
          (select count(*) from subreddits) as subreddits,
          (select count(*) from topics) as topics,
          (select min(created_utc) from posts) as min_post_date,
          (select max(created_utc) from posts) as max_post_date,
          (select min(created_utc) from comments) as min_comment_date,
          (select max(created_utc) from comments) as max_comment_date
        """
    )
    topics_query = text(
        """
        select topic_index, label, share_of_posts, topic_type, trend_score, persistence_score, dominant_stance
        from topics
        order by share_of_posts desc
        """
    )
    with engine.connect() as conn:
        counts = dict(conn.execute(counts_query).mappings().one())
        topics_df = pd.DataFrame(conn.execute(topics_query).mappings().all())
    return counts, topics_df


def example_block(title: str, entries: list[tuple[str, Any]]) -> str:
    lines = [rf"\textbf{{{latex_escape(title)}}}", r"\begin{quote}\small"]
    for label, value in entries:
        lines.append(rf"\textbf{{{latex_escape(label)}}} {tex_text(value)}\\")
    lines.append(r"\end{quote}")
    return "\n".join(lines)


def render_eval_set_section(title: str, path: Path, examples: list[dict[str, Any]], example_fields: list[str]) -> str:
    sample_count = len(examples)
    field_names = ", ".join(latex_escape(key) for key in examples[0].keys()) if examples else ""
    blocks = [
        rf"\subsection*{{{latex_escape(title)}}}",
        rf"Full reference set: {tex_path(path)}.",
        rf"Sample count: {sample_count}. Fields: {field_names}.",
    ]
    for idx, example in enumerate(examples[:2], start=1):
        entries = [(field.replace("_", " ").title() + ":", example.get(field, "")) for field in example_fields if field in example]
        blocks.append(example_block(f"Example {idx} ({example.get('id', 'sample')})", entries))
    return "\n\n".join(blocks)


def render_file_list(title: str, items: list[str]) -> str:
    lines = [rf"\subsection*{{{latex_escape(title)}}}", r"\begin{itemize}"]
    for item in items:
        lines.append(rf"\item {tex_path(item)}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def build_report_text() -> str:
    summary = read_json(ARTIFACTS_DIR / "summary.json")
    rag_metadata = read_json(RAG_INDEX_DIR / "metadata.json")
    meta_manifest = read_json(ARTIFACTS_DIR / "post_meta_manifest.json")
    counts, topics_df = build_db_snapshot()

    qa_eval = read_json(PART2_EVAL_DIR / "qa_eval_set.json")
    hi_translation = read_json(PART2_EVAL_DIR / "hindi_translation_eval_set.json")
    hi_cross = read_json(PART2_EVAL_DIR / "hindi_cross_lingual_qa_eval_set.json")
    hi_summary = read_json(PART2_EVAL_DIR / "hindi_summarization_eval_set.json")
    hi_norm = read_json(PART2_EVAL_DIR / "hindi_code_mixed_normalization_eval_set.json")

    qa_summary = pd.read_csv(PART2_REPORTS_DIR / "qa_summary.csv")
    qa_type_breakdown = pd.read_csv(PART2_REPORTS_DIR / "qa_type_breakdown.csv")
    hindi_overall = pd.read_csv(PART2_REPORTS_DIR / "hindi_suite_overall_summary.csv")
    hindi_summary = pd.read_csv(PART2_REPORTS_DIR / "hindi_suite_summary.csv")
    qa_manual = pd.read_csv(PART2_REPORTS_DIR / "qa_manual_faithfulness_review.csv")
    hindi_manual = pd.read_csv(PART2_REPORTS_DIR / "hindi_suite_manual_review.csv")

    topic_table = topics_df.copy()
    if not topic_table.empty:
        topic_table["share_of_posts"] = topic_table["share_of_posts"].map(lambda v: f"{100 * float(v):.2f}%")
        topic_table["trend_score"] = topic_table["trend_score"].map(lambda v: f"{float(v):.4f}")
        topic_table["persistence_score"] = topic_table["persistence_score"].map(lambda v: f"{float(v):.4f}")

    qa_summary_fmt = qa_summary.copy()
    if not qa_summary_fmt.empty:
        for column in ["rouge_l", "bertscore_f1", "faithfulness_manual_or_auto_pct"]:
            if column in qa_summary_fmt.columns:
                qa_summary_fmt[column] = qa_summary_fmt[column].map(lambda v: round(float(v), 4))
    qa_type_fmt = qa_type_breakdown.copy()
    if not qa_type_fmt.empty:
        for column in ["rouge_l", "bertscore_f1", "faithfulness_manual_or_auto_pct"]:
            if column in qa_type_fmt.columns:
                qa_type_fmt[column] = qa_type_fmt[column].map(lambda v: round(float(v), 4))
    hindi_overall_fmt = hindi_overall.copy()
    if not hindi_overall_fmt.empty:
        for column in ["chrf", "multilingual_bertscore_f1", "manual_fluency_1_to_5", "manual_adequacy_1_to_5"]:
            if column in hindi_overall_fmt.columns:
                hindi_overall_fmt[column] = hindi_overall_fmt[column].map(lambda v: round(float(v), 4))
    hindi_summary_fmt = hindi_summary.copy()
    if not hindi_summary_fmt.empty:
        for column in ["chrf", "multilingual_bertscore_f1", "manual_fluency_1_to_5", "manual_adequacy_1_to_5"]:
            if column in hindi_summary_fmt.columns:
                hindi_summary_fmt[column] = hindi_summary_fmt[column].map(lambda v: round(float(v), 4) if pd.notna(v) else "-")

    reviewed_faithfulness = int(qa_manual["faithful_manual"].notna().sum()) if "faithful_manual" in qa_manual.columns else 0
    reviewed_hindi = int(hindi_manual["fluency_1_to_5"].notna().sum()) if "fluency_1_to_5" in hindi_manual.columns else 0

    qa_examples = render_eval_set_section(
        "QA Reference Set",
        PART2_EVAL_DIR / "qa_eval_set.json",
        qa_eval,
        ["question", "reference_answer", "type"],
    )
    translation_examples = render_eval_set_section(
        "Hindi Translation Reference Set",
        PART2_EVAL_DIR / "hindi_translation_eval_set.json",
        hi_translation,
        ["source", "reference_hindi", "tags"],
    )
    cross_examples = render_eval_set_section(
        "Hindi Cross-lingual QA Reference Set",
        PART2_EVAL_DIR / "hindi_cross_lingual_qa_eval_set.json",
        hi_cross,
        ["question_hi", "retrieval_query_en", "reference_hindi", "type"],
    )
    summary_examples = render_eval_set_section(
        "Hindi Summarisation Reference Set",
        PART2_EVAL_DIR / "hindi_summarization_eval_set.json",
        hi_summary,
        ["prompt_hi", "retrieval_query_en", "reference_hindi", "tags"],
    )
    norm_examples = render_eval_set_section(
        "Hinglish Normalization Reference Set",
        PART2_EVAL_DIR / "hindi_code_mixed_normalization_eval_set.json",
        hi_norm,
        ["input_text", "reference_hindi", "tags"],
    )

    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M")

    return textwrap.dedent(
        rf"""
        % !TEX program = xelatex
        \documentclass[11pt]{{article}}
        \usepackage[a4paper,margin=1in]{{geometry}}
        \usepackage{{booktabs}}
        \usepackage{{longtable}}
        \usepackage{{array}}
        \usepackage{{hyperref}}
        \usepackage{{enumitem}}
        \usepackage{{fontspec}}
        \usepackage{{polyglossia}}
        \setdefaultlanguage{{english}}
        \setotherlanguage{{hindi}}
        \newfontfamily\hindifont[Script=Devanagari]{{Noto Serif Devanagari}}
        \setlength{{\parindent}}{{0pt}}
        \setlength{{\parskip}}{{0.7em}}
        \begin{{document}}

        {{\LARGE \textbf{{Reddit Insights Project Report}}}}\\
        r/{latex_escape(summary['subreddit'])} corpus, Part 1 + Part 2 unified report. Generated on {latex_escape(generated_on)}.

        \section*{{Project Overview}}
        This project builds an analysis and question-answering system over Reddit discussions from {tex_path('r/gradadmissions')} using Arctic Shift as the data source. Part 1 focuses on corpus construction, topic discovery, stance analysis, trend labelling, keyword mining, and an exploratory dashboard. Part 2 adds a persisted retrieval-augmented generation (RAG) layer, a hybrid corpus-analytics query engine, API-provider comparison, and Hindi evaluation across four task formats.

        The current corpus contains {counts['posts']} posts, {counts['comments']} comments, and {counts['users']} distinct stored users from {counts['min_post_date']} to {counts['max_post_date']}. The analysis pipeline materializes {counts['topics']} topics, a post-level topic artifact, a post-level meta-query cache, and a chunked RAG index over posts plus comments.

        \section*{{Data and Storage}}
        \subsection*{{Collection and protocol}}
        Data collection uses the Arctic Shift Reddit endpoint rather than the official Reddit API. The collection protocol is time-window based: submissions are pulled for a recent date range, de-duplicated by Reddit submission id, normalized, and upserted into SQLite. Comments are then fetched for the same subreddit, matched back to known submissions through the linked post id, de-duplicated by Reddit comment id, and stored against the corresponding post. This protocol preserves the original post/comment separation while avoiding the PRAW credential requirement.

        \subsection*{{Preprocessing}}
        Posts are normalized at ingestion time for whitespace and text cleanliness. Later analysis uses a combined post text field built as title plus selftext, because many admissions posts carry the real semantics only when both are read together. Removed or deleted strings are retained in storage for provenance, but later preprocessing and topic-summary interfaces filter them aggressively when they would reduce analytical value.

        \subsection*{{Storage design}}
        The main relational entities are: \texttt{{subreddits}}, \texttt{{users}}, \texttt{{posts}}, \texttt{{comments}}, \texttt{{topics}}, \texttt{{topic\_assignments}}, \texttt{{topic\_weekly\_metrics}}, \texttt{{comment\_stances}}, and \texttt{{topic\_user\_stances}}. The design keeps raw discourse, topic-level abstractions, time-series metrics, per-comment stance predictions, and per-user aggregated stances separate. This makes the system auditable: a dashboard table or Part 2 answer can be traced back to stored posts/comments instead of existing only as an LLM output.

        \begin{{center}}
        \begin{{tabular}}{{|p{{0.22\textwidth}}|p{{0.68\textwidth}}|}}
        \hline
        Table & Purpose \\ \hline
        \texttt{{subreddits}} & One row per tracked subreddit. \\ \hline
        \texttt{{users}} & Reddit authors with deletion flag. \\ \hline
        \texttt{{posts}} & Submission metadata, title, body, score, comment count, permalink, and source. \\ \hline
        \texttt{{comments}} & Comment body, score, timestamp, parent linkage, and author linkage. \\ \hline
        \texttt{{topics}} & Topic label, keyword list, corpus share, trend/persistence scores, side summaries, and dominant stance. \\ \hline
        \texttt{{topic\_assignments}} & One post-to-topic assignment with numeric weight. \\ \hline
        \texttt{{topic\_weekly\_metrics}} & Weekly post-count and share time series per topic. \\ \hline
        \texttt{{comment\_stances}} & Comment stance toward a topic viewpoint, with confidence and rationale string. \\ \hline
        \texttt{{topic\_user\_stances}} & Per-user aggregated stance inside a topic, with comment count and average confidence. \\ \hline
        \end{{tabular}}
        \end{{center}}

        Key artifact files are {tex_path('data/artifacts/summary.json')}, {tex_path('data/artifacts/posts_with_topics.parquet')}, {tex_path('data/artifacts/post_meta_tags.parquet')}, {tex_path('data/artifacts/post_meta_manifest.json')}, {tex_path('data/part2/rag_index/documents.jsonl')}, and {tex_path('data/part2/rag_index/metadata.json')}.

        \section*{{Part 1 Pipeline}}
        \subsection*{{Topic extraction}}
        Posts are embedded with {latex_escape(summary['models']['topic_embedding_model'])}. MiniBatchKMeans with $k={summary['topic_count']}$ clusters the normalized post embeddings into topic groups. Each post receives exactly one cluster assignment. The stored \texttt{{topic\_weight}} is the cosine-style alignment between the post embedding $e_i$ and its assigned normalized centroid $c_{{z_i}}$: $\text{{topic\_weight}}_i = e_i^\top c_{{z_i}}$. Higher values mean the post is more centrally representative of its assigned topic.

        The fixed topic count is a design choice for this course setting: it produces a stable, inspectable inventory for a medium-sized single-subreddit corpus. With a larger or multi-subreddit corpus, a hierarchical or density-based topic model would become more attractive, but the fixed $k$ setup makes the downstream dashboard and stance analysis easier to reason about.

        \subsection*{{Keyword mining}}
        Topic keywords are not taken from raw tf-idf alone. Candidate phrases come from two sources: spaCy named entities and noun chunks, and a c-TF-IDF style lexical ranking over all posts assigned to a topic. Candidate phrases are cleaned with POS-aware lemmatization, a stop/block list, phrase-length filters, and domain allow-lists so that weak tokens such as ``just'' or ``like'' are removed. Each candidate is then reranked by a blended score:
        \[
        0.35\,\text{{lexical}} + 0.30\,\text{{spaCy frequency}} + 0.35\,\text{{semantic centroid similarity}} + \text{{length/domain bonus}}.
        \]
        Redundant phrases are removed if they are near-substrings of already selected keywords. This is why the later keyword extraction quality is much better than the initial version that leaked filler words.

        \subsection*{{Topic labelling and viewpoint generation}}
        The local generator model {latex_escape(summary['models']['generation_model'])} proposes human-readable topic labels and candidate discussion claims. Candidate claims are then filtered for obvious quality issues such as being too generic, too emotional, or too tied to one narrow school example. Remaining claims are ranked by how well comments split into explicit support and opposition under an NLI stance formulation, plus keyword overlap and an optional balance bonus. This makes topic labels and discussion viewpoints more discourse-aware than a plain keyword list.

        \subsection*{{Trend vs persistent classification}}
        Each topic gets a weekly time series using the share of posts assigned to that topic in each week. The classifier computes:
        \[
        \text{{trend\_score}} = (\text{{recent mean}} - \text{{earlier mean}}) + \text{{slope}} \times \text{{total weeks}}
        \]
        and
        \[
        \text{{persistence\_score}} = \frac{{\text{{active weeks}}}}{{\text{{total weeks}}}} - \text{{std(topic share)}}.
        \]
        A topic is labelled trending if its recent share is materially above its earlier share with positive slope, or if the trend score itself exceeds a minimum proportion-aware threshold. Otherwise it is treated as persistent. The trend score therefore measures recent lift plus slope, while the persistence score rewards appearing in many weeks and penalizes burstiness.

        \subsection*{{Stance detection}}
        Stance is framed as textual entailment. For a topic viewpoint, each comment is paired with an ``agree'' hypothesis and a ``disagree'' hypothesis, and the model {latex_escape(summary['models']['stance_nli_model'])} scores both. Let $s$ be the support entailment score and $o$ the oppose entailment score. The directional confidence is $\max(s,o)$ and the gap is $|s-o|$. A comment is labelled support or oppose if the directional score crosses the configured threshold and the gap is large enough; otherwise it is neutral.

        The current thresholds are: strong explicit stance when directional score $\geq {settings.stance_confidence_threshold:.2f}$ and gap $\geq {settings.stance_label_margin:.2f}$, weak explicit stance when directional score $\geq {settings.stance_weak_confidence_threshold:.2f}$ and gap $\geq {settings.stance_label_margin * 1.5:.2f}$, and neutral otherwise. The stored per-comment confidence is this directional entailment score, not a calibrated probability of correctness. It should be read as model certainty under the NLI formulation.

        \subsection*{{Dominant stance, user grouping, and side summaries}}
        Topic-level dominant stance is decided only when the topic has at least 10 explicit support/oppose comments and the support-vs-oppose margin exceeds 0.05. Otherwise the topic is marked as insufficient evidence. User grouping then aggregates all comment-level stances made by the same user inside the topic into a majority label plus average confidence. The stored \texttt{{avg\_confidence}} for a user-topic pair is the arithmetic mean of that user's explicit comment confidences in the topic.

        Support-side and oppose-side summaries are generated from the highest-confidence, highest-score comments on each side. The summarizer sees a short list of deduplicated comments and produces a 2--4 sentence side summary, which is then displayed in the dashboard.

        Additional score meanings used throughout Part 1 are straightforward but important: \texttt{{share\_of\_posts}} is the fraction of all analysed posts assigned to a topic, \texttt{{comment\_count}} in the user-stance table is the number of that user's topic-linked comments contributing to the aggregation, and \texttt{{avg\_confidence}} is the mean of the explicit comment-level NLI stance confidences for that user in that topic.

        \section*{{Part 1 Results}}
        The current topic inventory is shown below. The topic table is the bridge between the raw corpus and all later drilldowns such as representative posts, stance-grouped comments, and topic-grounded Hindi summarisation.

        {dataframe_to_latex(topic_table, ['topic_index', 'label', 'share_of_posts', 'topic_type', 'trend_score', 'persistence_score', 'dominant_stance'], float_digits=4)}

        At the interface level, Part 1 exposes topic overview, representative posts with full bodies, agreement-side and disagreement-side user tables with representative comments, and topic-specific post/comment drilldowns. This makes the topic and stance layers inspectable rather than purely numeric.

        \section*{{Part 2 System}}
        \subsection*{{RAG corpus and retrieval}}
        The Part 2 RAG index is built over both posts and comments, not posts alone. Documents are chunked into {rag_metadata['chunk_words']} word windows with {rag_metadata['chunk_overlap_words']} word overlap so that long posts/comments remain retrievable without losing local coherence. The index currently stores {rag_metadata['document_count']} chunks with {rag_metadata['embedding_dim']}-dimensional embeddings from {latex_escape(rag_metadata['embedding_model'])}. Embeddings are normalized at index time, so retrieval reduces to cosine similarity through dot products between the normalized query vector and normalized stored chunk vectors.

        Query-time retrieval returns top-$k$ chunks, then assembles a bounded context string up to the configured character budget. This bounded context is passed to the API provider with a grounded-answer prompt. The RAG path is therefore deterministic up to embedding retrieval and provider generation: retrieval itself is not delegated to the provider.

        \subsection*{{Provider comparison}}
        Part 2 compares Groq {latex_escape(settings.groq_model)} and Gemini {latex_escape(settings.gemini_model)} because the assignment requires API inference providers rather than purely local generation. Groq was chosen as a fast, strong long-context baseline through an OpenAI-compatible API. Gemini was chosen as a second commercially deployed provider with a different instruction-following profile. Using two providers turns the evaluation into a genuine comparative study rather than a single-system demo.

        \subsection*{{Hybrid corpus-QA: RAG vs meta query}}
        The project does not force every question through the same RAG path. Instead, the query layer first routes the question into one of three modes:
        \begin{{itemize}}
        \item \textbf{{rag}} for narrative questions such as ``What do users usually say about PhD admissions criteria?''
        \item \textbf{{meta}} for corpus-level analytics such as counts, comparisons, and trends.
        \item \textbf{{meta\_then\_rag}} for mixed questions such as ``How many posts discuss funding, and what do users say about it?''
        \end{{itemize}}

        Routing is rule-based rather than model-based. Regex detectors look for count/comparison/trend markers (for example ``how many'', ``vs'', ``over time'') and a second detector marks qualitative follow-ups such as ``what do users say''. This produces a structured query spec containing intent, concepts, optional comparison targets, and whether a qualitative explanation is also required.

        \subsection*{{Meta-query execution}}
        Meta queries run over a post-level cache rather than chunk retrieval. The cache lives in {tex_path('data/artifacts/post_meta_tags.parquet')} and contains {meta_manifest['row_count']} post rows plus normalized region flags, institution/country hits, and reusable theme tags such as funding, interview, SOP, LoR, profile review, admit, reject, waitlist, and visa/international. When a question maps cleanly to one of these known tags, the system answers by deterministic boolean filtering over cached columns. This path gets high confidence because the count comes from explicit cached structure.

        If the concept is not covered by the cached tag dictionary, the system falls back to semantic estimation. It embeds the query with the same {latex_escape(meta_manifest['embedding_model'])} encoder used for the cache, scores all posts by similarity, derives an adaptive threshold from the top-20 similarity anchor minus 0.08 with a 0.34 floor, applies a lexical-overlap restraint so broad semantic matches still share some important tokens with the query, and caps the matched fraction at 18\% of the corpus by tightening the threshold if the match set becomes too large. This path is marked medium confidence because the count is a semantic estimate rather than an exact symbolic filter.

        Comparison queries report left count, right count, and overlap count, because a single post can genuinely mention both sides, such as US and European universities. Trend queries aggregate matched posts by time bucket and compare earlier vs later frequency. Mixed queries first compute the meta result, then pass a follow-up question into the RAG layer to summarize what the matched posts are actually saying. Here \texttt{{matched\_count}} means the number of posts selected by the final meta-query mask, \texttt{{overlap\_count}} means the number of posts satisfying both sides of a comparison, and the reported meta confidence indicates whether the answer came from explicit cached tags (high) or embedding-based semantic estimation (medium).

        \subsection*{{Translation and Hindi summarisation interfaces}}
        The Streamlit interface includes a translation playground that shows 10 sampled English posts and translates user-selected or pasted text into Hindi. The summarisation section is topic-grounded rather than random: the user selects a topic, the app samples clean posts from that topic plus representative topic comments, and the provider produces a Hindi summary of that topic discussion. This is aligned with the assignment requirement that the summary be of a topic discussion, not an arbitrary post batch.

        \section*{{Part 2 Evaluation}}
        \subsection*{{QA evaluation design}}
        The QA evaluation set has {len(qa_eval)} questions in three categories: factual community questions, opinion-summary questions, and adversarial absent-answer questions. Factual items test whether the system captures stable corpus facts. Opinion-summary items test synthesis across multiple retrieved chunks. Adversarial absent-answer items test whether the system admits missing evidence rather than fabricating an answer.

        QA is scored with ROUGE-L, BERTScore-F1, and faithfulness. ROUGE-L measures longest-common-subsequence lexical overlap with the reference answer. BERTScore-F1 uses contextual encoder similarity and is more tolerant of paraphrase. Faithfulness measures whether the answer stays supported by retrieved evidence. The project first computes an automatic faithfulness flag, then allows manual override in {tex_path('data/part2/reports/qa_manual_faithfulness_review.csv')}. {reviewed_faithfulness} of {len(qa_manual)} QA outputs currently have a manual faithfulness judgment.

        {dataframe_to_latex(qa_summary_fmt, list(qa_summary_fmt.columns), float_digits=4)}

        {dataframe_to_latex(qa_type_fmt, list(qa_type_fmt.columns), float_digits=4)}

        \subsection*{{Hindi evaluation design}}
        Hindi evaluation covers all three assignment-suggested formats plus one extra format. For the cross-lingual QA and summarisation reference sets, each example stores both the Hindi instruction and an English retrieval query field (\texttt{{retrieval\_query\_en}}). This is deliberate: the underlying index is English-centric, so evaluation isolates Hindi generation quality on top of a stable English retrieval target.

        Hindi evaluation covers all three assignment-suggested formats plus one extra format:
        \begin{{itemize}}
        \item Translation: English admissions text to Hindi ({len(hi_translation)} samples).
        \item Cross-lingual QA: Hindi questions answered from English Reddit evidence ({len(hi_cross)} samples).
        \item Summarisation: Hindi summaries of topic discussions retrieved from English evidence ({len(hi_summary)} samples).
        \item Extra experiment: Hinglish/code-mixed normalization into clean Hindi ({len(hi_norm)} samples).
        \end{{itemize}}
        All tasks meet or exceed the minimum 20-sample requirement. Hindi was chosen because it is a high-value Indian language for this domain and also allows meaningful code-mixed evaluation.

        The Hindi suite uses chrF, multilingual BERTScore-F1, and manual fluency/adequacy review. chrF is a character n-gram F-score and is especially useful when multiple valid Hindi phrasings differ morphologically or orthographically. Multilingual BERTScore compares contextual embeddings using {latex_escape(settings.hindi_bertscore_model)}. Manual fluency rates naturalness on a 1--5 scale, and manual adequacy rates how well the meaning of the reference/source is preserved on a 1--5 scale. {reviewed_hindi} of {len(hindi_manual)} Hindi outputs currently have manual review scores filled in.

        {dataframe_to_latex(hindi_overall_fmt, list(hindi_overall_fmt.columns), float_digits=4)}

        {dataframe_to_latex(hindi_summary_fmt, list(hindi_summary_fmt.columns), float_digits=4)}

        \subsection*{{Interpretation}}
        On QA, Gemini scores slightly higher on lexical and semantic overlap metrics, while Groq scores higher on final faithfulness after manual review. On the Hindi suite, Groq is substantially stronger both automatically and under manual review. The manual review sheets are important here because some provider failures are truncation or instruction-following failures that are not fully captured by a single automatic score.

        \section*{{Reference Sets and Result Files}}
        This section lists the full reference and result artifacts and shows only a couple of examples from each reference set. The complete datasets and provider outputs stay in their dedicated JSON/CSV/JSONL files.

        {qa_examples}

        {translation_examples}

        {cross_examples}

        {summary_examples}

        {norm_examples}

        \subsection*{{Main result files}}
        QA outputs are stored in {tex_path('data/part2/reports/qa_results.jsonl')} and {tex_path('data/part2/reports/qa_results.csv')}. Aggregate QA tables are in {tex_path('data/part2/reports/qa_summary.csv')} and {tex_path('data/part2/reports/qa_type_breakdown.csv')}. Hindi outputs are stored in {tex_path('data/part2/reports/hindi_suite_results.jsonl')} and {tex_path('data/part2/reports/hindi_suite_results.csv')}. Aggregate Hindi tables are in {tex_path('data/part2/reports/hindi_suite_overall_summary.csv')} and {tex_path('data/part2/reports/hindi_suite_summary.csv')}. Manual review sheets are {tex_path('data/part2/reports/qa_manual_faithfulness_review.csv')} and {tex_path('data/part2/reports/hindi_suite_manual_review.csv')}. Additional qualitative summaries are in {tex_path('data/part2/reports/hindi_suite_edge_analysis.csv')}, {tex_path('data/part2/reports/bias_note.md')}, and {tex_path('data/part2/reports/ethics_note.md')}. Bias probes are in {tex_path('data/part2/eval/bias_probes.json')} and the provider outputs are in {tex_path('data/part2/reports/bias_probe_results.jsonl')}.

        \section*{{Bias, Ethics, and Limitations}}
        This system works on public Reddit discourse, but public does not mean risk-free. Graduate-admissions posts often contain quasi-identifiers such as GPA, country, subfield, school list, application year, interview timing, and scholarship details. In combination, those can make a user easier to re-identify than a single post would suggest. For that reason, the dashboard and Part 2 prompts avoid exposing usernames in generated answers and focus on aggregated discussion patterns rather than person-level profiles.

        A second issue is archival tension. Arctic Shift captures discourse that users may later delete or regret. Even when collection is allowed for research or coursework, the ethical posture should remain conservative: retained text should not be used to profile individuals, infer identities, or make consequential claims about a real applicant. The project is framed as a course prototype for corpus analysis and QA over community discourse, not a deployment-ready advising product.

        The corpus itself is biased. Reddit users are self-selected, English-heavy, internet-literate, and disproportionately likely to post when anxious, uncertain, or seeking reassurance. Discussions also skew toward prestige discourse, GPA anxiety, international-student constraints, and financially constrained decision-making. That means the system captures what this subreddit talks about, not ground truth about admissions policy. A strong answer in this system is therefore ``what this corpus says'', not ``what universities actually do''.

        Model bias also matters. Topic labels compress diverse posts into a fixed inventory. NLI stance detection can mistake hedged disagreement for neutrality or over-read strongly worded emotional comments. Provider-side Hindi generation quality varies sharply, as the evaluation tables show. Meta-query counts are exact only when a concept maps cleanly to cached symbolic tags; otherwise they are semantic estimates with explicit lower confidence.

        The safeguards in the current design are modest but concrete: grounded prompts, evidence-first retrieval, no username disclosure in generated answers, manual review sheets for faithfulness and Hindi adequacy, and explicit confidence labelling for meta queries. These are appropriate safeguards for a course project, but not enough on their own for a production student-advising system.

        \section*{{Reproducibility and Artifact Map}}
        The project is meant to be regenerated from stored data and artifacts rather than edited by hand. The main Part 1 summary artifact is {tex_path('data/artifacts/summary.json')}. The topic-level post artifact is {tex_path('data/artifacts/posts_with_topics.parquet')}. The RAG index lives under {tex_path('data/part2/rag_index')}. The post-level corpus-analytics cache is {tex_path('data/artifacts/post_meta_tags.parquet')} with embeddings in {tex_path('data/artifacts/post_meta_embeddings.npy')}. The current unified LaTeX report is {tex_path(FINAL_REPORT_TEX_PATH)}.

        To regenerate the evaluation outputs, the relevant CLI commands are the existing Part 2 evaluation commands plus the LaTeX report generator command added for this report. The report is intentionally plain: it is meant for Overleaf compilation and direct course submission, not journal-style formatting.

        \end{{document}}
        """
    ).strip() + "\n"


def write_final_project_report_tex(output_path: Path = FINAL_REPORT_TEX_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_text(), encoding="utf-8")
    return output_path
