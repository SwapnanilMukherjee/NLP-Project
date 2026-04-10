from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from reddit_insights.config import ARTIFACTS_DIR, normalize_subreddit_name, settings
from reddit_insights.modern_models import DOMAIN_TERMS, ModernNLPStack
from reddit_insights.models import Comment, CommentStance, Post, Subreddit, Topic, TopicAssignment, TopicUserStance, TopicWeeklyMetric
from reddit_insights.preprocess import CUSTOM_STOPWORDS, combined_post_text, top_terms


MIN_EXPLICIT_STANCE_COMMENTS = 10
MIN_DOMINANT_STANCE_MARGIN = 0.05
KEYWORD_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]+")
KEYWORD_CLEAN_RE = re.compile(r"[^a-zA-Z0-9&+./'-]+")
KEYWORD_ALLOWLIST = {"ai", "cs", "gpa", "gre", "mscs", "phd", "ra", "ta", "uiuc", "usc", "uva", "nyu"}
KEYWORD_BLOCKLIST = CUSTOM_STOPWORDS.union({
    "advice",
    "anyone",
    "best",
    "better",
    "different",
    "good",
    "help",
    "interesting",
    "looking",
    "question",
    "questions",
    "school",
    "schools",
    "sure",
    "thank",
    "thanks",
    "use",
    "used",
    "using",
    "way",
    "ways",
})
KEYWORD_ENTITY_LABELS = {"EVENT", "FAC", "GPE", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}


@dataclass
class TopicResult:
    topic_index: int
    label: str
    keywords: list[str]
    share_of_posts: float
    topic_type: str
    active_weeks: int
    total_weeks: int
    trend_score: float
    persistence_score: float


def load_posts_dataframe(session: Session, subreddit_name: str | None = None) -> pd.DataFrame:
    stmt = select(
        Post.id,
        Post.title,
        Post.selftext,
        Post.created_utc,
        Post.score,
        Post.num_comments,
    ).order_by(Post.created_utc.asc())
    if subreddit_name:
        stmt = stmt.join(Subreddit, Post.subreddit_id == Subreddit.id).where(
            Subreddit.name == normalize_subreddit_name(subreddit_name)
        )
    rows = session.execute(stmt).all()
    data = [
        {
            "post_id": row.id,
            "title": row.title,
            "selftext": row.selftext,
            "text": combined_post_text(row.title, row.selftext),
            "created_utc": row.created_utc,
            "score": row.score,
            "num_comments": row.num_comments,
        }
        for row in rows
    ]
    return pd.DataFrame(data)


def cluster_posts(embeddings: np.ndarray, topic_count: int) -> tuple[MiniBatchKMeans, np.ndarray, np.ndarray, np.ndarray]:
    clusterer = MiniBatchKMeans(
        n_clusters=topic_count,
        random_state=42,
        batch_size=max(2048, settings.embedding_batch_size * 8),
        n_init="auto",
    )
    assignments = clusterer.fit_predict(embeddings)
    centroids = clusterer.cluster_centers_.astype(np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(centroid_norms, 1e-12, None)
    weights = (embeddings * centroids[assignments]).sum(axis=1)
    return clusterer, assignments, weights, centroids


@lru_cache(maxsize=1)
def spacy_pipeline():
    try:
        return spacy.load("en_core_web_sm", disable=["textcat"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required for keyword extraction. "
            "Run `python -m spacy download en_core_web_sm` inside the project venv."
        ) from exc


def keyword_terms(phrase: str) -> list[str]:
    return [token.lower() for token in KEYWORD_TOKEN_RE.findall(phrase)]


def normalized_keyword_token(token) -> str:
    if token.text.lower() in KEYWORD_ALLOWLIST:
        raw = token.text.lower()
    elif token.pos_ == "PROPN":
        raw = token.text.lower()
    else:
        raw = token.lemma_.lower()
    return KEYWORD_CLEAN_RE.sub("", raw).strip("'-./")


def normalized_span_phrase(span) -> str:
    tokens: list[str] = []
    for token in span:
        if token.is_space or token.is_punct or token.like_url:
            continue
        if not tokens and (token.is_stop or token.pos_ in {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}):
            continue
        value = normalized_keyword_token(token)
        if not value or value == "-pron-":
            continue
        tokens.append(value)

    while tokens and tokens[0] in KEYWORD_BLOCKLIST:
        tokens.pop(0)
    while tokens and tokens[-1] in KEYWORD_BLOCKLIST:
        tokens.pop()
    if not tokens or len(tokens) > 5:
        return ""
    return " ".join(tokens)


def is_keyword_candidate(phrase: str) -> bool:
    tokens = keyword_terms(phrase)
    if not tokens:
        return False
    if any(token.isdigit() for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in KEYWORD_ALLOWLIST:
            return True
        if len(token) < 4:
            return False
        if token in KEYWORD_BLOCKLIST:
            return False
        if token.endswith("ly"):
            return False
        return True
    if tokens[0] in KEYWORD_BLOCKLIST or tokens[-1] in KEYWORD_BLOCKLIST:
        return False
    if sum(token in DOMAIN_TERMS for token in tokens) == 0 and len(tokens) == 2 and any(token in KEYWORD_BLOCKLIST for token in tokens):
        return False
    return True


def keyword_score_bonus(phrase: str) -> float:
    tokens = keyword_terms(phrase)
    bonus = 0.0
    if len(tokens) >= 2:
        bonus += 0.08
    if len(tokens) >= 3:
        bonus += 0.04
    if any(token in DOMAIN_TERMS for token in tokens):
        bonus += 0.03
    return bonus


def spacy_keyword_candidates(texts: list[str], assignments: np.ndarray, topic_count: int) -> dict[int, Counter[str]]:
    counters = {topic_index: Counter() for topic_index in range(topic_count)}
    nlp = spacy_pipeline()
    docs = nlp.pipe(texts, batch_size=64)
    for topic_index, doc in zip(assignments.tolist(), docs):
        doc_counter: Counter[str] = Counter()
        for ent in doc.ents:
            if ent.label_ not in KEYWORD_ENTITY_LABELS:
                continue
            phrase = normalized_span_phrase(ent)
            if is_keyword_candidate(phrase):
                doc_counter[phrase] = max(doc_counter.get(phrase, 0.0), 2.0)
        for chunk in doc.noun_chunks:
            phrase = normalized_span_phrase(chunk)
            if is_keyword_candidate(phrase):
                doc_counter[phrase] = max(doc_counter.get(phrase, 0.0), 1.0)
        counters[int(topic_index)].update(doc_counter)
    return counters


def is_redundant_keyword(candidate: str, selected: list[str]) -> bool:
    candidate_tokens = set(keyword_terms(candidate))
    candidate_phrase = " ".join(keyword_terms(candidate))
    for existing in selected:
        existing_tokens = set(keyword_terms(existing))
        existing_phrase = " ".join(keyword_terms(existing))
        if candidate_phrase == existing_phrase:
            return True
        if candidate_phrase in existing_phrase or existing_phrase in candidate_phrase:
            return True
        if candidate_tokens and existing_tokens and (candidate_tokens.issubset(existing_tokens) or existing_tokens.issubset(candidate_tokens)):
            return True
    return False


def reranked_topic_keywords(
    nlp: ModernNLPStack,
    feature_names: np.ndarray,
    ctfidf_row: np.ndarray,
    centroid: np.ndarray,
    fallback_texts: list[str],
    spacy_counter: Counter[str],
) -> list[str]:
    lexical_scores_by_phrase: dict[str, float] = {}
    for idx in ctfidf_row.argsort()[::-1]:
        score = float(ctfidf_row[idx])
        if score <= 0:
            break
        phrase = str(feature_names[idx])
        if not is_keyword_candidate(phrase):
            continue
        lexical_scores_by_phrase[phrase] = max(lexical_scores_by_phrase.get(phrase, 0.0), score)
        if len(lexical_scores_by_phrase) >= 96:
            break

    spacy_scores_by_phrase: dict[str, float] = {}
    for phrase, score in spacy_counter.most_common(96):
        if is_keyword_candidate(phrase):
            spacy_scores_by_phrase[phrase] = float(score)

    if not lexical_scores_by_phrase and not spacy_scores_by_phrase:
        fallback_terms = [term for term, _ in top_terms(fallback_texts, limit=24) if is_keyword_candidate(term)]
        return fallback_terms[:8]

    phrases = list(dict.fromkeys([*spacy_scores_by_phrase.keys(), *lexical_scores_by_phrase.keys()]))
    lexical_scores = np.asarray([lexical_scores_by_phrase.get(phrase, 0.0) for phrase in phrases], dtype=np.float32)
    spacy_scores = np.asarray([spacy_scores_by_phrase.get(phrase, 0.0) for phrase in phrases], dtype=np.float32)

    lexical_max = float(lexical_scores.max()) if len(lexical_scores) else 1.0
    lexical_norm = lexical_scores / max(lexical_max, 1e-12)
    spacy_max = float(spacy_scores.max()) if len(spacy_scores) else 1.0
    spacy_norm = spacy_scores / max(spacy_max, 1e-12)

    phrase_embeddings = nlp.embed_texts(phrases)
    semantic_scores = np.clip(phrase_embeddings @ centroid, 0.0, None)
    if semantic_scores.max() > semantic_scores.min():
        semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
    else:
        semantic_norm = semantic_scores

    blended_scores = []
    for idx, phrase in enumerate(phrases):
        blended_scores.append(
            0.35 * float(lexical_norm[idx])
            + 0.30 * float(spacy_norm[idx])
            + 0.35 * float(semantic_norm[idx])
            + keyword_score_bonus(phrase)
        )

    selected: list[str] = []
    for idx in np.argsort(np.asarray(blended_scores))[::-1]:
        phrase = phrases[int(idx)]
        if is_redundant_keyword(phrase, selected):
            continue
        selected.append(phrase)
        if len(selected) >= 8:
            break

    if selected:
        return selected

    fallback_terms = [term for term, _ in top_terms(fallback_texts, limit=24) if is_keyword_candidate(term)]
    return fallback_terms[:8]


def extract_topic_keywords(
    texts: list[str],
    assignments: np.ndarray,
    topic_count: int,
    nlp: ModernNLPStack,
    centroids: np.ndarray,
) -> tuple[CountVectorizer, dict[int, list[str]]]:
    min_df = 2 if len(texts) < 5000 else 5
    vectorizer = CountVectorizer(
        stop_words=sorted(CUSTOM_STOPWORDS),
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    topic_term_matrix = np.zeros((topic_count, len(feature_names)), dtype=np.float64)

    for topic_index in range(topic_count):
        topic_rows = np.where(assignments == topic_index)[0]
        if len(topic_rows) == 0:
            continue
        topic_term_matrix[topic_index] = np.asarray(matrix[topic_rows].sum(axis=0)).ravel()

    cluster_frequency = (topic_term_matrix > 0).sum(axis=0)
    cluster_lengths = np.clip(topic_term_matrix.sum(axis=1, keepdims=True), 1.0, None)
    idf = np.log((1 + topic_count) / (1 + cluster_frequency)) + 1.0
    ctfidf = (topic_term_matrix / cluster_lengths) * idf
    spacy_candidates_by_topic = spacy_keyword_candidates(texts, assignments, topic_count)

    keywords_by_topic: dict[int, list[str]] = {}
    for topic_index in range(topic_count):
        topic_texts = [texts[idx] for idx in np.where(assignments == topic_index)[0]]
        keywords = reranked_topic_keywords(
            nlp=nlp,
            feature_names=feature_names,
            ctfidf_row=ctfidf[topic_index],
            centroid=centroids[topic_index],
            fallback_texts=topic_texts,
            spacy_counter=spacy_candidates_by_topic[topic_index],
        )
        keywords_by_topic[topic_index] = keywords[:8]
    return vectorizer, keywords_by_topic


def representative_posts(topic_df: pd.DataFrame, limit: int = 6) -> list[str]:
    ordered = topic_df.sort_values(
        ["topic_weight", "score", "num_comments"],
        ascending=[False, False, False],
    )
    snippets = []
    for row in ordered.itertuples(index=False):
        snippets.append(combined_post_text(row.title, row.selftext))
        if len(snippets) >= limit:
            break
    return snippets


def representative_comments(topic_comments: pd.DataFrame, limit: int = 8) -> list[str]:
    if topic_comments.empty:
        return []
    comment_df = topic_comments.copy()
    comment_df["body_length"] = comment_df["body"].str.len().fillna(0)
    comment_df = comment_df[comment_df["body_length"] >= 40]
    ordered = comment_df.sort_values(["score", "body_length"], ascending=[False, False])
    return ordered["body"].head(limit).tolist()


def summary_candidates(topic_comments: pd.DataFrame, stance: str, limit: int = 12) -> list[str]:
    stance_df = topic_comments[topic_comments["stance"] == stance].copy()
    if stance_df.empty:
        return []
    stance_df["body_length"] = stance_df["body"].str.len().fillna(0)
    ordered = stance_df.sort_values(["confidence", "score", "body_length"], ascending=[False, False, False])
    return ordered["body"].head(limit).tolist()


def dominant_stance_from_counts(stance_counts: Counter[str]) -> str:
    support_count = stance_counts.get("support", 0)
    oppose_count = stance_counts.get("oppose", 0)
    explicit_total = support_count + oppose_count
    if explicit_total < MIN_EXPLICIT_STANCE_COMMENTS:
        return "insufficient_evidence"
    if support_count == 0 and oppose_count == 0:
        return "insufficient_evidence"
    if abs(support_count - oppose_count) / explicit_total < MIN_DOMINANT_STANCE_MARGIN:
        return "insufficient_evidence"
    return "support" if support_count > oppose_count else "oppose"


def normalize_stance_against_dominant(raw_stance: str, dominant_stance: str) -> str:
    if raw_stance == "neutral":
        return "neutral"
    if dominant_stance == "support":
        return raw_stance
    if dominant_stance == "oppose":
        return "support" if raw_stance == "oppose" else "oppose"
    return raw_stance


def topic_type_from_series(topic_series: pd.Series) -> tuple[str, int, int, float, float]:
    values = topic_series.to_numpy(dtype=float)
    total_weeks = int(len(values))
    active_weeks = int((values > 0).sum())
    if total_weeks == 0:
        return "persistent", 0, 0, 0.0, 0.0

    mean_share = float(values.mean())
    std_share = float(values.std(ddof=0)) if total_weeks > 1 else 0.0
    recent_window = min(4, total_weeks)
    recent_mean = float(values[-recent_window:].mean())
    earlier_values = values[:-recent_window]
    earlier_mean = float(earlier_values.mean()) if len(earlier_values) else recent_mean
    slope = float(np.polyfit(np.arange(total_weeks), values, 1)[0]) if total_weeks > 1 else 0.0
    trend_score = (recent_mean - earlier_mean) + slope * total_weeks
    persistence_score = (active_weeks / total_weeks) - std_share

    topic_type = "persistent"
    if recent_mean > max(0.01, earlier_mean * 1.25) and slope > 0:
        topic_type = "trending"
    elif trend_score > max(0.02, mean_share * 0.35):
        topic_type = "trending"
    return topic_type, active_weeks, total_weeks, trend_score, persistence_score


def run_analysis(session: Session, topic_count: int, subreddit_name: str | None = None) -> dict:
    normalized_subreddit = normalize_subreddit_name(subreddit_name) if subreddit_name else None
    df = load_posts_dataframe(session, normalized_subreddit)
    if len(df) < topic_count:
        scope = f" for r/{normalized_subreddit}" if normalized_subreddit else ""
        raise RuntimeError(f"Need at least {topic_count} posts{scope} to build {topic_count} topics.")

    nlp = ModernNLPStack()
    embeddings = nlp.embed_texts(df["text"].tolist())
    clusterer, assignments, weights, centroids = cluster_posts(embeddings, topic_count)
    keyword_vectorizer, keywords_by_topic = extract_topic_keywords(
        df["text"].tolist(),
        assignments,
        topic_count,
        nlp,
        centroids,
    )

    df["topic_index"] = assignments
    df["topic_weight"] = weights
    df["week_start"] = pd.to_datetime(df["created_utc"]).dt.to_period("W").dt.start_time

    comment_rows = session.execute(
        select(Comment.id, Comment.post_id, Comment.author_id, Comment.body, Comment.score).where(func.length(Comment.body) > 0)
    ).all()
    comments_df = pd.DataFrame(comment_rows, columns=["comment_id", "post_id", "author_id", "body", "score"])
    if not comments_df.empty:
        comments_df = comments_df.merge(df[["post_id", "topic_index"]], on="post_id", how="left").dropna().reset_index(drop=True)

    session.execute(delete(CommentStance))
    session.execute(delete(TopicUserStance))
    session.execute(delete(TopicWeeklyMetric))
    session.execute(delete(TopicAssignment))
    session.execute(delete(Topic))
    session.flush()

    topic_records: list[TopicResult] = []
    topic_id_by_index: dict[int, int] = {}
    weekly_totals = df.groupby("week_start").size().rename("total_posts")

    for topic_index in range(topic_count):
        topic_df = df[df["topic_index"] == topic_index].copy()
        topic_comments = comments_df[comments_df["topic_index"] == topic_index].copy() if not comments_df.empty else pd.DataFrame()
        keywords = keywords_by_topic.get(topic_index, [])
        topic_profile = nlp.build_topic_profile(
            keywords=keywords,
            representative_posts=representative_posts(topic_df),
            representative_comments=representative_comments(topic_comments),
        )

        share = float(len(topic_df) / len(df))
        weekly_counts = topic_df.groupby("week_start").size().rename("post_count")
        weekly = weekly_totals.to_frame().join(weekly_counts, how="left").fillna(0)
        weekly["topic_share"] = weekly["post_count"] / weekly["total_posts"]
        topic_type, active_weeks, total_weeks, trend_score, persistence_score = topic_type_from_series(weekly["topic_share"])

        record = Topic(
            topic_index=topic_index,
            label=topic_profile.label,
            keywords=json.dumps(keywords),
            share_of_posts=share,
            topic_type=topic_type,
            active_weeks=active_weeks,
            total_weeks=total_weeks,
            trend_score=trend_score,
            persistence_score=persistence_score,
            support_summary="No clear arguments detected.",
            oppose_summary="No clear arguments detected.",
            dominant_stance="insufficient_evidence",
        )
        session.add(record)
        session.flush()
        topic_id_by_index[topic_index] = record.id

        topic_records.append(
            TopicResult(
                topic_index=topic_index,
                label=topic_profile.label,
                keywords=keywords,
                share_of_posts=share,
                topic_type=topic_type,
                active_weeks=active_weeks,
                total_weeks=total_weeks,
                trend_score=trend_score,
                persistence_score=persistence_score,
            )
        )

        for _, row in weekly.reset_index().iterrows():
            session.add(
                TopicWeeklyMetric(
                    topic_id=record.id,
                    week_start=pd.Timestamp(row["week_start"]).to_pydatetime(),
                    post_count=int(row["post_count"]),
                    topic_share=float(row["topic_share"]),
                )
            )

        if topic_comments.empty:
            continue

        predictions = nlp.classify_comment_stances(topic_comments["body"].tolist(), topic_profile.dominant_viewpoint)
        topic_comments = topic_comments.assign(
            raw_stance=[item[0] for item in predictions],
            confidence=[item[1] for item in predictions],
            rationale=[item[2] for item in predictions],
        )

        stance_counts = Counter(topic_comments["raw_stance"])
        dominant_stance = dominant_stance_from_counts(stance_counts)
        topic_comments = topic_comments.assign(
            stance=topic_comments["raw_stance"].apply(
                lambda raw_stance: normalize_stance_against_dominant(raw_stance, dominant_stance)
            )
        )

        record.dominant_stance = dominant_stance
        record.support_summary = nlp.summarize_side(record.label, "agreement-side", summary_candidates(topic_comments, "support"))
        record.oppose_summary = nlp.summarize_side(record.label, "disagreement-side", summary_candidates(topic_comments, "oppose"))

        for row in topic_comments.itertuples(index=False):
            session.add(
                CommentStance(
                    comment_id=int(row.comment_id),
                    topic_id=record.id,
                    stance=row.stance,
                    confidence=float(row.confidence),
                    rationale=row.rationale,
                )
            )

        topic_user_comments = topic_comments.dropna(subset=["author_id"])
        for author_id, group in topic_user_comments.groupby("author_id"):
            grouped_counts = Counter(group["stance"])
            if grouped_counts["support"] > grouped_counts["oppose"]:
                user_stance = "support"
            elif grouped_counts["oppose"] > grouped_counts["support"]:
                user_stance = "oppose"
            else:
                user_stance = "neutral"
            session.add(
                TopicUserStance(
                    user_id=int(author_id),
                    topic_id=record.id,
                    stance=user_stance,
                    comment_count=int(len(group)),
                    avg_confidence=float(group["confidence"].mean()),
                )
            )

    for row in df.itertuples(index=False):
        session.add(
            TopicAssignment(
                post_id=int(row.post_id),
                topic_id=topic_id_by_index[int(row.topic_index)],
                weight=float(row.topic_weight),
            )
        )

    session.commit()

    artifact_base = Path(ARTIFACTS_DIR)
    joblib.dump(
        {
            "model_type": "embedding_kmeans",
            "embedding_model": settings.topic_embedding_model,
            "clusterer": clusterer,
            "centroids": centroids,
        },
        artifact_base / "topic_model.joblib",
    )
    joblib.dump(keyword_vectorizer, artifact_base / "topic_keyword_vectorizer.joblib")
    df.to_parquet(artifact_base / "posts_with_topics.parquet", index=False)

    summary = {
        "post_count": int(len(df)),
        "topic_count": int(topic_count),
        "subreddit": normalized_subreddit,
        "top_terms": top_terms(df["text"].tolist()),
        "methods": {
            "topic_extraction": "SentenceTransformer embeddings + MiniBatchKMeans clustering + spaCy noun-chunk/entity candidates + phrase-filtered c-TF-IDF keywording with embedding reranking",
            "trend_labeling": "weekly topic-share time-series with recency lift and slope heuristics",
            "stance_detection": "transformer NLI with agree/disagree hypothesis templates over comment/claim pairs",
            "summarization": "instruction-tuned causal LM summaries over stance-grouped comments",
            "user_grouping": "per-topic majority vote over model-based comment stances",
        },
        "models": {
            "topic_embedding_model": settings.topic_embedding_model,
            "stance_nli_model": settings.stance_nli_model,
            "generation_model": settings.generation_model,
        },
    }
    (artifact_base / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
