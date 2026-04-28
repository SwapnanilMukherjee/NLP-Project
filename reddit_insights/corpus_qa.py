from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from reddit_insights.config import ARTIFACTS_DIR, settings
from reddit_insights.db import SessionLocal
from reddit_insights.llm_providers import ChatProvider
from reddit_insights.rag import RAGAnswer, RagIndex, _load_embedder, answer_question


POST_META_TAGS_PATH = ARTIFACTS_DIR / "post_meta_tags.parquet"
POST_META_EMBEDDINGS_PATH = ARTIFACTS_DIR / "post_meta_embeddings.npy"
POST_META_MANIFEST_PATH = ARTIFACTS_DIR / "post_meta_manifest.json"
POSTS_WITH_TOPICS_PATH = ARTIFACTS_DIR / "posts_with_topics.parquet"

COUNT_RE = re.compile(r"\b(how many|number of|count of|count|how much|what share|what percentage|percentage of|proportion of)\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(vs\.?|versus|compared to|compare)\b", re.IGNORECASE)
TREND_RE = re.compile(r"\b(trend|over time|increase|increased|decrease|decreased|more common|less common|growing|rise|fall)\b", re.IGNORECASE)
QUAL_RE = re.compile(
    r"\b(what do users(?: [a-z]+){0,3} say|what do people(?: [a-z]+){0,3} say|what do users(?: [a-z]+){0,3} think|how do users(?: [a-z]+){0,3} discuss|why do users(?: [a-z]+){0,3}|what are users saying)\b",
    re.IGNORECASE,
)
METRIC_RE = re.compile(r"\b(posts?|comments?|users?)\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+&'./-]+")


THEME_DEFINITIONS: dict[str, dict[str, Any]] = {
    "funding": {
        "aliases": [
            "funding",
            "funded",
            "unfunded",
            "financial aid",
            "fellowship",
            "stipend",
            "fully funded",
            "partially funded",
        ],
        "description": "posts about graduate-admissions funding, scholarships, fellowships, stipends, or unfunded offers",
    },
    "scholarship": {
        "aliases": ["scholarship", "scholarships", "grant", "grants", "merit aid"],
        "description": "posts about scholarships, grants, and merit aid",
    },
    "assistantship": {
        "aliases": ["assistantship", "assistantships", "ta", "ra", "teaching assistantship", "research assistantship"],
        "description": "posts about TA or RA support and assistantships",
    },
    "tuition_cost": {
        "aliases": ["tuition", "cost", "costly", "expensive", "debt", "loan", "afford", "affordability", "cost of attendance"],
        "description": "posts about tuition, debt, affordability, and program cost",
    },
    "visa_international": {
        "aliases": ["visa", "international student", "international applicant", "f1", "f-1", "foreign applicant", "credential evaluation"],
        "description": "posts about visas, international applicants, and foreign-credential evaluation",
    },
    "interview": {
        "aliases": ["interview", "interviews", "zoom interview", "faculty interview", "pi interview"],
        "description": "posts about graduate-admissions interviews",
    },
    "profile_review": {
        "aliases": ["profile review", "chance me", "roast my cv", "roast my sop", "my profile", "profile eval"],
        "description": "posts asking for profile review or admissions chances",
    },
    "sop": {
        "aliases": ["sop", "statement of purpose", "personal statement", "motivation letter"],
        "description": "posts about statements of purpose or personal statements",
    },
    "lor": {
        "aliases": ["lor", "recommendation letter", "recommendation letters", "letter of recommendation", "letters of recommendation", "recommender"],
        "description": "posts about recommendation letters",
    },
    "admit": {
        "aliases": ["admit", "admission offer", "accepted", "acceptance", "offer letter"],
        "description": "posts about admits and acceptances",
    },
    "reject": {
        "aliases": ["reject", "rejection", "rejected", "deny", "denied"],
        "description": "posts about rejections",
    },
    "waitlist": {
        "aliases": ["waitlist", "waitlisted", "waiting list"],
        "description": "posts about waitlists",
    },
}


COUNTRY_ALIASES: dict[str, list[str]] = {
    "united_states": ["united states", "usa", "u.s.", "american universities", "american university"],
    "united_kingdom": ["united kingdom", "uk", "u.k.", "britain", "british", "england", "scotland"],
    "germany": ["germany", "german", "tum", "rwth", "tu munich", "technical university of munich"],
    "netherlands": ["netherlands", "dutch", "tu delft", "uva", "university of amsterdam", "amsterdam university"],
    "france": ["france", "french", "sorbonne", "ecole polytechnique"],
    "switzerland": ["switzerland", "swiss", "eth zurich", "epfl"],
    "belgium": ["belgium", "belgian", "ku leuven"],
    "sweden": ["sweden", "swedish", "kth"],
    "italy": ["italy", "italian", "politecnico di milano"],
    "spain": ["spain", "spanish", "barcelona", "madrid"],
    "canada": ["canada", "canadian", "waterloo", "ubc", "mcgill", "toronto"],
    "india": ["india", "indian", "iit", "nit"],
    "china": ["china", "chinese", "tsinghua", "peking university"],
    "singapore": ["singapore", "nus", "ntu singapore"],
    "japan": ["japan", "japanese", "tokyo university", "todai"],
    "south_korea": ["south korea", "korea", "kaist", "postech"],
    "hong_kong": ["hong kong", "hku", "hkust", "cuhk"],
}


INSTITUTION_REGION_ALIASES: dict[str, list[str]] = {
    "us": [
        "mit", "stanford", "harvard", "berkeley", "uc berkeley", "cmu", "carnegie mellon", "cornell",
        "columbia", "princeton", "yale", "nyu", "ucla", "ucsd", "usc", "uiuc", "purdue", "georgia tech",
        "umich", "michigan", "ut austin", "texas austin", "northwestern", "duke", "johns hopkins", "brown",
        "rice", "asu", "rutgers", "umass", "wisconsin", "virginia tech", "tamu", "texas a&m", "caltech",
    ],
    "europe": [
        "oxford", "cambridge", "imperial", "ucl", "edinburgh", "manchester", "eth zurich", "epfl",
        "tu delft", "rwth", "tum", "technical university of munich", "kth", "ku leuven", "sorbonne",
        "politecnico di milano", "university of amsterdam", "uva", "eindhoven", "erasmus", "heidelberg",
    ],
    "canada": ["waterloo", "ubc", "university of british columbia", "mcgill", "uoft", "university of toronto"],
    "asia": ["nus", "ntu", "kaist", "postech", "tsinghua", "peking university", "hku", "hkust", "cuhk", "todai"],
}


CONCEPT_ALIASES: dict[str, list[str]] = {
    "region_us": ["us universities", "u.s. universities", "american universities", "universities in the us", "universities in the usa", "admission to us universities", "admission to american universities"],
    "region_europe": ["european universities", "universities in europe", "admission to european universities", "europe universities", "eu universities"],
    "region_canada": ["canadian universities", "universities in canada", "admission to canadian universities"],
    "region_asia": ["asian universities", "universities in asia", "admission to asian universities"],
}
for theme_key, theme_info in THEME_DEFINITIONS.items():
    CONCEPT_ALIASES[theme_key] = list(theme_info["aliases"])


REGION_COUNTRIES: dict[str, set[str]] = {
    "us": {"united_states"},
    "europe": {"united_kingdom", "germany", "netherlands", "france", "switzerland", "belgium", "sweden", "italy", "spain"},
    "canada": {"canada"},
    "asia": {"india", "china", "singapore", "japan", "south_korea", "hong_kong"},
}


@dataclass(frozen=True)
class MetaExamplePost:
    post_id: int
    title: str
    selftext: str
    created_utc: str
    score: int
    num_comments: int
    topic_index: int | None
    topic_weight: float | None
    match_score: float


@dataclass(frozen=True)
class MetaQueryResult:
    question: str
    intent: str
    answer: str
    method: str
    confidence: str
    matched_count: int
    total_posts: int
    examples: list[MetaExamplePost]
    overlap_count: int | None = None
    debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class HybridAnswer:
    mode: str
    answer: str
    meta: MetaQueryResult | None = None
    rag: RAGAnswer | None = None


@dataclass(frozen=True)
class MetaQuerySpec:
    intent: str
    subject: str
    concepts: list[str]
    compare_concepts: list[str]
    wants_qualitative: bool
    metric: str | None = None


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("\u2019", "'")
    return WHITESPACE_RE.sub(" ", lowered).strip()


def _contains_alias(text: str, alias: str) -> bool:
    normalized_alias = _normalize_text(alias)
    return normalized_alias in text


def _json_dumps(values: list[str]) -> str:
    return json.dumps(sorted(dict.fromkeys(values)), ensure_ascii=False)


def _post_embedding_text(df: pd.DataFrame) -> list[str]:
    return [f"Title: {title}\nText: {text}" for title, text in zip(df["title"].fillna(""), df["text"].fillna(""), strict=False)]


def _load_source_posts() -> pd.DataFrame:
    if POSTS_WITH_TOPICS_PATH.exists():
        df = pd.read_parquet(POSTS_WITH_TOPICS_PATH)
    else:
        from reddit_insights.analysis import load_posts_dataframe

        with SessionLocal() as session:
            df = load_posts_dataframe(session, settings.subreddit_name)
        df["topic_index"] = pd.NA
        df["topic_weight"] = pd.NA
        df["week_start"] = pd.to_datetime(df["created_utc"]).dt.to_period("W").dt.start_time
    if "week_start" not in df.columns:
        df["week_start"] = pd.to_datetime(df["created_utc"]).dt.to_period("W").dt.start_time
    return df.copy()


def build_post_meta_cache(force: bool = False) -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if POST_META_TAGS_PATH.exists() and POST_META_EMBEDDINGS_PATH.exists() and POST_META_MANIFEST_PATH.exists() and not force:
        manifest = json.loads(POST_META_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["status"] = "exists"
        return manifest

    df = _load_source_posts()
    if df.empty:
        raise RuntimeError("No posts available to build the corpus meta cache.")

    normalized_texts = df["text"].fillna("").map(_normalize_text).tolist()
    country_hits: list[list[str]] = []
    institution_hits: list[list[str]] = []
    region_hits: list[list[str]] = []

    tag_columns: dict[str, list[bool]] = {f"tag_{key}": [] for key in THEME_DEFINITIONS}
    region_columns: dict[str, list[bool]] = {f"region_{key}": [] for key in REGION_COUNTRIES}

    for text in normalized_texts:
        post_country_hits: list[str] = []
        post_institution_hits: list[str] = []
        post_region_hits: set[str] = set()

        for country_key, aliases in COUNTRY_ALIASES.items():
            if any(_contains_alias(text, alias) for alias in aliases):
                post_country_hits.append(country_key)
        for region_key, countries in REGION_COUNTRIES.items():
            if any(country in countries for country in post_country_hits):
                post_region_hits.add(region_key)
        for region_key, aliases in INSTITUTION_REGION_ALIASES.items():
            for alias in aliases:
                if _contains_alias(text, alias):
                    post_institution_hits.append(alias)
                    post_region_hits.add(region_key)
        if "europe" in text or "european universit" in text or "universities in europe" in text:
            post_region_hits.add("europe")
        if "united states" in text or "usa" in text or "american universit" in text or "universities in the us" in text:
            post_region_hits.add("us")
        if "canadian universit" in text or "universities in canada" in text:
            post_region_hits.add("canada")
        if "asian universit" in text or "universities in asia" in text:
            post_region_hits.add("asia")

        country_hits.append(sorted(dict.fromkeys(post_country_hits)))
        institution_hits.append(sorted(dict.fromkeys(post_institution_hits)))
        region_hits.append(sorted(post_region_hits))

        for theme_key, theme_info in THEME_DEFINITIONS.items():
            tag_columns[f"tag_{theme_key}"].append(any(_contains_alias(text, alias) for alias in theme_info["aliases"]))
        for region_key in REGION_COUNTRIES:
            region_columns[f"region_{region_key}"].append(region_key in post_region_hits)

    cache_df = df[["post_id", "title", "selftext", "text", "created_utc", "score", "num_comments", "topic_index", "topic_weight", "week_start"]].copy()
    cache_df["country_hits_json"] = [_json_dumps(values) for values in country_hits]
    cache_df["institution_hits_json"] = [_json_dumps(values) for values in institution_hits]
    cache_df["region_hits_json"] = [_json_dumps(values) for values in region_hits]
    for key, values in tag_columns.items():
        cache_df[key] = values
    for key, values in region_columns.items():
        cache_df[key] = values
    cache_df.to_parquet(POST_META_TAGS_PATH, index=False)

    embedder = _load_embedder(settings.rag_embedding_model)
    embeddings = embedder.encode(
        _post_embedding_text(cache_df),
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(cache_df) >= settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(POST_META_EMBEDDINGS_PATH, embeddings)

    manifest = {
        "status": "created",
        "row_count": int(len(cache_df)),
        "embedding_model": settings.rag_embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "tag_columns": sorted(tag_columns),
        "region_columns": sorted(region_columns),
        "source": str(POSTS_WITH_TOPICS_PATH if POSTS_WITH_TOPICS_PATH.exists() else "database"),
    }
    POST_META_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class MetaCorpusIndex:
    def __init__(self, force_rebuild: bool = False) -> None:
        build_post_meta_cache(force=force_rebuild)
        self.df = pd.read_parquet(POST_META_TAGS_PATH)
        self.embeddings = np.load(POST_META_EMBEDDINGS_PATH, mmap_mode="r")
        self.manifest = json.loads(POST_META_MANIFEST_PATH.read_text(encoding="utf-8"))
        if len(self.df) != int(self.embeddings.shape[0]):
            raise RuntimeError("Meta corpus cache is inconsistent: tag rows and embedding rows differ.")

    @property
    def total_posts(self) -> int:
        return int(len(self.df))

    @property
    def embedder(self) -> SentenceTransformer:
        return _load_embedder(self.manifest.get("embedding_model", settings.rag_embedding_model))

    def semantic_scores(self, query: str) -> np.ndarray:
        query_embedding = self.embedder.encode([query], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        return np.asarray(self.embeddings @ query_embedding)



def _extract_known_concepts(question: str) -> list[str]:
    lowered = _normalize_text(question)
    matches: list[str] = []
    for concept_key, aliases in CONCEPT_ALIASES.items():
        if any(_contains_alias(lowered, alias) for alias in aliases):
            matches.append(concept_key)
    return list(dict.fromkeys(matches))


def _extract_compare_concepts(question: str) -> list[str]:
    lowered = _normalize_text(question)
    parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompared to\b", lowered)
    if len(parts) < 2:
        return []
    matched: list[str] = []
    for part in parts[:2]:
        part_matches = _extract_known_concepts(part)
        if part_matches:
            matched.append(part_matches[0])
        else:
            if "europe" in part:
                matched.append("region_europe")
            elif "us" in part or "usa" in part or "american" in part:
                matched.append("region_us")
            elif "canada" in part or "canadian" in part:
                matched.append("region_canada")
            elif "asia" in part or "asian" in part:
                matched.append("region_asia")
    return list(dict.fromkeys(matched))


def _extract_subject(question: str) -> str:
    lowered = _normalize_text(question)
    lowered = re.sub(
        r"(?:,?\s+and\s+)?(what do users(?: [a-z]+){0,3} say|what do people(?: [a-z]+){0,3} say|what do users(?: [a-z]+){0,3} think|how do users(?: [a-z]+){0,3} discuss|why do users(?: [a-z]+){0,3}|what are users saying).*$",
        "",
        lowered,
    )
    lowered = re.sub(r"^(how many|number of|count of|count|what share of|what percentage of|percentage of|proportion of)\s+", "", lowered)
    lowered = re.sub(r"\b(posts?|comments?|users?)\b", "", lowered)
    lowered = re.sub(r"\b(are about|about|discuss|discussing|mention|mentions|related to|talk about|talking about|focused on)\b", " ", lowered)
    lowered = re.sub(r"\b(vs\.?|versus|compared to|compare|over time|increase|increased|decrease|decreased|trend)\b", " ", lowered)
    lowered = re.sub(r"\b(and|the|a|an)\b", " ", lowered)
    return WHITESPACE_RE.sub(" ", lowered).strip(" ?.")


def detect_query_mode(question: str) -> str:
    if COUNT_RE.search(question) or COMPARE_RE.search(question) or TREND_RE.search(question):
        return "meta"
    return "rag"


def parse_meta_query(question: str) -> MetaQuerySpec:
    known_concepts = _extract_known_concepts(question)
    compare_concepts = _extract_compare_concepts(question)
    wants_qualitative = bool(QUAL_RE.search(question))
    metric_match = METRIC_RE.search(question)
    metric = metric_match.group(1).lower() if metric_match else None

    if compare_concepts:
        intent = "compare"
    elif TREND_RE.search(question):
        intent = "trend"
    elif metric in {"post", "posts", "comment", "comments", "user", "users"} and not known_concepts and not _extract_subject(question):
        intent = "metric"
    else:
        intent = "count"

    subject = _extract_subject(question) or _normalize_text(question)
    return MetaQuerySpec(
        intent=intent,
        subject=subject,
        concepts=known_concepts,
        compare_concepts=compare_concepts,
        wants_qualitative=wants_qualitative,
        metric=metric,
    )


def _mask_from_known_concept(index: MetaCorpusIndex, concept_key: str) -> tuple[pd.Series, str]:
    df = index.df
    if concept_key.startswith("region_"):
        region_key = concept_key.split("_", 1)[1]
        column = f"region_{region_key}"
        if column in df.columns:
            return df[column].fillna(False).astype(bool), f"deterministic cached region tags ({region_key})"
    theme_column = f"tag_{concept_key}"
    if theme_column in df.columns:
        return df[theme_column].fillna(False).astype(bool), f"deterministic cached theme tags ({concept_key})"
    return pd.Series(False, index=df.index), "no cached tag available"


def _token_overlap_mask(index: MetaCorpusIndex, query: str) -> pd.Series:
    tokens = [token.lower() for token in TOKEN_RE.findall(query) if len(token) >= 4]
    if not tokens:
        return pd.Series(True, index=index.df.index)
    lowered = index.df["text"].fillna("").str.lower()
    mask = pd.Series(False, index=index.df.index)
    for token in tokens[:6]:
        mask = mask | lowered.str.contains(re.escape(token), regex=True)
    return mask


def _semantic_mask(index: MetaCorpusIndex, query: str) -> tuple[pd.Series, np.ndarray, float]:
    scores = index.semantic_scores(query)
    sorted_scores = np.sort(scores)[::-1]
    if len(sorted_scores) == 0:
        return pd.Series(False, index=index.df.index), scores, 1.0
    anchor = float(np.mean(sorted_scores[: min(20, len(sorted_scores))]))
    threshold = max(0.34, anchor - 0.08)
    mask = pd.Series(scores >= threshold, index=index.df.index)
    lexical_mask = _token_overlap_mask(index, query)
    if lexical_mask.any():
        high_confidence_mask = pd.Series(scores >= max(threshold + 0.05, 0.45), index=index.df.index)
        mask = (mask & lexical_mask) | high_confidence_mask
    max_fraction = 0.18
    if mask.mean() > max_fraction:
        tighter_threshold = max(threshold, float(np.quantile(scores, 1 - max_fraction)))
        mask = pd.Series(scores >= tighter_threshold, index=index.df.index)
        threshold = tighter_threshold
    return mask, scores, float(threshold)


def _examples_from_mask(index: MetaCorpusIndex, mask: pd.Series, scores: np.ndarray | None = None, limit: int = 5) -> list[MetaExamplePost]:
    if not mask.any():
        return []
    subset = index.df[mask].copy()
    if scores is not None:
        subset["match_score"] = scores[mask.to_numpy()]
        subset = subset.sort_values(["match_score", "score", "num_comments"], ascending=[False, False, False])
    else:
        subset["match_score"] = 1.0
        subset = subset.sort_values(["score", "num_comments", "topic_weight"], ascending=[False, False, False])
    examples: list[MetaExamplePost] = []
    for row in subset.head(limit).itertuples(index=False):
        examples.append(
            MetaExamplePost(
                post_id=int(row.post_id),
                title=str(row.title or ""),
                selftext=str(row.selftext or ""),
                created_utc=pd.Timestamp(row.created_utc).isoformat() if pd.notna(row.created_utc) else "",
                score=int(row.score or 0),
                num_comments=int(row.num_comments or 0),
                topic_index=int(row.topic_index) if pd.notna(row.topic_index) else None,
                topic_weight=float(row.topic_weight) if pd.notna(row.topic_weight) else None,
                match_score=float(row.match_score),
            )
        )
    return examples


def _metric_result(index: MetaCorpusIndex, spec: MetaQuerySpec) -> MetaQueryResult:
    metric = (spec.metric or "posts").lower()
    if metric.startswith("comment"):
        from reddit_insights.models import Comment, Post, Subreddit
        with SessionLocal() as session:
            count = session.scalar(
                select(pd_sql_count())
                .select_from(Comment)
                .join(Post, Comment.post_id == Post.id)
                .join(Subreddit, Post.subreddit_id == Subreddit.id)
                .where(Subreddit.name == settings.subreddit_name)
            ) or 0
        answer = f"The stored corpus currently contains {count:,} comments for r/{settings.subreddit_name}."
        return MetaQueryResult(spec.subject, "metric", answer, "direct database aggregate", "high", int(count), int(count), [], None, {"metric": "comments"})
    if metric.startswith("user"):
        from reddit_insights.models import Comment, Post, RedditUser, Subreddit
        post_users = (
            select(Post.author_id.label("author_id"))
            .join(Subreddit, Post.subreddit_id == Subreddit.id)
            .where(Subreddit.name == settings.subreddit_name, Post.author_id.is_not(None))
        )
        comment_users = (
            select(Comment.author_id.label("author_id"))
            .join(Post, Comment.post_id == Post.id)
            .join(Subreddit, Post.subreddit_id == Subreddit.id)
            .where(Subreddit.name == settings.subreddit_name, Comment.author_id.is_not(None))
        )
        unique_user_ids = pd_sql_union(post_users, comment_users).subquery()
        with SessionLocal() as session:
            count = session.scalar(select(pd_sql_count()).select_from(select(unique_user_ids.c.author_id).distinct().subquery())) or 0
        answer = f"The stored corpus currently contains {count:,} distinct posting users for r/{settings.subreddit_name}."
        return MetaQueryResult(spec.subject, "metric", answer, "direct database aggregate", "high", int(count), int(count), [], None, {"metric": "users"})
    count = index.total_posts
    answer = f"The stored corpus currently contains {count:,} posts for r/{settings.subreddit_name}."
    return MetaQueryResult(spec.subject, "metric", answer, "preprocessed post corpus count", "high", count, count, [], None, {"metric": "posts"})


# Small indirection so imports stay local and the top of the file stays lightweight.
def pd_sql_count():
    from sqlalchemy import func
    return func.count()


def pd_sql_union(left, right):
    from sqlalchemy import union
    return union(left, right)


def _combined_mask(index: MetaCorpusIndex, spec: MetaQuerySpec) -> tuple[pd.Series, str, str, np.ndarray | None]:
    if spec.compare_concepts:
        raise RuntimeError("Comparison queries should not use _combined_mask().")
    if spec.concepts:
        concept_masks: list[pd.Series] = []
        methods: list[str] = []
        for concept in spec.concepts:
            mask, method = _mask_from_known_concept(index, concept)
            concept_masks.append(mask)
            methods.append(method)
        combined = concept_masks[0].copy()
        for mask in concept_masks[1:]:
            combined = combined | mask
        return combined, "; ".join(dict.fromkeys(methods)), "high", None

    semantic_mask, scores, threshold = _semantic_mask(index, spec.subject)
    method = f"semantic similarity estimate over cached post embeddings (threshold={threshold:.3f}, model={settings.rag_embedding_model})"
    return semantic_mask, method, "medium", scores


def _count_result(index: MetaCorpusIndex, spec: MetaQuerySpec) -> MetaQueryResult:
    mask, method, confidence, scores = _combined_mask(index, spec)
    matched_count = int(mask.sum())
    share_pct = (100.0 * matched_count / index.total_posts) if index.total_posts else 0.0
    subject_label = spec.subject or ", ".join(spec.concepts) or "the requested concept"
    answer = (
        f"Estimated matching posts for '{subject_label}': {matched_count:,} of {index.total_posts:,} "
        f"posts ({share_pct:.1f}%)."
    )
    examples = _examples_from_mask(index, mask, scores=scores, limit=5)
    return MetaQueryResult(spec.subject, "count", answer, method, confidence, matched_count, index.total_posts, examples, None, {"concepts": spec.concepts})


def _compare_result(index: MetaCorpusIndex, spec: MetaQuerySpec) -> MetaQueryResult:
    left_key = spec.compare_concepts[0] if spec.compare_concepts else ""
    right_key = spec.compare_concepts[1] if len(spec.compare_concepts) > 1 else ""
    left_mask, left_method = _mask_from_known_concept(index, left_key)
    right_mask, right_method = _mask_from_known_concept(index, right_key)
    overlap_mask = left_mask & right_mask
    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    overlap_count = int(overlap_mask.sum())
    left_label = left_key.replace("region_", "").replace("_", " ") or "left concept"
    right_label = right_key.replace("region_", "").replace("_", " ") or "right concept"
    answer = (
        f"Estimated post counts: {left_label.title()} = {left_count:,}, {right_label.title()} = {right_count:,}, "
        f"overlap = {overlap_count:,}."
    )
    examples = _examples_from_mask(index, left_mask | right_mask, scores=None, limit=5)
    method = f"comparison via cached region/theme tags; left={left_method}; right={right_method}"
    return MetaQueryResult(spec.subject, "compare", answer, method, "high", int((left_mask | right_mask).sum()), index.total_posts, examples, overlap_count, {"left": left_key, "right": right_key})


def _trend_result(index: MetaCorpusIndex, spec: MetaQuerySpec) -> MetaQueryResult:
    mask, method, confidence, scores = _combined_mask(index, spec)
    subset = index.df[mask].copy()
    if subset.empty:
        return MetaQueryResult(spec.subject, "trend", f"No posts matched '{spec.subject}'.", method, confidence, 0, index.total_posts, [], None, {"concepts": spec.concepts})
    subset["month_start"] = pd.to_datetime(subset["created_utc"]).dt.to_period("M").dt.start_time
    monthly = subset.groupby("month_start").size().rename("post_count").reset_index()
    midpoint = max(1, len(monthly) // 2)
    first_half = int(monthly.iloc[:midpoint]["post_count"].sum())
    second_half = int(monthly.iloc[midpoint:]["post_count"].sum())
    direction = "increased" if second_half > first_half else "decreased" if second_half < first_half else "stayed roughly flat"
    peak_row = monthly.sort_values("post_count", ascending=False).iloc[0]
    answer = (
        f"Estimated trend for '{spec.subject}': matching-post volume {direction} over time. "
        f"First-half matches = {first_half:,}, second-half matches = {second_half:,}, peak month = "
        f"{pd.Timestamp(peak_row['month_start']).strftime('%Y-%m')} ({int(peak_row['post_count']):,} posts)."
    )
    examples = _examples_from_mask(index, mask, scores=scores, limit=5)
    return MetaQueryResult(spec.subject, "trend", answer, method, confidence, int(mask.sum()), index.total_posts, examples, None, {"monthly_counts": monthly.to_dict(orient="records")})


def run_meta_query(question: str, index: MetaCorpusIndex | None = None) -> MetaQueryResult:
    meta_index = index or MetaCorpusIndex()
    spec = parse_meta_query(question)
    if spec.intent == "metric":
        return _metric_result(meta_index, spec)
    if spec.intent == "compare":
        return _compare_result(meta_index, spec)
    if spec.intent == "trend":
        return _trend_result(meta_index, spec)
    return _count_result(meta_index, spec)


def _qualitative_follow_up(question: str, meta: MetaQueryResult) -> str:
    if meta.intent == "compare" and meta.debug:
        left = str(meta.debug.get("left", "left concept")).replace("region_", "").replace("_", " ")
        right = str(meta.debug.get("right", "right concept")).replace("region_", "").replace("_", " ")
        return f"How do users discuss {left} versus {right} graduate-admissions options in this corpus?"
    subject = meta.question or "the matching posts"
    return f"What do users say about {subject} in this corpus?"


def answer_query(
    question: str,
    provider: ChatProvider,
    rag_index: RagIndex | None = None,
    top_k: int | None = None,
    meta_index: MetaCorpusIndex | None = None,
) -> HybridAnswer:
    mode = detect_query_mode(question)
    if mode == "rag":
        rag_result = answer_question(provider, question, index=rag_index, top_k=top_k)
        return HybridAnswer(mode="rag", answer=rag_result.answer, rag=rag_result)

    meta_result = run_meta_query(question, index=meta_index)
    spec = parse_meta_query(question)
    if spec.wants_qualitative:
        qualitative_prompt = _qualitative_follow_up(question, meta_result)
        rag_result = answer_question(provider, qualitative_prompt, index=rag_index, top_k=top_k)
        answer = (
            f"{meta_result.answer}\n\n"
            f"Method: {meta_result.method}. Confidence: {meta_result.confidence}.\n\n"
            f"Qualitative summary: {rag_result.answer}"
        )
        return HybridAnswer(mode="meta_then_rag", answer=answer, meta=meta_result, rag=rag_result)

    answer = f"{meta_result.answer}\n\nMethod: {meta_result.method}. Confidence: {meta_result.confidence}."
    return HybridAnswer(mode="meta", answer=answer, meta=meta_result)


def meta_examples_to_records(meta: MetaQueryResult) -> list[dict[str, Any]]:
    return [asdict(example) for example in meta.examples]
