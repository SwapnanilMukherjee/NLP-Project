from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import and_, func, select, union

from reddit_insights.config import ARTIFACTS_DIR, PART2_REPORTS_DIR, RAG_INDEX_DIR, settings
from reddit_insights.corpus_qa import POST_META_MANIFEST_PATH, answer_query
from reddit_insights.db import SessionLocal
from reddit_insights.models import Comment, CommentStance, Post, RedditUser, Subreddit, Topic, TopicAssignment, TopicUserStance, TopicWeeklyMetric
HINDI_TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert English-to-Hindi translator for Reddit graduate-admissions text. "
    "Translate into natural Hindi. Preserve names, university names, acronyms such as PhD, MSCS, SOP, LoR, GPA, TA, and product names. "
    "Keep code-mixed Reddit slang understandable rather than over-literal. Output only the Hindi translation."
)

HINDI_SUMMARIZATION_SYSTEM_PROMPT = (
    "You summarize graduate-admissions discussions in Hindi using only the supplied English Reddit context. "
    "Write 2-4 sentences in natural Hindi, preserve important acronyms and named entities, and do not invent claims that are not supported by the context."
)

POSTS_WITH_TOPICS_PATH = ARTIFACTS_DIR / "posts_with_topics.parquet"
SUMMARY_EXCLUSION_PATTERN = (
    r"chance me|profile review|cv review|resume review|sop review|lor review|review my cv|review my resume|review my sop|review my profile|roast my cv|roast my resume|roast my sop"
)



def load_overview(session, subreddit_name: str) -> dict:
    post_users = (
        select(Post.author_id.label("author_id"))
        .join(Subreddit, Post.subreddit_id == Subreddit.id)
        .where(Subreddit.name == subreddit_name, Post.author_id.is_not(None))
    )
    comment_users = (
        select(Comment.author_id.label("author_id"))
        .join(Post, Comment.post_id == Post.id)
        .join(Subreddit, Post.subreddit_id == Subreddit.id)
        .where(Subreddit.name == subreddit_name, Comment.author_id.is_not(None))
    )
    unique_user_ids = union(post_users, comment_users).subquery()

    return {
        "posts": session.scalar(
            select(func.count())
            .select_from(Post)
            .join(Subreddit, Post.subreddit_id == Subreddit.id)
            .where(Subreddit.name == subreddit_name)
        ),
        "comments": session.scalar(
            select(func.count())
            .select_from(Comment)
            .join(Post, Comment.post_id == Post.id)
            .join(Subreddit, Post.subreddit_id == Subreddit.id)
            .where(Subreddit.name == subreddit_name)
        ),
        "users": session.scalar(
            select(func.count()).select_from(select(unique_user_ids.c.author_id).distinct().subquery())
        )
        or 0,
        "avg_comments_per_post": session.scalar(
            select(func.avg(Post.num_comments))
            .join(Subreddit, Post.subreddit_id == Subreddit.id)
            .where(Subreddit.name == subreddit_name)
        )
        or 0.0,
    }



def load_topics(session) -> pd.DataFrame:
    rows = session.execute(
        select(
            Topic.id,
            Topic.topic_index,
            Topic.label,
            Topic.keywords,
            Topic.share_of_posts,
            Topic.topic_type,
            Topic.active_weeks,
            Topic.total_weeks,
            Topic.trend_score,
            Topic.persistence_score,
            Topic.dominant_stance,
        ).order_by(Topic.share_of_posts.desc())
    ).all()
    return pd.DataFrame(rows, columns=[
        "topic_id",
        "topic_index",
        "label",
        "keywords",
        "share_of_posts",
        "topic_type",
        "active_weeks",
        "total_weeks",
        "trend_score",
        "persistence_score",
        "dominant_stance",
    ])



def load_weekly_metrics(session, topic_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(TopicWeeklyMetric.week_start, TopicWeeklyMetric.post_count, TopicWeeklyMetric.topic_share)
        .where(TopicWeeklyMetric.topic_id == topic_id)
        .order_by(TopicWeeklyMetric.week_start.asc())
    ).all()
    return pd.DataFrame(rows, columns=["week_start", "post_count", "topic_share"])



def load_top_posts(session, topic_id: int, limit: int = 15) -> pd.DataFrame:
    rows = session.execute(
        select(
            Post.id,
            Post.title,
            Post.selftext,
            Post.score,
            Post.num_comments,
            Post.created_utc,
            Post.permalink,
            TopicAssignment.weight,
        )
        .join(TopicAssignment, TopicAssignment.post_id == Post.id)
        .where(TopicAssignment.topic_id == topic_id)
        .order_by(TopicAssignment.weight.desc(), Post.score.desc())
        .limit(limit)
    ).all()
    return pd.DataFrame(
        rows,
        columns=["post_id", "title", "selftext", "score", "num_comments", "created_utc", "permalink", "topic_weight"],
    )



def load_random_posts(session, subreddit_name: str, limit: int) -> pd.DataFrame:
    rows = session.execute(
        select(
            Post.id,
            Post.title,
            Post.selftext,
            Post.score,
            Post.num_comments,
            Post.created_utc,
            Post.permalink,
        )
        .join(Subreddit, Post.subreddit_id == Subreddit.id)
        .where(
            Subreddit.name == subreddit_name,
            func.length(func.trim(func.coalesce(Post.selftext, ""))) > 0,
        )
        .order_by(func.random())
        .limit(limit)
    ).all()
    return pd.DataFrame(
        rows,
        columns=["post_id", "title", "selftext", "score", "num_comments", "created_utc", "permalink"],
    )



@st.cache_data(show_spinner=False)
def load_preprocessed_posts_for_summary() -> pd.DataFrame:
    if not POSTS_WITH_TOPICS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(POSTS_WITH_TOPICS_PATH)
    keep_columns = ["post_id", "title", "selftext", "text", "created_utc", "score", "num_comments", "topic_index", "topic_weight"]
    available_columns = [column for column in keep_columns if column in df.columns]
    summary_df = df[available_columns].copy()
    if "text" in summary_df.columns:
        summary_df = summary_df.rename(columns={"text": "clean_text"})
    return summary_df



def load_topic_posts_for_summary(topic_index: int, limit: int) -> pd.DataFrame:
    df = load_preprocessed_posts_for_summary()
    if df.empty:
        return pd.DataFrame()

    topic_df = df[df["topic_index"] == topic_index].copy()
    if topic_df.empty:
        return pd.DataFrame(columns=df.columns)

    title = topic_df["title"].fillna("").astype(str).str.strip()
    body = topic_df["selftext"].fillna("").astype(str).str.strip()
    clean_text = topic_df.get("clean_text", pd.Series("", index=topic_df.index)).fillna("").astype(str).str.strip()

    valid_mask = (
        body.ne("")
        & ~body.str.lower().isin({"[removed]", "[deleted]", "removed", "deleted"})
        & ~title.str.lower().str.contains(SUMMARY_EXCLUSION_PATTERN, regex=True, na=False)
        & ~body.str.lower().str.contains(SUMMARY_EXCLUSION_PATTERN, regex=True, na=False)
        & (body.str.len() >= 220)
        & (clean_text.str.len() >= 320)
    )
    filtered = topic_df[valid_mask].copy()
    if len(filtered) < min(limit, 8):
        fallback_mask = body.ne("") & ~body.str.lower().isin({"[removed]", "[deleted]", "removed", "deleted"}) & (clean_text.str.len() >= 220)
        filtered = topic_df[fallback_mask].copy()
    if filtered.empty:
        return pd.DataFrame(columns=topic_df.columns)
    sample_size = min(limit, len(filtered))
    return filtered.sample(n=sample_size).reset_index(drop=True)



def load_topic_comments_for_summary(session, topic_id: int, per_stance_limit: int = 6) -> pd.DataFrame:
    rows = session.execute(
        select(
            Comment.body,
            Comment.score,
            Comment.created_utc,
            CommentStance.stance,
            CommentStance.confidence,
            Post.title,
        )
        .select_from(CommentStance)
        .join(Comment, CommentStance.comment_id == Comment.id)
        .join(Post, Comment.post_id == Post.id)
        .where(CommentStance.topic_id == topic_id)
        .order_by(CommentStance.confidence.desc(), Comment.score.desc(), Comment.created_utc.desc())
    ).all()
    df = pd.DataFrame(rows, columns=["body", "score", "created_utc", "stance", "confidence", "post_title"])
    if df.empty:
        return df

    body = df["body"].fillna("").astype(str).str.strip()
    df = df[
        body.ne("")
        & ~body.str.lower().isin({"[removed]", "[deleted]", "removed", "deleted"})
        & (body.str.len() >= 80)
    ].copy()
    if df.empty:
        return df

    parts = []
    for stance in ["support", "oppose", "neutral"]:
        stance_limit = per_stance_limit if stance != "neutral" else max(2, per_stance_limit // 2)
        part = df[df["stance"] == stance].head(stance_limit)
        if not part.empty:
            parts.append(part)
    if not parts:
        return df.head(per_stance_limit * 2).reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)



def refresh_topic_summary_sample(topic_id: int, topic_index: int, limit: int) -> pd.DataFrame:
    posts_df = load_topic_posts_for_summary(topic_index, limit=limit)
    st.session_state[f"summarization_sample_posts_{topic_id}"] = posts_df.to_dict(orient="records")
    return posts_df



def get_topic_summary_sample(topic_id: int, topic_index: int, limit: int) -> pd.DataFrame:
    key = f"summarization_sample_posts_{topic_id}"
    records = st.session_state.get(key)
    if not records:
        return refresh_topic_summary_sample(topic_id, topic_index, limit)
    return pd.DataFrame(records)



def load_stance_breakdown(session, topic_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(CommentStance.stance, func.count())
        .where(CommentStance.topic_id == topic_id)
        .group_by(CommentStance.stance)
    ).all()
    return pd.DataFrame(rows, columns=["stance", "count"])



def load_user_stance_groups(session, topic_id: int) -> pd.DataFrame:
    representative_comment_subquery = (
        select(
            TopicUserStance.user_id.label("user_id"),
            TopicUserStance.topic_id.label("topic_id"),
            Comment.body.label("representative_comment"),
            func.row_number().over(
                partition_by=(TopicUserStance.user_id, TopicUserStance.topic_id),
                order_by=(
                    CommentStance.confidence.desc(),
                    Comment.score.desc(),
                    Comment.created_utc.desc(),
                    Comment.id.desc(),
                ),
            ).label("row_num"),
        )
        .join(Comment, Comment.author_id == TopicUserStance.user_id)
        .join(
            CommentStance,
            and_(
                CommentStance.comment_id == Comment.id,
                CommentStance.topic_id == TopicUserStance.topic_id,
                CommentStance.stance == TopicUserStance.stance,
            ),
        )
        .where(TopicUserStance.topic_id == topic_id)
        .subquery()
    )

    rows = session.execute(
        select(
            RedditUser.username,
            TopicUserStance.stance,
            TopicUserStance.avg_confidence,
            representative_comment_subquery.c.representative_comment,
        )
        .join(RedditUser, TopicUserStance.user_id == RedditUser.id)
        .outerjoin(
            representative_comment_subquery,
            and_(
                representative_comment_subquery.c.user_id == TopicUserStance.user_id,
                representative_comment_subquery.c.topic_id == TopicUserStance.topic_id,
                representative_comment_subquery.c.row_num == 1,
            ),
        )
        .where(TopicUserStance.topic_id == topic_id)
        .order_by(TopicUserStance.stance.asc(), TopicUserStance.avg_confidence.desc(), RedditUser.username.asc())
    ).all()
    return pd.DataFrame(rows, columns=["username", "stance", "avg_confidence", "representative_comment"])



def load_topic_record(session, topic_id: int) -> Topic:
    return session.get(Topic, topic_id)



def format_timestamp(value) -> str:
    if pd.isna(value):
        return "-"
    return pd.to_datetime(value).strftime("%Y-%m-%d")



def truncate_text(value: str, limit: int = 100) -> str:
    clean = (value or "").strip()
    if len(clean) <= limit:
        return clean or "(untitled)"
    return clean[: limit - 1].rstrip() + "…"



def post_input_text(title: str, selftext: str) -> str:
    clean_title = (title or "").strip()
    clean_body = (selftext or "").strip()
    if clean_body:
        return f"Title: {clean_title}\n\nBody:\n{clean_body}"
    return f"Title: {clean_title}"



def render_corpus_post_rows(posts_df: pd.DataFrame, key_prefix: str) -> None:
    if posts_df.empty:
        st.info("No sampled posts were available.")
        return

    header_cols = st.columns([5, 1, 1, 1])
    header_cols[0].write("**Title**")
    header_cols[1].write("**Score**")
    header_cols[2].write("**Comments**")
    header_cols[3].write("**Created**")

    for row in posts_df.itertuples(index=False):
        row_cols = st.columns([5, 1, 1, 1])
        title_label = truncate_text(str(row.title), 110)
        permalink = getattr(row, "permalink", "") or ""
        clean_text = getattr(row, "clean_text", "") or ""
        body_text = str(row.selftext or "").strip() or str(clean_text).strip()
        with row_cols[0].popover(title_label, key=f"{key_prefix}-sample-{row.post_id}", use_container_width=True):
            st.markdown(f"**{row.title or '(untitled)'}**")
            if body_text:
                st.write(body_text)
            else:
                st.caption("No body text available for this post.")
            if str(permalink).strip():
                st.caption(permalink)
        row_cols[1].write(int(row.score or 0))
        row_cols[2].write(int(row.num_comments or 0))
        row_cols[3].write(format_timestamp(row.created_utc))



def refresh_post_sample(session, subreddit_name: str, state_key: str, limit: int, loader=load_random_posts) -> pd.DataFrame:
    posts_df = loader(session, subreddit_name, limit=limit)
    st.session_state[state_key] = posts_df.to_dict(orient="records")
    return posts_df



def get_post_sample(session, subreddit_name: str, state_key: str, limit: int, loader=load_random_posts) -> pd.DataFrame:
    records = st.session_state.get(state_key)
    if not records:
        return refresh_post_sample(session, subreddit_name, state_key, limit, loader=loader)
    return pd.DataFrame(records)



def build_translation_post_options(posts_df: pd.DataFrame) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for row in posts_df.itertuples(index=False):
        options.append(
            {
                "post_id": int(row.post_id),
                "label": f"{truncate_text(str(row.title), 70)} | {format_timestamp(row.created_utc)} | score={int(row.score or 0)}",
            }
        )
    return options



def build_topic_summary_context(topic_label: str, keywords_text: str, posts_df: pd.DataFrame, comments_df: pd.DataFrame, max_chars: int = 9000) -> str:
    context_parts: list[str] = [f"Topic: {topic_label}", f"Keywords: {keywords_text}"]
    running_chars = sum(len(part) for part in context_parts) + 4

    for position, row in enumerate(posts_df.itertuples(index=False), start=1):
        title = str(row.title or "").strip()
        clean_text = str(getattr(row, "clean_text", "") or "").strip()
        body = clean_text or str(row.selftext or "").strip()
        if not body:
            continue
        snippet = body[:700].strip()
        part = f"[Post {position}]\nTitle: {title}\nText: {snippet}"
        projected = running_chars + len(part) + 2
        if projected > max_chars and len(context_parts) > 2:
            break
        context_parts.append(part)
        running_chars = projected

    for position, row in enumerate(comments_df.itertuples(index=False), start=1):
        body = str(row.body or "").strip()
        if not body:
            continue
        snippet = body[:280].strip()
        part = f"[Comment {position}]\nStance: {row.stance}\nSource post: {row.post_title}\nText: {snippet}"
        projected = running_chars + len(part) + 2
        if projected > max_chars:
            break
        context_parts.append(part)
        running_chars = projected
    return "\n\n".join(context_parts)



def translate_text_to_hindi(provider, text: str) -> str:
    user_prompt = f"Translate this text to Hindi:\n{text.strip()}"
    return provider.chat(HINDI_TRANSLATION_SYSTEM_PROMPT, user_prompt, max_tokens=512, temperature=0.0).text



def summarize_topic_in_hindi(provider, topic_label: str, keywords_text: str, posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> str:
    context = build_topic_summary_context(topic_label, keywords_text, posts_df, comments_df)
    if not context.strip():
        raise RuntimeError("The selected topic did not contain enough content to summarize.")
    user_prompt = (
        "इस topic discussion का 3-5 वाक्यों में हिंदी में सार लिखिए। मुख्य themes, repeated concerns, useful admissions details, और disagreement where relevant शामिल कीजिए.\n\n"
        f"English Reddit topic context:\n{context}\n\n"
        "सार केवल हिंदी में लिखिए।"
    )
    return provider.chat(HINDI_SUMMARIZATION_SYSTEM_PROMPT, user_prompt, max_tokens=360, temperature=0.0).text



def render_post_rows(posts_df: pd.DataFrame, key_prefix: str) -> None:
    if posts_df.empty:
        st.info("No representative posts were available for this topic.")
        return

    header_cols = st.columns([5, 1, 1, 1, 1])
    header_cols[0].write("**Title**")
    header_cols[1].write("**Score**")
    header_cols[2].write("**Comments**")
    header_cols[3].write("**Created**")
    header_cols[4].write("**Weight**")

    for row in posts_df.itertuples(index=False):
        row_cols = st.columns([5, 1, 1, 1, 1])
        title_label = truncate_text(str(row.title), 110)
        with row_cols[0].popover(title_label, key=f"{key_prefix}-post-{row.post_id}", use_container_width=True):
            st.markdown(f"**{row.title or '(untitled)'}**")
            if str(row.selftext or "").strip():
                st.write(row.selftext)
            else:
                st.caption("No body text available for this post.")
            if str(row.permalink or "").strip():
                st.caption(row.permalink)
        row_cols[1].write(int(row.score or 0))
        row_cols[2].write(int(row.num_comments or 0))
        row_cols[3].write(format_timestamp(row.created_utc))
        row_cols[4].write(f"{float(row.topic_weight or 0.0):.3f}")


def render_meta_example_rows(examples: list, key_prefix: str) -> None:
    if not examples:
        st.info("No representative example posts were available for this query.")
        return

    header_cols = st.columns([5, 1, 1, 1, 1])
    header_cols[0].write("**Title**")
    header_cols[1].write("**Score**")
    header_cols[2].write("**Comments**")
    header_cols[3].write("**Created**")
    header_cols[4].write("**Match**")

    for example in examples:
        row_cols = st.columns([5, 1, 1, 1, 1])
        title_label = truncate_text(str(example.title), 110)
        with row_cols[0].popover(title_label, key=f"{key_prefix}-meta-{example.post_id}", use_container_width=True):
            st.markdown(f"**{example.title or '(untitled)'}**")
            if str(example.selftext or "").strip():
                st.write(example.selftext)
            else:
                st.caption("No body text available for this post.")
            if example.topic_index is not None:
                st.caption(f"Topic index: {example.topic_index}")
        row_cols[1].write(int(example.score or 0))
        row_cols[2].write(int(example.num_comments or 0))
        row_cols[3].write(format_timestamp(example.created_utc))
        row_cols[4].write(f"{float(example.match_score or 0.0):.3f}")



def render_topic_overview(topics_df: pd.DataFrame) -> int | None:
    header_cols = st.columns([1, 3, 1.5, 1, 3, 1.5])
    header_cols[0].write("**Topic #**")
    header_cols[1].write("**Label**")
    header_cols[2].write("**Type**")
    header_cols[3].write("**Share**")
    header_cols[4].write("**Keywords**")
    header_cols[5].write("**Dominant stance**")

    for row in topics_df.itertuples(index=False):
        row_cols = st.columns([1, 3, 1.5, 1, 3, 1.5])
        row_cols[0].write(int(row.topic_index))
        if row_cols[1].button(
            str(row.label),
            key=f"topic-overview-{row.topic_id}",
            type="tertiary",
            use_container_width=True,
        ):
            st.session_state["overview_topic_id"] = int(row.topic_id)
        row_cols[2].write(str(row.topic_type).title())
        row_cols[3].write(f"{float(row.share_pct):.1f}%")
        row_cols[4].write(str(row.keywords_display))
        row_cols[5].write(str(row.dominant_stance).replace("_", " ").title())

    return st.session_state.get("overview_topic_id")



def main() -> None:
    subreddit_name = settings.subreddit_name
    st.set_page_config(page_title="Reddit Topic and Stance Explorer", layout="wide")
    st.title("Reddit Topic and Stance Explorer")
    st.caption(
        f"Assignment implementation for r/{subreddit_name}: aggregates, topic discovery, trend/persistence separation, and topic-level stance analysis."
    )

    with SessionLocal() as session:
        overview = load_overview(session, subreddit_name)
        topics_df = load_topics(session)

        if overview["posts"] == 0:
            st.warning(
                f"No posts are currently stored for r/{subreddit_name}. Run ingestion and analysis for this subreddit before using the dashboard."
            )
            return

        if topics_df.empty:
            st.warning(
                f"No analyzed topics found. Run `python -m reddit_insights.cli analyze --subreddit {subreddit_name}` after loading data."
            )
            return

        metric_cols = st.columns(4)
        metric_cols[0].metric("Posts", f"{overview['posts']:,}")
        metric_cols[1].metric("Comments", f"{overview['comments']:,}")
        metric_cols[2].metric("Users", f"{overview['users']:,}")
        metric_cols[3].metric("Avg comments/post", f"{overview['avg_comments_per_post']:.2f}")

        topics_df["keywords_display"] = topics_df["keywords"].apply(lambda value: ", ".join(json.loads(value)[:8]))
        topics_df["share_pct"] = topics_df["share_of_posts"] * 100

        st.subheader("Topic Overview")
        st.caption("Click a topic label to inspect its top posts and open full post bodies.")
        overview_topic_id = render_topic_overview(
            topics_df[["topic_id", "topic_index", "label", "topic_type", "share_pct", "keywords_display", "dominant_stance"]]
        )

        if overview_topic_id is not None and int(overview_topic_id) in topics_df["topic_id"].tolist():
            overview_topic = topics_df[topics_df["topic_id"] == int(overview_topic_id)].iloc[0]
            st.write(f"**Top posts for topic:** {overview_topic['label']}")
            st.caption(
                f"{overview_topic['topic_type'].title()} topic • {overview_topic['share_pct']:.1f}% of posts • click a title to view the full body."
            )
            st.write(f"**Keywords:** {overview_topic['keywords_display']}")
            overview_top_k = st.number_input(
                "Top posts to show for the selected topic",
                min_value=5,
                max_value=50,
                value=int(st.session_state.get("overview_topic_top_k", 20)),
                key="overview_topic_top_k",
            )
            overview_posts_df = load_top_posts(session, int(overview_topic_id), limit=int(overview_top_k))
            render_post_rows(overview_posts_df, key_prefix=f"overview-topic-{int(overview_topic_id)}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                topics_df.sort_values("share_of_posts", ascending=False),
                x="label",
                y="share_pct",
                color="topic_type",
                title="Share of Total Posts by Topic",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                topics_df,
                x="persistence_score",
                y="trend_score",
                size="share_pct",
                color="topic_type",
                hover_name="label",
                title="Trending vs Persistent Topics",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Topic Drilldown")
        selected_label = st.selectbox("Select a topic", topics_df["label"].tolist())
        selected = topics_df[topics_df["label"] == selected_label].iloc[0]
        topic_id = int(selected["topic_id"])
        topic_record = load_topic_record(session, topic_id)

        detail_cols = st.columns(3)
        detail_cols[0].metric("Topic share", f"{selected['share_pct']:.1f}%")
        detail_cols[1].metric("Topic type", selected["topic_type"].title())
        detail_cols[2].metric("Dominant stance", topic_record.dominant_stance.replace("_", " ").title())

        st.write(f"**Keywords:** {selected['keywords_display']}")

        weekly_df = load_weekly_metrics(session, topic_id)
        if not weekly_df.empty:
            st.plotly_chart(
                px.line(weekly_df, x="week_start", y="topic_share", title="Weekly Topic Share"),
                use_container_width=True,
            )

        top_posts_df = load_top_posts(session, topic_id)
        st.write("**Representative posts**")
        st.caption("Click a title to view the full post body.")
        render_post_rows(top_posts_df, key_prefix=f"drilldown-topic-{topic_id}")

        stance_df = load_stance_breakdown(session, topic_id)
        st.write("**Agreement vs disagreement**")
        if not stance_df.empty:
            st.plotly_chart(
                px.pie(stance_df, names="stance", values="count", title="Comment stance distribution"),
                use_container_width=True,
            )
        else:
            st.info("No stance-bearing comments were available for this topic.")

        user_groups_df = load_user_stance_groups(session, topic_id)
        st.write("**User groups by stance**")
        if not user_groups_df.empty:
            user_count_df = user_groups_df.groupby("stance", as_index=False).agg(users=("username", "count"))
            st.plotly_chart(
                px.bar(user_count_df, x="stance", y="users", title="Users grouped by stance"),
                use_container_width=True,
            )
            display_user_groups_df = user_groups_df.copy()
            display_user_groups_df["avg_confidence"] = display_user_groups_df["avg_confidence"].round(3)
            agreement_cols = st.columns(2)
            agreement_cols[0].write("**Agreement-side users**")
            agreement_cols[0].dataframe(
                display_user_groups_df[display_user_groups_df["stance"] == "support"].head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "representative_comment": st.column_config.TextColumn("Representative comment", width="large"),
                    "avg_confidence": st.column_config.NumberColumn("Avg confidence", format="%.3f"),
                },
            )
            agreement_cols[1].write("**Disagreement-side users**")
            agreement_cols[1].dataframe(
                display_user_groups_df[display_user_groups_df["stance"] == "oppose"].head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "representative_comment": st.column_config.TextColumn("Representative comment", width="large"),
                    "avg_confidence": st.column_config.NumberColumn("Avg confidence", format="%.3f"),
                },
            )
        else:
            st.info("No user-level stance groups were available for this topic.")

        summary_cols = st.columns(2)
        summary_cols[0].write("**Agreement-side arguments**")
        summary_cols[0].write(topic_record.support_summary or "No clear arguments detected.")
        summary_cols[1].write("**Disagreement-side arguments**")
        summary_cols[1].write(topic_record.oppose_summary or "No clear arguments detected.")

        st.divider()
        st.subheader("Part 2: Hybrid QA and Corpus Analytics")
        st.caption("Ask narrative questions over the Reddit corpus with RAG, or ask count/compare/trend questions that operate directly on the stored and preprocessed post data.")

        metadata_path = RAG_INDEX_DIR / "metadata.json"
        if metadata_path.exists():
            st.success(f"RAG index found at `{RAG_INDEX_DIR}`.")
        else:
            st.warning("No RAG index found. Narrative RAG questions will fail until you run `python -m reddit_insights.cli part2-build-index --subreddit gradadmissions`. Meta-level corpus questions can still build and use the post cache.")

        if POST_META_MANIFEST_PATH.exists():
            st.success(f"Corpus analytics cache found at `{POST_META_MANIFEST_PATH}`.")
        else:
            st.info("No corpus-analytics cache found yet. The first meta-level query will build it from the preprocessed post artifact.")

        rag_query = st.text_area("Question", placeholder="How many posts discuss funding, and what do users usually say about it?")
        rag_cols = st.columns([1, 1, 2])
        provider_name = rag_cols[0].selectbox("Provider", ["groq", "gemini"])
        top_k = rag_cols[1].number_input("Retrieved snippets", min_value=3, max_value=15, value=settings.rag_top_k)

        if st.button("Answer question", disabled=not rag_query.strip()):
            try:
                from reddit_insights.llm_providers import build_provider

                hybrid_result = answer_query(
                    rag_query,
                    provider=build_provider(provider_name),
                    top_k=int(top_k),
                )
                st.caption(f"Mode: {hybrid_result.mode.replace('_', ' ').title()}")
                st.write(hybrid_result.answer)
                if hybrid_result.meta:
                    with st.expander("Corpus analytics details", expanded=True):
                        st.write(f"**Method:** {hybrid_result.meta.method}")
                        st.write(f"**Confidence:** {hybrid_result.meta.confidence.title()}")
                        st.write(f"**Matched posts:** {hybrid_result.meta.matched_count:,} / {hybrid_result.meta.total_posts:,}")
                        if hybrid_result.meta.overlap_count is not None:
                            st.write(f"**Overlap count:** {hybrid_result.meta.overlap_count:,}")
                        render_meta_example_rows(hybrid_result.meta.examples, key_prefix=f"meta-query-{provider_name}")
                if hybrid_result.rag:
                    with st.expander("Retrieved evidence"):
                        for item in hybrid_result.rag.retrieved:
                            doc = item.document
                            st.write(f"**[{item.rank}] {doc.source_type} | {doc.topic_label} | score={item.score:.3f}**")
                            st.write(doc.title)
                            st.caption(doc.text[:800])
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

        st.subheader("Part 2: Translation")
        st.caption("Pick one of 10 sampled English posts, inspect its full content, optionally copy it into the input box, and generate a Hindi translation.")

        translation_posts_df = get_post_sample(session, subreddit_name, state_key="translation_sample_posts", limit=10)
        translation_action_cols = st.columns([1, 3])
        if translation_action_cols[0].button("Refresh sample posts", key="translation-refresh"):
            translation_posts_df = refresh_post_sample(session, subreddit_name, state_key="translation_sample_posts", limit=10)

        if translation_posts_df.empty:
            st.info("No translatable posts with body text were available.")
        else:
            translation_options = build_translation_post_options(translation_posts_df)
            translation_post_ids = [option["post_id"] for option in translation_options]
            if st.session_state.get("translation_selected_post_id") not in translation_post_ids:
                st.session_state["translation_selected_post_id"] = translation_post_ids[0]
            selected_post_id = st.selectbox(
                "Sample English posts",
                options=translation_post_ids,
                format_func=lambda post_id: next(option["label"] for option in translation_options if option["post_id"] == post_id),
                key="translation_selected_post_id",
            )
            selected_translation_post = translation_posts_df[translation_posts_df["post_id"] == selected_post_id].iloc[0]
            st.write(f"**Title:** {selected_translation_post['title']}")
            st.write(selected_translation_post["selftext"])

            if st.button("Copy selected post into the translation box", key="translation-prefill"):
                st.session_state["translation_input"] = post_input_text(
                    str(selected_translation_post["title"] or ""),
                    str(selected_translation_post["selftext"] or ""),
                )

            translation_input = st.text_area(
                "English text to translate",
                key="translation_input",
                height=220,
                placeholder="Paste or edit the English post text here before translating.",
            )

            translation_cols = st.columns([1, 1, 2])
            translation_provider_name = translation_cols[0].selectbox("Provider", ["groq", "gemini"], key="translation_provider")
            if translation_cols[1].button("Translate", key="translation-submit", disabled=not translation_input.strip()):
                try:
                    from reddit_insights.llm_providers import build_provider

                    translated_text = translate_text_to_hindi(
                        build_provider(translation_provider_name),
                        translation_input,
                    )
                    st.write("**Hindi translation**")
                    st.write(translated_text)
                except Exception as exc:
                    st.error(f"Translation failed: {exc}")

        st.subheader("Part 2: Hindi Summarization")
        st.caption("Select a topic discussion, inspect 20 clean posts from that topic, and generate a Hindi summary grounded in topic posts plus representative topic comments.")

        summarization_topic_ids = topics_df["topic_id"].astype(int).tolist()
        if st.session_state.get("summarization_topic_id") not in summarization_topic_ids:
            st.session_state["summarization_topic_id"] = topic_id
        summarization_topic_id = st.selectbox(
            "Topic for Hindi summarization",
            options=summarization_topic_ids,
            format_func=lambda current_topic_id: topics_df.loc[topics_df["topic_id"] == current_topic_id, "label"].iloc[0],
            key="summarization_topic_id",
        )
        summarization_topic = topics_df[topics_df["topic_id"] == int(summarization_topic_id)].iloc[0]
        st.write(f"**Selected topic:** {summarization_topic['label']}")
        st.write(f"**Keywords:** {summarization_topic['keywords_display']}")

        summarization_posts_df = get_topic_summary_sample(
            int(summarization_topic_id),
            int(summarization_topic['topic_index']),
            limit=20,
        )
        summarization_comments_df = load_topic_comments_for_summary(session, int(summarization_topic_id), per_stance_limit=6)
        summarization_action_cols = st.columns([1, 1, 3])
        if summarization_action_cols[0].button("Update posts", key="summarization-refresh"):
            summarization_posts_df = refresh_topic_summary_sample(
                int(summarization_topic_id),
                int(summarization_topic['topic_index']),
                limit=20,
            )
        if not summarization_posts_df.empty:
            render_corpus_post_rows(summarization_posts_df, key_prefix=f"summarization-topic-{int(summarization_topic_id)}")
            summarization_provider_name = summarization_action_cols[1].selectbox("Provider", ["groq", "gemini"], key="summarization_provider")
            if st.button("Generate Hindi summary", key="summarization-submit"):
                try:
                    from reddit_insights.llm_providers import build_provider

                    summary_text = summarize_topic_in_hindi(
                        build_provider(summarization_provider_name),
                        str(summarization_topic['label']),
                        str(summarization_topic['keywords_display']),
                        summarization_posts_df,
                        summarization_comments_df,
                    )
                    st.write("**Hindi summary**")
                    st.write(summary_text)
                except Exception as exc:
                    st.error(f"Hindi summarization failed: {exc}")
        else:
            st.info("No clean content-rich posts were available for the selected topic.")


        report_path = PART2_REPORTS_DIR / "part2_report.md"
        if report_path.exists():
            with st.expander("Generated Part 2 report"):
                st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info("Run the Part 2 evaluation commands to generate report files under `data/part2/reports`.")
