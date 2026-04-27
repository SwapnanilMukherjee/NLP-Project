from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import and_, func, select, union

from reddit_insights.config import PART2_REPORTS_DIR, RAG_INDEX_DIR, settings
from reddit_insights.db import SessionLocal
from reddit_insights.models import Comment, CommentStance, Post, RedditUser, Subreddit, Topic, TopicAssignment, TopicUserStance, TopicWeeklyMetric



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
        st.subheader("Part 2: RAG Conversation System")
        st.caption("Ask questions over the stored Reddit corpus using the local RAG index and either Groq or Gemini as the generation endpoint.")

        metadata_path = RAG_INDEX_DIR / "metadata.json"
        if metadata_path.exists():
            st.success(f"RAG index found at `{RAG_INDEX_DIR}`.")
        else:
            st.warning("No RAG index found. Run `python -m reddit_insights.cli part2-build-index --subreddit gradadmissions` first.")

        rag_query = st.text_area("Question", placeholder="What do users say matters most for PhD admissions?")
        rag_cols = st.columns([1, 1, 2])
        provider_name = rag_cols[0].selectbox("Provider", ["groq", "gemini"])
        top_k = rag_cols[1].number_input("Retrieved snippets", min_value=3, max_value=15, value=settings.rag_top_k)

        if st.button("Answer with RAG", disabled=not metadata_path.exists() or not rag_query.strip()):
            try:
                from reddit_insights.llm_providers import build_provider
                from reddit_insights.rag import RagIndex, answer_question

                rag_result = answer_question(
                    build_provider(provider_name),
                    rag_query,
                    index=RagIndex(),
                    top_k=int(top_k),
                )
                st.write(rag_result.answer)
                with st.expander("Retrieved evidence"):
                    for item in rag_result.retrieved:
                        doc = item.document
                        st.write(f"**[{item.rank}] {doc.source_type} | {doc.topic_label} | score={item.score:.3f}**")
                        st.write(doc.title)
                        st.caption(doc.text[:800])
            except Exception as exc:
                st.error(f"RAG answer failed: {exc}")

        report_path = PART2_REPORTS_DIR / "part2_report.md"
        if report_path.exists():
            with st.expander("Generated Part 2 report"):
                st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info("Run the Part 2 evaluation commands to generate report files under `data/part2/reports`.")
