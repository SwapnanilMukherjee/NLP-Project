from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select, union

from reddit_insights.config import settings
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



def load_top_posts(session, topic_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(Post.title, Post.score, Post.num_comments, Post.created_utc, TopicAssignment.weight)
        .join(TopicAssignment, TopicAssignment.post_id == Post.id)
        .where(TopicAssignment.topic_id == topic_id)
        .order_by(TopicAssignment.weight.desc(), Post.score.desc())
        .limit(15)
    ).all()
    return pd.DataFrame(rows, columns=["title", "score", "num_comments", "created_utc", "topic_weight"])



def load_stance_breakdown(session, topic_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(CommentStance.stance, func.count())
        .where(CommentStance.topic_id == topic_id)
        .group_by(CommentStance.stance)
    ).all()
    return pd.DataFrame(rows, columns=["stance", "count"])



def load_user_stance_groups(session, topic_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(
            RedditUser.username,
            TopicUserStance.stance,
            TopicUserStance.comment_count,
            TopicUserStance.avg_confidence,
        )
        .join(RedditUser, TopicUserStance.user_id == RedditUser.id)
        .where(TopicUserStance.topic_id == topic_id)
        .order_by(TopicUserStance.stance.asc(), TopicUserStance.comment_count.desc(), RedditUser.username.asc())
    ).all()
    return pd.DataFrame(rows, columns=["username", "stance", "comment_count", "avg_confidence"])



def load_topic_record(session, topic_id: int) -> Topic:
    return session.get(Topic, topic_id)



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
        st.dataframe(
            topics_df[["topic_index", "label", "topic_type", "share_pct", "keywords_display", "dominant_stance"]],
            use_container_width=True,
            hide_index=True,
        )

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
        st.dataframe(top_posts_df, use_container_width=True, hide_index=True)

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
            agreement_cols = st.columns(2)
            agreement_cols[0].write("**Agreement-side users**")
            agreement_cols[0].dataframe(
                user_groups_df[user_groups_df["stance"] == "support"].head(20),
                use_container_width=True,
                hide_index=True,
            )
            agreement_cols[1].write("**Disagreement-side users**")
            agreement_cols[1].dataframe(
                user_groups_df[user_groups_df["stance"] == "oppose"].head(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No user-level stance groups were available for this topic.")

        summary_cols = st.columns(2)
        summary_cols[0].write("**Agreement-side arguments**")
        summary_cols[0].write(topic_record.support_summary or "No clear arguments detected.")
        summary_cols[1].write("**Disagreement-side arguments**")
        summary_cols[1].write(topic_record.oppose_summary or "No clear arguments detected.")
