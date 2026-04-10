from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Subreddit(Base):
    __tablename__ = "subreddits"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    posts: Mapped[list["Post"]] = relationship(back_populates="subreddit")


class RedditUser(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    posts: Mapped[list["Post"]] = relationship(back_populates="author")
    comments: Mapped[list["Comment"]] = relationship(back_populates="author")


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    reddit_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    subreddit_id: Mapped[int] = mapped_column(ForeignKey("subreddits.id"), index=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    selftext: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[int] = mapped_column(Integer, default=0)
    num_comments: Mapped[int] = mapped_column(Integer, default=0)
    created_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    permalink: Mapped[str] = mapped_column(Text, default="")
    listing_source: Mapped[str] = mapped_column(String(50), default="unknown")

    subreddit: Mapped["Subreddit"] = relationship(back_populates="posts")
    author: Mapped["RedditUser | None"] = relationship(back_populates="posts")
    comments: Mapped[list["Comment"]] = relationship(back_populates="post")
    topic_assignments: Mapped[list["TopicAssignment"]] = relationship(back_populates="post")


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    reddit_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), index=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    parent_reddit_id: Mapped[str] = mapped_column(String(20), default="")
    body: Mapped[str] = mapped_column(Text)
    score: Mapped[int] = mapped_column(Integer, default=0)
    created_utc: Mapped[datetime] = mapped_column(DateTime, index=True)

    post: Mapped["Post"] = relationship(back_populates="comments")
    author: Mapped["RedditUser | None"] = relationship(back_populates="comments")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_index: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(255))
    keywords: Mapped[str] = mapped_column(Text)
    share_of_posts: Mapped[float] = mapped_column(Float)
    topic_type: Mapped[str] = mapped_column(String(20))
    active_weeks: Mapped[int] = mapped_column(Integer, default=0)
    total_weeks: Mapped[int] = mapped_column(Integer, default=0)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0)
    persistence_score: Mapped[float] = mapped_column(Float, default=0.0)
    support_summary: Mapped[str] = mapped_column(Text, default="")
    oppose_summary: Mapped[str] = mapped_column(Text, default="")
    dominant_stance: Mapped[str] = mapped_column(String(20), default="support")

    assignments: Mapped[list["TopicAssignment"]] = relationship(back_populates="topic")
    metrics: Mapped[list["TopicWeeklyMetric"]] = relationship(back_populates="topic")


class TopicAssignment(Base):
    __tablename__ = "topic_assignments"
    __table_args__ = (UniqueConstraint("post_id", name="uq_topic_assignment_post"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    weight: Mapped[float] = mapped_column(Float)

    post: Mapped["Post"] = relationship(back_populates="topic_assignments")
    topic: Mapped["Topic"] = relationship(back_populates="assignments")


class TopicWeeklyMetric(Base):
    __tablename__ = "topic_weekly_metrics"
    __table_args__ = (UniqueConstraint("topic_id", "week_start", name="uq_topic_week"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    week_start: Mapped[datetime] = mapped_column(DateTime, index=True)
    post_count: Mapped[int] = mapped_column(Integer)
    topic_share: Mapped[float] = mapped_column(Float)

    topic: Mapped["Topic"] = relationship(back_populates="metrics")


class CommentStance(Base):
    __tablename__ = "comment_stances"
    __table_args__ = (UniqueConstraint("comment_id", "topic_id", name="uq_comment_topic_stance"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    comment_id: Mapped[int] = mapped_column(ForeignKey("comments.id"), index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    stance: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[str] = mapped_column(Text, default="")


class TopicUserStance(Base):
    __tablename__ = "topic_user_stances"
    __table_args__ = (UniqueConstraint("user_id", "topic_id", name="uq_topic_user_stance"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    stance: Mapped[str] = mapped_column(String(20))
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
