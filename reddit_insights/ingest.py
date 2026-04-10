from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from reddit_insights.arctic_shift import ArcticShiftClient, parse_arctic_datetime, strip_fullname
from reddit_insights.config import normalize_subreddit_name
from reddit_insights.models import Comment, Post, RedditUser, Subreddit
from reddit_insights.preprocess import normalize_text


@dataclass
class IngestStats:
    posts_created: int = 0
    posts_updated: int = 0
    comments_created: int = 0
    comments_updated: int = 0



def get_or_create_subreddit(session: Session, name: str) -> Subreddit:
    record = session.scalar(select(Subreddit).where(Subreddit.name == name))
    if record:
        return record
    record = Subreddit(name=name)
    session.add(record)
    session.flush()
    return record



def get_or_create_user(session: Session, username: str | None) -> RedditUser | None:
    if not username:
        return None
    username = username.strip()
    if not username:
        return None
    record = session.scalar(select(RedditUser).where(RedditUser.username == username))
    if record:
        return record
    record = RedditUser(username=username, is_deleted=(username == "[deleted]"))
    session.add(record)
    session.flush()
    return record



def upsert_post(session: Session, subreddit: Subreddit, payload: dict, stats: IngestStats) -> Post:
    record = session.scalar(select(Post).where(Post.reddit_id == payload["reddit_id"]))
    author = get_or_create_user(session, payload.get("author"))
    is_create = record is None
    if is_create:
        record = Post(reddit_id=payload["reddit_id"], subreddit=subreddit)
        session.add(record)
    record.subreddit = subreddit
    record.author = author
    record.title = payload["title"]
    record.selftext = payload.get("selftext", "")
    record.url = payload.get("url", "")
    record.score = payload.get("score", 0)
    record.num_comments = payload.get("num_comments", 0)
    record.created_utc = payload["created_utc"]
    record.permalink = payload.get("permalink", "")
    record.listing_source = payload.get("listing_source", "unknown")
    session.flush()
    if is_create:
        stats.posts_created += 1
    else:
        stats.posts_updated += 1
    return record



def upsert_comment(session: Session, post: Post, payload: dict, stats: IngestStats) -> Comment:
    record = session.scalar(select(Comment).where(Comment.reddit_id == payload["reddit_id"]))
    author = get_or_create_user(session, payload.get("author"))
    is_create = record is None
    if is_create:
        record = Comment(reddit_id=payload["reddit_id"], post=post)
        session.add(record)
    record.post = post
    record.author = author
    record.parent_reddit_id = payload.get("parent_reddit_id", "")
    record.body = payload["body"]
    record.score = payload.get("score", 0)
    record.created_utc = payload["created_utc"]
    session.flush()
    if is_create:
        stats.comments_created += 1
    else:
        stats.comments_updated += 1
    return record



def scrape_subreddit(session: Session, subreddit_name: str, days: int, post_target: int, comment_limit: int) -> IngestStats:
    subreddit_name = normalize_subreddit_name(subreddit_name)
    client = ArcticShiftClient()
    subreddit = get_or_create_subreddit(session, subreddit_name)
    stats = IngestStats()
    after = datetime.utcnow() - timedelta(days=days)
    before = datetime.utcnow()

    seen_post_ids: set[str] = set()
    for submission in tqdm(
        client.iter_posts(subreddit_name=subreddit_name, after=after, before=before),
        desc=f"arctic:posts:{subreddit_name}",
    ):
        submission_id = str(submission.get("id") or "").strip()
        if not submission_id or submission_id in seen_post_ids:
            continue
        seen_post_ids.add(submission_id)
        upsert_post(
            session,
            subreddit,
            {
                "reddit_id": submission_id,
                "author": submission.get("author"),
                "title": normalize_text(submission.get("title", "")),
                "selftext": normalize_text(submission.get("selftext", "")),
                "url": submission.get("url", ""),
                "score": int(submission.get("score") or 0),
                "num_comments": int(submission.get("num_comments") or 0),
                "created_utc": parse_arctic_datetime(submission["created_utc"]),
                "permalink": submission.get("permalink") or f"/r/{subreddit_name}/comments/{submission_id}",
                "listing_source": "arctic_shift_api",
            },
            stats,
        )

        if len(seen_post_ids) % 250 == 0:
            session.commit()
        if len(seen_post_ids) >= post_target:
            break

    session.commit()

    if comment_limit == 0 or not seen_post_ids:
        return stats

    post_rows = session.execute(
        select(Post.reddit_id, Post.id, Post.num_comments, Post.created_utc)
        .join(Subreddit, Post.subreddit_id == Subreddit.id)
        .where(Subreddit.name == subreddit_name)
    ).all()
    post_map = {reddit_id: post_id for reddit_id, post_id, _, _ in post_rows}
    comment_counts = {
        post_id: count
        for post_id, count in session.execute(
            select(Comment.post_id, func.count(Comment.id)).group_by(Comment.post_id)
        )
    }
    if comment_limit < 0:
        remaining_post_ids = {post_id for _, post_id, _, _ in post_rows}
        comment_after = after
    else:
        comment_targets = {
            post_id: min(max(num_comments or 0, 0), comment_limit)
            for _, post_id, num_comments, _ in post_rows
        }
        remaining_post_ids = {
            post_id
            for _, post_id, _, _ in post_rows
            if comment_counts.get(post_id, 0) < comment_targets.get(post_id, 0)
        }
        comment_after = min(
            (created_utc for _, post_id, _, created_utc in post_rows if post_id in remaining_post_ids),
            default=after,
        )

    if not remaining_post_ids:
        return stats

    seen_comment_ids: set[str] = set()

    for item in tqdm(
        client.iter_comments(subreddit_name=subreddit_name, after=comment_after, before=before),
        desc=f"arctic:comments:{subreddit_name}",
    ):
        if not remaining_post_ids:
            break
        comment_id = str(item.get("id") or "").strip()
        if not comment_id or comment_id in seen_comment_ids:
            continue
        seen_comment_ids.add(comment_id)

        link_id = strip_fullname(item.get("link_id"), "t3_")
        post_id = post_map.get(link_id)
        if not post_id or post_id not in remaining_post_ids:
            continue

        post = session.get(Post, post_id)
        if post is None:
            continue

        created_before = stats.comments_created
        upsert_comment(
            session,
            post,
            {
                "reddit_id": comment_id,
                "author": item.get("author"),
                "parent_reddit_id": item.get("parent_id", ""),
                "body": normalize_text(item.get("body", "")),
                "score": int(item.get("score") or 0),
                "created_utc": parse_arctic_datetime(item["created_utc"]),
            },
            stats,
        )
        if stats.comments_created > created_before:
            comment_counts[post_id] = comment_counts.get(post_id, 0) + 1
            if comment_limit > 0 and comment_counts[post_id] >= comment_limit:
                remaining_post_ids.discard(post_id)

        if (stats.comments_created + stats.comments_updated) % 500 == 0:
            session.commit()

    session.commit()
    return stats



def import_posts_jsonl(session: Session, path: str, subreddit_name: str) -> IngestStats:
    subreddit = get_or_create_subreddit(session, normalize_subreddit_name(subreddit_name))
    stats = IngestStats()
    input_path = Path(path)
    for line in tqdm(input_path.read_text().splitlines(), desc="import:posts"):
        if not line.strip():
            continue
        item = json.loads(line)
        created_utc = parse_arctic_datetime(item["created_utc"])
        upsert_post(
            session,
            subreddit,
            {
                "reddit_id": item["id"],
                "author": item.get("author"),
                "title": normalize_text(item.get("title", "")),
                "selftext": normalize_text(item.get("selftext", "")),
                "url": item.get("url", ""),
                "score": int(item.get("score", 0)),
                "num_comments": int(item.get("num_comments", 0)),
                "created_utc": created_utc,
                "permalink": item.get("permalink", ""),
                "listing_source": "archive",
            },
            stats,
        )
    session.commit()
    return stats



def import_comments_jsonl(session: Session, path: str) -> IngestStats:
    stats = IngestStats()
    post_map = {reddit_id: post_id for reddit_id, post_id in session.execute(select(Post.reddit_id, Post.id))}
    input_path = Path(path)
    for line in tqdm(input_path.read_text().splitlines(), desc="import:comments"):
        if not line.strip():
            continue
        item = json.loads(line)
        link_id = strip_fullname(item.get("link_id"), "t3_")
        post_id = post_map.get(link_id)
        if not post_id:
            continue
        post = session.get(Post, post_id)
        created_utc = parse_arctic_datetime(item["created_utc"])
        upsert_comment(
            session,
            post,
            {
                "reddit_id": item["id"],
                "author": item.get("author"),
                "parent_reddit_id": item.get("parent_id", ""),
                "body": normalize_text(item.get("body", "")),
                "score": int(item.get("score", 0)),
                "created_utc": created_utc,
            },
            stats,
        )
    session.commit()
    return stats
