from __future__ import annotations

import argparse

from reddit_insights.analysis import run_analysis
from reddit_insights.config import settings
from reddit_insights.db import SessionLocal, engine
from reddit_insights.ingest import import_comments_jsonl, import_posts_jsonl, scrape_subreddit
from reddit_insights.models import Base



def cmd_init_db(_: argparse.Namespace) -> None:
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")



def cmd_ingest(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        stats = scrape_subreddit(
            session=session,
            subreddit_name=args.subreddit,
            days=args.days,
            post_target=args.post_target,
            comment_limit=args.comment_limit,
        )
    print(vars(stats))



def cmd_import_posts(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        stats = import_posts_jsonl(session, args.path, args.subreddit)
    print(vars(stats))



def cmd_import_comments(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        stats = import_comments_jsonl(session, args.path)
    print(vars(stats))



def cmd_analyze(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        summary = run_analysis(session, args.topic_count, args.subreddit)
    print(summary)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit topic and stance explorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create database tables")
    init_db.set_defaults(func=cmd_init_db)

    ingest = subparsers.add_parser("ingest", help="Load Reddit posts and comments from Arctic Shift")
    ingest.add_argument("--subreddit", default=settings.subreddit_name)
    ingest.add_argument("--days", type=int, default=settings.lookback_days)
    ingest.add_argument("--post-target", type=int, default=settings.min_posts)
    ingest.add_argument("--comment-limit", type=int, default=20)
    ingest.set_defaults(func=cmd_ingest)

    import_posts = subparsers.add_parser("import-posts", help="Import posts from a JSONL archive")
    import_posts.add_argument("--path", required=True)
    import_posts.add_argument("--subreddit", default=settings.subreddit_name)
    import_posts.set_defaults(func=cmd_import_posts)

    import_comments = subparsers.add_parser("import-comments", help="Import comments from a JSONL archive")
    import_comments.add_argument("--path", required=True)
    import_comments.set_defaults(func=cmd_import_comments)

    analyze = subparsers.add_parser("analyze", help="Compute topics, time-series labels, and stance summaries")
    analyze.add_argument("--topic-count", type=int, default=settings.topic_count)
    analyze.add_argument("--subreddit", default=settings.subreddit_name)
    analyze.set_defaults(func=cmd_analyze)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
