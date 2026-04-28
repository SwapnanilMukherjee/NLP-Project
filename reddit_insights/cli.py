from __future__ import annotations

import argparse

from reddit_insights.analysis import run_analysis
from reddit_insights.corpus_qa import answer_query, build_post_meta_cache
from reddit_insights.config import settings
from reddit_insights.db import SessionLocal, engine
from reddit_insights.ingest import import_comments_jsonl, import_posts_jsonl, scrape_subreddit
from reddit_insights.llm_providers import build_provider
from reddit_insights.models import Base
from reddit_insights.part2_datasets import ensure_part2_eval_files
from reddit_insights.part2_eval import (
    evaluate_hindi_suite,
    evaluate_qa,
    write_bias_note,
    write_ethics_note,
    write_part2_report,
)
from reddit_insights.report_tex import write_final_project_report_tex
from reddit_insights.rag import RagIndex, build_rag_index



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


def cmd_part2_init_eval(args: argparse.Namespace) -> None:
    files = ensure_part2_eval_files(overwrite=args.overwrite)
    print({key: str(path) for key, path in files.items()})


def cmd_part2_build_index(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        summary = build_rag_index(
            session=session,
            subreddit_name=args.subreddit,
            include_comments=not args.posts_only,
            limit=args.limit,
            force=args.force,
        )
    print(summary)


def cmd_part2_build_meta_cache(args: argparse.Namespace) -> None:
    print(build_post_meta_cache(force=args.force))


def cmd_part2_ask(args: argparse.Namespace) -> None:
    provider = build_provider(args.provider)
    result = answer_query(args.query, provider=provider, top_k=args.top_k)
    print(f"Mode: {result.mode}")
    print(result.answer)
    if result.meta:
        print("\nMatched example posts:")
        for example in result.meta.examples:
            print(
                f"- post_id={example.post_id} score={example.score} topic={example.topic_index} "
                f"title={example.title[:120]}"
            )
    if result.rag:
        print("\nSources:")
        for item in result.rag.retrieved:
            doc = item.document
            print(f"[{item.rank}] {doc.source_type} {doc.source_id} score={item.score:.3f} topic={doc.topic_label} title={doc.title[:120]}")


def cmd_part2_evaluate_qa(args: argparse.Namespace) -> None:
    outputs = evaluate_qa(
        provider_name_list=args.providers,
        limit=args.limit,
        top_k=args.top_k,
        skip_bertscore=args.skip_bertscore,
    )
    print({key: str(path) for key, path in outputs.items()})


def cmd_part2_evaluate_translation(args: argparse.Namespace) -> None:
    outputs = evaluate_hindi_suite(
        provider_name_list=args.providers,
        limit_per_task=args.limit,
        top_k=args.top_k,
        skip_bertscore=args.skip_bertscore,
    )
    print({key: str(path) for key, path in outputs.items()})


def cmd_part2_write_notes(args: argparse.Namespace) -> None:
    paths = {}
    if not args.ethics_only:
        paths["bias"] = str(write_bias_note(provider_name_list=args.providers, top_k=args.top_k))
    if not args.bias_only:
        with SessionLocal() as session:
            paths["ethics"] = str(write_ethics_note(session))
    paths["report"] = str(write_part2_report())
    print(paths)


def cmd_part2_report(_: argparse.Namespace) -> None:
    print(str(write_part2_report()))


def cmd_final_report_tex(_: argparse.Namespace) -> None:
    print(str(write_final_project_report_tex()))


def cmd_part2_run_all(args: argparse.Namespace) -> None:
    ensure_part2_eval_files(overwrite=False)
    with SessionLocal() as session:
        index_summary = build_rag_index(
            session=session,
            subreddit_name=args.subreddit,
            include_comments=True,
            limit=args.index_limit,
            force=args.force_index,
        )
        ethics_path = write_ethics_note(session)
    qa_paths = evaluate_qa(
        provider_name_list=args.providers,
        limit=args.limit,
        top_k=args.top_k,
        skip_bertscore=args.skip_bertscore,
    )
    translation_paths = evaluate_hindi_suite(
        provider_name_list=args.providers,
        limit_per_task=args.limit,
        top_k=args.top_k,
        skip_bertscore=args.skip_bertscore,
    )
    bias_path = write_bias_note(provider_name_list=args.providers, top_k=args.top_k)
    report_path = write_part2_report()
    latex_report_path = write_final_project_report_tex()
    print(
        {
            "index": index_summary,
            "qa": {key: str(path) for key, path in qa_paths.items()},
            "hindi_suite": {key: str(path) for key, path in translation_paths.items()},
            "bias": str(bias_path),
            "ethics": str(ethics_path),
            "report": str(report_path),
            "latex_report": str(latex_report_path),
        }
    )



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

    part2_init = subparsers.add_parser("part2-init-eval", help="Write the Part 2 QA, Hindi multi-format evaluation, and bias probe reference files")
    part2_init.add_argument("--overwrite", action="store_true")
    part2_init.set_defaults(func=cmd_part2_init_eval)

    part2_index = subparsers.add_parser("part2-build-index", help="Build the persisted RAG vector index over posts and comments")
    part2_index.add_argument("--subreddit", default=settings.subreddit_name)
    part2_index.add_argument("--limit", type=int, default=None, help="Optional document limit for smoke tests")
    part2_index.add_argument("--posts-only", action="store_true", help="Index posts only, excluding comments")
    part2_index.add_argument("--force", action="store_true", help="Rebuild even if an index already exists")
    part2_index.set_defaults(func=cmd_part2_build_index)

    part2_meta = subparsers.add_parser("part2-build-meta-cache", help="Build the cached post-level metadata and embeddings for corpus analytics")
    part2_meta.add_argument("--force", action="store_true", help="Rebuild the cached post metadata and embeddings")
    part2_meta.set_defaults(func=cmd_part2_build_meta_cache)

    part2_ask = subparsers.add_parser("part2-ask", help="Ask a RAG question using one API provider")
    part2_ask.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    part2_ask.add_argument("--query", required=True)
    part2_ask.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_ask.set_defaults(func=cmd_part2_ask)

    part2_qa = subparsers.add_parser("part2-evaluate-qa", help="Run the expanded RAG QA evaluation set")
    part2_qa.add_argument("--providers", nargs="+", choices=["groq", "gemini"], default=["groq", "gemini"])
    part2_qa.add_argument("--limit", type=int, default=None)
    part2_qa.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_qa.add_argument("--skip-bertscore", action="store_true", help="Only for quick smoke tests; omit for assignment results")
    part2_qa.set_defaults(func=cmd_part2_evaluate_qa)

    part2_translation = subparsers.add_parser("part2-evaluate-translation", help="Run the full Hindi suite: translation, cross-lingual QA, summarisation, and Hinglish normalization")
    part2_translation.add_argument("--providers", nargs="+", choices=["groq", "gemini"], default=["groq", "gemini"])
    part2_translation.add_argument("--limit", type=int, default=None, help="Optional per-task example limit for smoke tests")
    part2_translation.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_translation.add_argument("--skip-bertscore", action="store_true", help="Only for quick smoke tests; omit for assignment results")
    part2_translation.set_defaults(func=cmd_part2_evaluate_translation)

    part2_hindi = subparsers.add_parser("part2-evaluate-hindi", help="Alias for the full Hindi multi-format evaluation suite")
    part2_hindi.add_argument("--providers", nargs="+", choices=["groq", "gemini"], default=["groq", "gemini"])
    part2_hindi.add_argument("--limit", type=int, default=None, help="Optional per-task example limit for smoke tests")
    part2_hindi.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_hindi.add_argument("--skip-bertscore", action="store_true", help="Only for quick smoke tests; omit for assignment results")
    part2_hindi.set_defaults(func=cmd_part2_evaluate_translation)

    part2_notes = subparsers.add_parser("part2-write-notes", help="Generate bias and ethics notes")
    part2_notes.add_argument("--providers", nargs="+", choices=["groq", "gemini"], default=["groq", "gemini"])
    part2_notes.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_notes.add_argument("--bias-only", action="store_true")
    part2_notes.add_argument("--ethics-only", action="store_true")
    part2_notes.set_defaults(func=cmd_part2_write_notes)

    part2_report = subparsers.add_parser("part2-report", help="Combine generated Part 2 sections into one markdown report")
    part2_report.set_defaults(func=cmd_part2_report)

    final_report = subparsers.add_parser("final-report-tex", help="Generate the unified Part 1 + Part 2 LaTeX report for Overleaf")
    final_report.set_defaults(func=cmd_final_report_tex)

    part2_all = subparsers.add_parser("part2-run-all", help="Build index and run all Part 2 API evaluations and notes")
    part2_all.add_argument("--subreddit", default=settings.subreddit_name)
    part2_all.add_argument("--providers", nargs="+", choices=["groq", "gemini"], default=["groq", "gemini"])
    part2_all.add_argument("--limit", type=int, default=None, help="Optional per-task eval limit for smoke tests")
    part2_all.add_argument("--index-limit", type=int, default=None, help="Optional RAG document limit for smoke tests")
    part2_all.add_argument("--top-k", type=int, default=settings.rag_top_k)
    part2_all.add_argument("--force-index", action="store_true")
    part2_all.add_argument("--skip-bertscore", action="store_true", help="Only for quick smoke tests; omit for assignment results")
    part2_all.set_defaults(func=cmd_part2_run_all)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
