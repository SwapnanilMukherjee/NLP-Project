from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from reddit_insights.config import RAG_INDEX_DIR, normalize_subreddit_name, settings
from reddit_insights.llm_providers import ChatProvider
from reddit_insights.models import Comment, Post, Subreddit, Topic, TopicAssignment
from reddit_insights.preprocess import combined_post_text


SPACE_RE = re.compile(r"\s+")

RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented QA assistant for a course project about r/gradadmissions. "
    "Answer using only the supplied Reddit context. Cite evidence with the bracketed source numbers, "
    "for example [1] or [2]. If the context does not contain enough evidence, say that the corpus "
    "does not provide enough information instead of guessing. For opinion questions, summarize the "
    "range of views rather than claiming a single universal consensus. Do not reveal usernames or infer "
    "private identities."
)


@dataclass(frozen=True)
class RagDocument:
    doc_id: str
    source_type: str
    source_id: str
    post_id: str
    title: str
    text: str
    created_utc: str
    score: int
    permalink: str
    topic_label: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedDocument:
    rank: int
    score: float
    document: RagDocument


@dataclass(frozen=True)
class RAGAnswer:
    provider: str
    model: str
    question: str
    answer: str
    retrieved: list[RetrievedDocument]


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def word_chunks(text: str, chunk_words: int, overlap_words: int) -> Iterable[tuple[int, str]]:
    words = clean_text(text).split()
    if not words:
        return
    if len(words) <= chunk_words:
        yield 0, " ".join(words)
        return
    step = max(1, chunk_words - overlap_words)
    chunk_index = 0
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            continue
        yield chunk_index, " ".join(chunk)
        chunk_index += 1
        if start + chunk_words >= len(words):
            break


def _topic_outerjoin(stmt):
    return stmt.outerjoin(TopicAssignment, TopicAssignment.post_id == Post.id).outerjoin(Topic, Topic.id == TopicAssignment.topic_id)


def load_rag_documents(
    session: Session,
    subreddit_name: str,
    include_comments: bool = True,
    limit: int | None = None,
) -> list[RagDocument]:
    normalized_subreddit = normalize_subreddit_name(subreddit_name)
    documents: list[RagDocument] = []
    post_stmt = _topic_outerjoin(
        select(
            Post.id,
            Post.reddit_id,
            Post.title,
            Post.selftext,
            Post.score,
            Post.created_utc,
            Post.permalink,
            Topic.label,
        ).join(Subreddit, Post.subreddit_id == Subreddit.id)
    ).where(Subreddit.name == normalized_subreddit)

    for row in session.execute(post_stmt).all():
        full_text = combined_post_text(row.title, row.selftext)
        for chunk_index, chunk in word_chunks(full_text, settings.rag_chunk_words, settings.rag_chunk_overlap_words):
            if len(chunk) < settings.rag_min_text_chars:
                continue
            documents.append(
                RagDocument(
                    doc_id=f"post:{row.reddit_id}:{chunk_index}",
                    source_type="post",
                    source_id=row.reddit_id,
                    post_id=row.reddit_id,
                    title=clean_text(row.title),
                    text=chunk,
                    created_utc=row.created_utc.isoformat(),
                    score=int(row.score or 0),
                    permalink=row.permalink or "",
                    topic_label=row.label or "unanalyzed",
                    chunk_index=chunk_index,
                )
            )
            if limit and len(documents) >= limit:
                return documents

    if not include_comments:
        return documents

    comment_stmt = _topic_outerjoin(
        select(
            Comment.id,
            Comment.reddit_id,
            Comment.body,
            Comment.score,
            Comment.created_utc,
            Post.id.label("post_pk"),
            Post.reddit_id.label("post_reddit_id"),
            Post.title,
            Post.permalink,
            Topic.label,
        )
        .join(Post, Comment.post_id == Post.id)
        .join(Subreddit, Post.subreddit_id == Subreddit.id)
    ).where(Subreddit.name == normalized_subreddit)

    for row in session.execute(comment_stmt).all():
        for chunk_index, chunk in word_chunks(row.body, settings.rag_chunk_words, settings.rag_chunk_overlap_words):
            if len(chunk) < settings.rag_min_text_chars:
                continue
            documents.append(
                RagDocument(
                    doc_id=f"comment:{row.reddit_id}:{chunk_index}",
                    source_type="comment",
                    source_id=row.reddit_id,
                    post_id=row.post_reddit_id,
                    title=clean_text(row.title),
                    text=chunk,
                    created_utc=row.created_utc.isoformat(),
                    score=int(row.score or 0),
                    permalink=row.permalink or "",
                    topic_label=row.label or "unanalyzed",
                    chunk_index=chunk_index,
                )
            )
            if limit and len(documents) >= limit:
                return documents
    return documents


def _documents_path(index_dir: Path) -> Path:
    return index_dir / "documents.jsonl"


def _embeddings_path(index_dir: Path) -> Path:
    return index_dir / "embeddings.npy"


def _metadata_path(index_dir: Path) -> Path:
    return index_dir / "metadata.json"


def _load_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def build_rag_index(
    session: Session,
    subreddit_name: str,
    include_comments: bool = True,
    limit: int | None = None,
    force: bool = False,
    index_dir: Path = RAG_INDEX_DIR,
) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    docs_path = _documents_path(index_dir)
    embeddings_path = _embeddings_path(index_dir)
    metadata_path = _metadata_path(index_dir)
    if docs_path.exists() and embeddings_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        metadata["status"] = "exists"
        return metadata

    documents = load_rag_documents(session, subreddit_name, include_comments=include_comments, limit=limit)
    if not documents:
        raise RuntimeError(f"No RAG documents found for r/{normalize_subreddit_name(subreddit_name)}")

    embedder = _load_embedder(settings.rag_embedding_model)
    texts = [f"Title: {doc.title}\nTopic: {doc.topic_label}\nText: {doc.text}" for doc in documents]
    embeddings = embedder.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(texts) >= settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    with docs_path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    np.save(embeddings_path, embeddings)
    metadata = {
        "status": "created",
        "subreddit": normalize_subreddit_name(subreddit_name),
        "document_count": len(documents),
        "embedding_model": settings.rag_embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "include_comments": include_comments,
        "limit": limit,
        "chunk_words": settings.rag_chunk_words,
        "chunk_overlap_words": settings.rag_chunk_overlap_words,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class RagIndex:
    def __init__(self, index_dir: Path = RAG_INDEX_DIR) -> None:
        self.index_dir = index_dir
        self.documents = self._load_documents()
        self.embeddings = np.load(_embeddings_path(index_dir), mmap_mode="r")
        self.metadata = json.loads(_metadata_path(index_dir).read_text(encoding="utf-8"))
        if len(self.documents) != int(self.embeddings.shape[0]):
            raise RuntimeError("RAG index is inconsistent: document and embedding counts differ.")

    def _load_documents(self) -> list[RagDocument]:
        docs_path = _documents_path(self.index_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"RAG documents file not found: {docs_path}")
        documents: list[RagDocument] = []
        with docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(RagDocument(**json.loads(line)))
        return documents

    @cached_property
    def embedder(self) -> SentenceTransformer:
        return _load_embedder(self.metadata.get("embedding_model", settings.rag_embedding_model))

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        k = top_k or settings.rag_top_k
        query_embedding = self.embedder.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        scores = np.asarray(self.embeddings @ query_embedding)
        if len(scores) == 0:
            return []
        top_indices = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        ordered = top_indices[np.argsort(-scores[top_indices])]
        return [
            RetrievedDocument(rank=rank + 1, score=float(scores[idx]), document=self.documents[int(idx)])
            for rank, idx in enumerate(ordered.tolist())
        ]


def format_context(retrieved: list[RetrievedDocument], max_chars: int | None = None) -> str:
    budget = max_chars or settings.rag_max_context_chars
    lines: list[str] = []
    used = 0
    for item in retrieved:
        doc = item.document
        header = (
            f"[{item.rank}] type={doc.source_type}; score={doc.score}; date={doc.created_utc[:10]}; "
            f"topic={doc.topic_label}; title={doc.title}"
        )
        body = clean_text(doc.text)
        block = f"{header}\n{body}\n"
        if used + len(block) > budget and lines:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def answer_question(
    provider: ChatProvider,
    question: str,
    index: RagIndex | None = None,
    top_k: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> RAGAnswer:
    rag_index = index or RagIndex()
    retrieved = rag_index.search(question, top_k=top_k)
    context = format_context(retrieved)
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Reddit context:\n"
        f"{context}\n\n"
        "Write a concise answer. Include citations to the source numbers you used."
    )
    response = provider.chat(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return RAGAnswer(
        provider=response.provider,
        model=response.model,
        question=question,
        answer=response.text,
        retrieved=retrieved,
    )


def retrieved_to_dicts(retrieved: list[RetrievedDocument]) -> list[dict]:
    result = []
    for item in retrieved:
        result.append({"rank": item.rank, "score": item.score, "document": asdict(item.document)})
    return result


def iter_index_documents(index_dir: Path = RAG_INDEX_DIR) -> Iterable[RagDocument]:
    docs_path = _documents_path(index_dir)
    with docs_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="rag:documents"):
            if line.strip():
                yield RagDocument(**json.loads(line))
