from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z][a-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "reddit",
    "subreddit",
    "thread",
    "post",
    "comment",
    "comments",
    "deleted",
    "removed",
    "amp",
    "https",
    "www",
    "just",
    "like",
    "really",
    "know",
    "got",
    "getting",
    "think",
    "thinking",
    "want",
    "wanted",
    "trying",
    "try",
    "probably",
    "basically",
    "honestly",
    "actually",
    "maybe",
    "pretty",
    "quite",
    "super",
    "yeah",
    "okay",
    "ok",
    "lot",
    "lots",
    "thing",
    "things",
    "stuff",
    "does",
    "did",
    "don",
    "doesn",
    "didn",
    "isn",
    "wasn",
    "weren",
    "ve",
    "ll",
    "re",
    "im",
    "ive",
    "id",
})


def normalize_text(text: str) -> str:
    text = URL_RE.sub(" ", text or "")
    text = text.replace("\n", " ")
    text = WS_RE.sub(" ", text)
    return text.strip()


def combined_post_text(title: str, body: str) -> str:
    return normalize_text(f"{title}. {body}".strip())


def tokenize(text: str) -> list[str]:
    return [token for token in WORD_RE.findall(text.lower()) if token not in CUSTOM_STOPWORDS]


def top_terms(texts: list[str], limit: int = 15) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter.most_common(limit)


def split_sentences(text: str) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    return [sentence.strip() for sentence in SENTENCE_RE.split(clean) if sentence.strip()]
