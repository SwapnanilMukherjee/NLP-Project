from __future__ import annotations

from collections import Counter
from functools import cached_property
import math
import re
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize_for_overlap(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = tokenize_for_overlap(candidate)
    ref_tokens = tokenize_for_overlap(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def character_ngrams(text: str, n: int) -> Counter[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return Counter()
    padded = f" {clean} "
    if len(padded) < n:
        return Counter([padded])
    return Counter(padded[idx : idx + n] for idx in range(len(padded) - n + 1))


def chrf(candidate: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        cand_counts = character_ngrams(candidate, order)
        ref_counts = character_ngrams(reference, order)
        overlap = sum((cand_counts & ref_counts).values())
        precisions.append(overlap / max(1, sum(cand_counts.values())))
        recalls.append(overlap / max(1, sum(ref_counts.values())))
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


class BertScoreScorer:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self):
        kwargs = {}
        if self.device == "cuda":
            kwargs["dtype"] = torch.float16
        model = AutoModel.from_pretrained(self.model_name, **kwargs)
        model = model.to(self.device)
        model.eval()
        return model

    def _text_embeddings(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text or " ",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state[0].float()
        input_ids = encoded["input_ids"][0]
        attention = encoded["attention_mask"][0].bool()
        special_ids = set(self.tokenizer.all_special_ids)
        keep = attention & torch.tensor([int(token_id) not in special_ids for token_id in input_ids], device=self.device)
        vectors = hidden[keep]
        if vectors.numel() == 0:
            vectors = hidden[attention]
        return torch.nn.functional.normalize(vectors, p=2, dim=1)

    def score_pair(self, candidate: str, reference: str) -> float:
        cand_vectors = self._text_embeddings(candidate)
        ref_vectors = self._text_embeddings(reference)
        if cand_vectors.numel() == 0 or ref_vectors.numel() == 0:
            return 0.0
        similarities = cand_vectors @ ref_vectors.T
        precision = float(similarities.max(dim=1).values.mean().item())
        recall = float(similarities.max(dim=0).values.mean().item())
        if precision + recall == 0 or math.isnan(precision) or math.isnan(recall):
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> list[float]:
        return [self.score_pair(candidate, reference) for candidate, reference in zip(candidates, references)]


def mean_or_zero(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return 0.0
    return float(np.mean(clean))
