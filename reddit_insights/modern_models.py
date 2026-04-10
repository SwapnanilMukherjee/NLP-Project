from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import re
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from reddit_insights.config import settings


LABEL_RE = re.compile(r"^LABEL:\s*(.+)$", re.MULTILINE)
VIEWPOINT_RE = re.compile(r"^VIEWPOINT:\s*(.+)$", re.MULTILINE)
CLAIM_RE = re.compile(r"^CLAIM_(\d+):\s*(.+)$", re.MULTILINE)

TOPIC_PROFILE_SYSTEM = (
    "You analyze Reddit discussion clusters and create topic metadata for downstream stance analysis. "
    "Produce short labels and one broad debatable claim that many commenters could plausibly agree with or disagree with. "
    "Follow the requested output format exactly."
)

VIEWPOINT_REPAIR_SYSTEM = (
    "You rewrite discussion summaries into broad debatable claims for stance analysis. "
    "If no reusable claim exists, return insufficient_evidence."
)

VIEWPOINT_CANDIDATE_SYSTEM = (
    "You propose evidence-grounded debate claims for Reddit discussion clusters. "
    "Good claims are concrete propositions that a commenter can directly agree with or disagree with. "
    "Bad claims just summarize emotions, status updates, or the fact that people are discussing something."
)

SUMMARY_SYSTEM = (
    "You summarize discussion arguments faithfully. "
    "Write concise summaries grounded only in the provided comments."
)

META_VIEWPOINT_PREFIXES = (
    "the dominant viewpoint",
    "the majority of",
    "the majority",
    "the discussion",
    "the conversation",
    "the thread",
    "the dominant sentiment",
    "the main sentiment",
    "participants",
    "respondents",
    "commenters",
)

SUPPORT_HYPOTHESIS_TEMPLATE = "The author agrees with the claim: {viewpoint}"
OPPOSE_HYPOTHESIS_TEMPLATE = "The author disagrees with the claim: {viewpoint}"

CLAIM_EXAMPLES = (
    "GOOD: Research experience matters more than GPA in PhD admissions.\n"
    "GOOD: Universities should communicate admissions decisions earlier and more transparently.\n"
    "GOOD: Unfunded graduate offers are often not worth the cost.\n"
    "BAD: Applicants are frustrated while waiting for decisions.\n"
    "BAD: Acceptance brings relief and rejection causes stress.\n"
    "BAD: People are discussing interviews and updates."
)

EMOTIONAL_NARRATION_PATTERNS = (
    "provides relief",
    "lead to stress and uncertainty",
    "leads to stress and uncertainty",
    "emotional impact",
    "excitement and gratitude",
    "gratitude as individuals share",
    "self-doubt",
    "mixed reactions",
)

GENERIC_ADVICE_PREFIXES = (
    "it is crucial",
    "applicants should",
    "students should",
    "candidates should",
    "one should",
)

UNGROUNDED_TREND_PATTERNS = (
    "success rate",
    "is declining",
    "rare and significant achievement",
)

DOMAIN_TERMS = {
    "admission",
    "admissions",
    "application",
    "applications",
    "interview",
    "interviews",
    "program",
    "programs",
    "funding",
    "funded",
    "unfunded",
    "tuition",
    "cost",
    "costs",
    "gpa",
    "research",
    "letters",
    "recommendation",
    "recommenders",
    "deadline",
    "deadlines",
    "communication",
    "offer",
    "offers",
    "waitlist",
    "waitlists",
    "acceptance",
    "rejection",
}


@dataclass(frozen=True)
class TopicProfile:
    label: str
    dominant_viewpoint: str


def truncate_text(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def dedupe_preserve_order(items: Sequence[str], limit: int, max_chars: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = truncate_text(item, max_chars)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
        if len(result) >= limit:
            break
    return result


def looks_like_meta_viewpoint(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered or lowered == "insufficient_evidence":
        return False
    if any(lowered.startswith(prefix) for prefix in META_VIEWPOINT_PREFIXES):
        return True
    return any(token in lowered for token in ("the discussion", "participants", "respondents", "commenters"))


def normalized_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", (text or "").lower()) if len(term) >= 3}


def keyword_overlap_count(viewpoint: str, keywords: Sequence[str]) -> int:
    viewpoint_terms = normalized_terms(viewpoint)
    keyword_terms: set[str] = set()
    for keyword in keywords:
        keyword_terms.update(normalized_terms(keyword))
    return len(viewpoint_terms & keyword_terms)


def claim_quality_issues(viewpoint: str, keywords: Sequence[str]) -> list[str]:
    named_tokens = re.findall(r"\b[A-Z][A-Za-z0-9&./-]{1,}\b", viewpoint or "")
    lowered = (viewpoint or "").strip().lower()
    if not lowered or lowered == "insufficient_evidence":
        return []

    issues: list[str] = []
    if looks_like_meta_viewpoint(lowered):
        issues.append("meta discussion summary")
    if any(pattern in lowered for pattern in EMOTIONAL_NARRATION_PATTERNS):
        issues.append("emotional narration instead of a debatable claim")
    if any(lowered.startswith(prefix) for prefix in GENERIC_ADVICE_PREFIXES):
        issues.append("generic advice framing instead of a concrete disputed proposition")
    if any(pattern in lowered for pattern in UNGROUNDED_TREND_PATTERNS):
        issues.append("unsupported trend or status claim")
    if len(named_tokens) >= 4:
        issues.append("specific school or program examples instead of a broad reusable claim")
    if keyword_overlap_count(lowered, keywords) == 0 and not (normalized_terms(lowered) & DOMAIN_TERMS):
        issues.append("weak grounding in the cluster evidence")
    return issues


class ModernNLPStack:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def embedder(self) -> SentenceTransformer:
        model = SentenceTransformer(settings.topic_embedding_model, device=self.device)
        model.eval()
        return model

    @cached_property
    def nli_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(settings.stance_nli_model)

    @cached_property
    def nli_model(self) -> AutoModelForSequenceClassification:
        kwargs = {}
        if self.device == "cuda":
            kwargs["dtype"] = torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(settings.stance_nli_model, **kwargs)
        if self.device == "cuda":
            model = model.to(self.device)
        model.eval()
        return model

    @cached_property
    def generator_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(settings.generation_model)

    @cached_property
    def generator_model(self) -> AutoModelForCausalLM:
        if self.device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                settings.generation_model,
                dtype=torch.float16,
                device_map="cuda",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(settings.generation_model)
            model = model.to(self.device)
        model.eval()
        return model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self.embedder.encode(
            list(texts),
            batch_size=settings.embedding_batch_size,
            show_progress_bar=len(texts) >= settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def build_topic_profile(
        self,
        keywords: Sequence[str],
        representative_posts: Sequence[str],
        representative_comments: Sequence[str],
    ) -> TopicProfile:
        fallback_label = " / ".join(keywords[:3]) if keywords else "miscellaneous discussion"
        keyword_text = ", ".join(keywords[:8]) if keywords else "none"
        post_lines = "\n".join(f"- {text}" for text in dedupe_preserve_order(representative_posts, 6, 220)) or "- none"
        comment_lines = "\n".join(f"- {text}" for text in dedupe_preserve_order(representative_comments, 8, 220)) or "- none"
        prompt = (
            "Create concise topic metadata for a Reddit discussion cluster.\n"
            "Return exactly two lines in this format:\n"
            "LABEL: <3-6 word topic label>\n"
            "VIEWPOINT: <one sentence broad debatable claim, or insufficient_evidence>\n\n"
            "Rules for VIEWPOINT:\n"
            "- Write a single broad claim that many comments in the cluster could support or oppose.\n"
            "- Prefer normative, comparative, or causal claims.\n"
            "- Ground the claim in recurring evidence from multiple posts or comments, not a one-off detail.\n"
            "- Avoid describing the discussion, emotions, or mixed reactions.\n"
            "- Do not mention participants, commenters, the thread, or Reddit.\n"
            "- If the cluster is mostly status updates, congratulations, or personal narratives with no recurring debatable claim, write insufficient_evidence.\n\n"
            f"Examples:\n{CLAIM_EXAMPLES}\n\n"
            f"Keywords: {keyword_text}\n"
            f"Representative posts:\n{post_lines}\n\n"
            f"Representative comments:\n{comment_lines}"
        )
        response = self.generate_text(TOPIC_PROFILE_SYSTEM, prompt, settings.topic_profile_max_new_tokens)
        label_match = LABEL_RE.search(response)
        viewpoint_match = VIEWPOINT_RE.search(response)
        label = truncate_text(label_match.group(1).strip(), 80) if label_match else fallback_label
        viewpoint = viewpoint_match.group(1).strip() if viewpoint_match else "insufficient_evidence"
        if not viewpoint:
            viewpoint = "insufficient_evidence"
        candidates = [viewpoint]
        candidates.extend(self.generate_viewpoint_candidates(keyword_text, post_lines, comment_lines))
        candidates = self.prepare_viewpoint_candidates(candidates, keywords, keyword_text, post_lines, comment_lines)
        viewpoint = self.select_best_viewpoint(candidates, representative_comments, keywords)
        return TopicProfile(label=label, dominant_viewpoint=viewpoint)

    def generate_viewpoint_candidates(self, keyword_text: str, post_lines: str, comment_lines: str) -> list[str]:
        prompt = (
            "Propose candidate debate claims for this Reddit discussion cluster.\n"
            "Return exactly three lines in this format:\n"
            "CLAIM_1: <one sentence claim or insufficient_evidence>\n"
            "CLAIM_2: <one sentence claim or insufficient_evidence>\n"
            "CLAIM_3: <one sentence claim or insufficient_evidence>\n\n"
            "Rules:\n"
            "- Each claim must be a proposition that a commenter could directly agree with or disagree with.\n"
            "- Prefer admissions criteria, communication, funding, cost, recommendation letters, interviews, fairness, or applicant strategy if supported by the evidence.\n"
            "- Do not output emotional summaries, congratulations, or generic waiting/status updates.\n"
            "- If the cluster has no reusable debate claim, use insufficient_evidence.\n\n"
            f"Examples:\n{CLAIM_EXAMPLES}\n\n"
            f"Keywords: {keyword_text}\n"
            f"Representative posts:\n{post_lines}\n\n"
            f"Representative comments:\n{comment_lines}"
        )
        response = self.generate_text(VIEWPOINT_CANDIDATE_SYSTEM, prompt, 120)
        candidates = [match[1].strip() for match in CLAIM_RE.findall(response)]
        if not candidates:
            return []
        return [truncate_text(candidate, 180) for candidate in dedupe_preserve_order(candidates, 3, 180)]

    def prepare_viewpoint_candidates(
        self,
        candidates: Sequence[str],
        keywords: Sequence[str],
        keyword_text: str,
        post_lines: str,
        comment_lines: str,
    ) -> list[str]:
        prepared: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            current = truncate_text(candidate, 180)
            for _ in range(2):
                issues = claim_quality_issues(current, keywords)
                if not issues:
                    break
                current = self.repair_viewpoint(current, keyword_text, post_lines, comment_lines, issues)
                if current == "insufficient_evidence":
                    break
            key = current.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            prepared.append(current)
        if "insufficient_evidence" not in seen:
            prepared.append("insufficient_evidence")
        return prepared

    def repair_viewpoint(
        self,
        viewpoint: str,
        keyword_text: str,
        post_lines: str,
        comment_lines: str,
        issues: Sequence[str],
    ) -> str:
        issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- weak claim quality"
        prompt = (
            "Rewrite the statement below as one broad debatable claim for stance analysis.\n"
            "If no reusable debatable claim exists, output exactly: insufficient_evidence\n\n"
            "Rules:\n"
            "- Keep it to one sentence.\n"
            "- Prefer a claim a commenter could agree with or disagree with directly.\n"
            "- Ground the claim in repeated evidence from the posts/comments below.\n"
            "- Do not mention the discussion, participants, commenters, emotions, or mixed reactions.\n\n"
            f"Examples:\n{CLAIM_EXAMPLES}\n\n"
            f"Problems to fix:\n{issue_lines}\n\n"
            f"Keywords: {keyword_text}\n"
            f"Representative posts:\n{post_lines}\n\n"
            f"Representative comments:\n{comment_lines}\n\n"
            f"Original statement: {viewpoint}"
        )
        repaired = self.generate_text(VIEWPOINT_REPAIR_SYSTEM, prompt, 64).splitlines()[0].strip()
        repaired = repaired.removeprefix("VIEWPOINT:").strip()
        if not repaired:
            return "insufficient_evidence"
        return truncate_text(repaired, 180)

    def select_best_viewpoint(
        self,
        candidates: Sequence[str],
        representative_comments: Sequence[str],
        keywords: Sequence[str],
    ) -> str:
        evaluation_comments = dedupe_preserve_order(representative_comments, 12, 260)
        best_candidate = "insufficient_evidence"
        best_score = float("-inf")
        for candidate in candidates:
            score = self.score_viewpoint_candidate(candidate, evaluation_comments, keywords)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate != "insufficient_evidence" and best_score < 0.28:
            return "insufficient_evidence"
        return best_candidate

    def score_viewpoint_candidate(
        self,
        candidate: str,
        evaluation_comments: Sequence[str],
        keywords: Sequence[str],
    ) -> float:
        if not candidate or candidate == "insufficient_evidence":
            return -0.1

        issues = claim_quality_issues(candidate, keywords)
        issue_penalty = 0.0
        for issue in issues:
            if "specific school or program examples" in issue:
                issue_penalty += 0.24
            elif "generic advice framing" in issue:
                issue_penalty += 0.16
            else:
                issue_penalty += 0.12
        overlap_bonus = min(0.15, keyword_overlap_count(candidate, keywords) * 0.03)
        if not evaluation_comments:
            return overlap_bonus - issue_penalty

        predictions = self.classify_comment_stances(evaluation_comments, candidate)
        counts = Counter(label for label, _, _ in predictions)
        explicit_total = counts["support"] + counts["oppose"]
        explicit_ratio = explicit_total / len(predictions)
        confidence_bonus = 0.0
        if explicit_total:
            confidence_bonus = 0.35 * float(
                np.mean([confidence for label, confidence, _ in predictions if label != "neutral"])
            )
        balance_bonus = 0.0
        if explicit_total:
            balance_bonus = 0.1 * min(counts["support"], counts["oppose"]) / explicit_total
        return explicit_ratio + confidence_bonus + overlap_bonus + balance_bonus - issue_penalty

    def classify_comment_stances(self, comments: Sequence[str], viewpoint: str) -> list[tuple[str, float, str]]:
        if not comments:
            return []
        if not viewpoint or viewpoint == "insufficient_evidence":
            return [("neutral", 0.0, "nli:insufficient_evidence")] * len(comments)

        model = self.nli_model
        tokenizer = self.nli_tokenizer
        id2label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
        entailment_id = next((idx for idx, label in id2label.items() if "entail" in label), None)
        neutral_id = next((idx for idx, label in id2label.items() if "neutral" in label), None)
        if entailment_id is None:
            raise RuntimeError(f"Could not find entailment label in {id2label!r}")

        support_hypothesis = SUPPORT_HYPOTHESIS_TEMPLATE.format(viewpoint=viewpoint)
        oppose_hypothesis = OPPOSE_HYPOTHESIS_TEMPLATE.format(viewpoint=viewpoint)
        predictions: list[tuple[str, float, str]] = []

        iterator = range(0, len(comments), settings.stance_batch_size)
        for start in tqdm(iterator, desc="stance:nli", disable=len(comments) < settings.stance_batch_size * 4):
            batch_comments = [truncate_text(text, settings.stance_max_comment_chars) for text in comments[start : start + settings.stance_batch_size]]
            premises = batch_comments + batch_comments
            hypotheses = [support_hypothesis] * len(batch_comments) + [oppose_hypothesis] * len(batch_comments)
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=settings.stance_max_length,
                return_tensors="pt",
            )
            if self.device == "cuda":
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = model(**encoded).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            split = len(batch_comments)
            support_probs = probs[:split]
            oppose_probs = probs[split:]
            for support_prob, oppose_prob in zip(support_probs, oppose_probs):
                support_score = float(support_prob[entailment_id])
                oppose_score = float(oppose_prob[entailment_id])
                neutral_score = 0.0
                if neutral_id is not None:
                    neutral_score = max(float(support_prob[neutral_id]), float(oppose_prob[neutral_id]))
                directional_score = max(support_score, oppose_score)
                gap = abs(support_score - oppose_score)
                if directional_score >= settings.stance_confidence_threshold and gap >= settings.stance_label_margin:
                    stance = "support" if support_score > oppose_score else "oppose"
                    mode = "strong"
                elif directional_score >= settings.stance_weak_confidence_threshold and gap >= settings.stance_label_margin * 1.5:
                    stance = "support" if support_score > oppose_score else "oppose"
                    mode = "weak"
                else:
                    stance = "neutral"
                    mode = "neutral"
                predictions.append(
                    (
                        stance,
                        directional_score,
                        (
                            "nli:"
                            f"agree={support_score:.3f};"
                            f"disagree={oppose_score:.3f};"
                            f"neutral={neutral_score:.3f};"
                            f"gap={gap:.3f};"
                            f"mode={mode}"
                        ),
                    )
                )

        return predictions

    def summarize_side(self, topic_label: str, side_name: str, comments: Sequence[str]) -> str:
        if not comments:
            return "No clear arguments detected."
        comment_lines = "\n".join(f"- {text}" for text in dedupe_preserve_order(comments, 12, 320))
        prompt = (
            f"Summarize the key {side_name} arguments for the Reddit topic '{topic_label}'.\n"
            "Write 2-4 sentences, stay concrete, and do not invent claims that are not in the comments.\n\n"
            f"Comments:\n{comment_lines}"
        )
        summary = self.generate_text(SUMMARY_SYSTEM, prompt, settings.summary_max_new_tokens)
        return summary or "No clear arguments detected."

    def generate_text(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        tokenizer = self.generator_tokenizer
        model = self.generator_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        target_device = model.device if hasattr(model, "device") else self.device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
