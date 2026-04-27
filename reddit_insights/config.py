from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PART2_DIR = DATA_DIR / "part2"
PART2_EVAL_DIR = PART2_DIR / "eval"
PART2_REPORTS_DIR = PART2_DIR / "reports"
RAG_INDEX_DIR = PART2_DIR / "rag_index"

load_dotenv(PROJECT_ROOT / ".env")


def normalize_subreddit_name(name: str) -> str:
    cleaned = (name or "").strip()
    if cleaned.lower().startswith("r/"):
        cleaned = cleaned[2:]
    return cleaned


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'reddit_insights.db').as_posix()}")
    arctic_shift_base_url: str = os.getenv("ARCTIC_SHIFT_BASE_URL", "https://arctic-shift.photon-reddit.com")
    arctic_shift_request_timeout: int = int(os.getenv("ARCTIC_SHIFT_REQUEST_TIMEOUT", "30"))
    arctic_shift_user_agent: str = os.getenv("ARCTIC_SHIFT_USER_AGENT", "nlp-project-insights/0.1")
    subreddit_name: str = normalize_subreddit_name(os.getenv("SUBREDDIT_NAME", "gradadmissions"))
    topic_count: int = int(os.getenv("TOPIC_COUNT", "10"))
    min_posts: int = int(os.getenv("MIN_POSTS", "15000"))
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "180"))
    topic_embedding_model: str = os.getenv("TOPIC_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    stance_nli_model: str = os.getenv("STANCE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    generation_model: str = os.getenv("GENERATION_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    stance_batch_size: int = int(os.getenv("STANCE_BATCH_SIZE", "64"))
    stance_max_length: int = int(os.getenv("STANCE_MAX_LENGTH", "320"))
    stance_max_comment_chars: int = int(os.getenv("STANCE_MAX_COMMENT_CHARS", "700"))
    stance_confidence_threshold: float = float(os.getenv("STANCE_CONFIDENCE_THRESHOLD", "0.42"))
    stance_weak_confidence_threshold: float = float(os.getenv("STANCE_WEAK_CONFIDENCE_THRESHOLD", "0.30"))
    stance_label_margin: float = float(os.getenv("STANCE_LABEL_MARGIN", "0.08"))
    topic_profile_max_new_tokens: int = int(os.getenv("TOPIC_PROFILE_MAX_NEW_TOKENS", "96"))
    summary_max_new_tokens: int = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "192"))
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    rag_chunk_words: int = int(os.getenv("RAG_CHUNK_WORDS", "180"))
    rag_chunk_overlap_words: int = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "40"))
    rag_min_text_chars: int = int(os.getenv("RAG_MIN_TEXT_CHARS", "40"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "8"))
    rag_max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "9000"))
    llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "90"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    qa_bertscore_model: str = os.getenv("QA_BERTSCORE_MODEL", "roberta-large")
    hindi_bertscore_model: str = os.getenv("HINDI_BERTSCORE_MODEL", "xlm-roberta-base")


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PART2_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PART2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
