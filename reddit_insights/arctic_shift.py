from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any, Iterator

import requests

from reddit_insights.config import normalize_subreddit_name, settings


POST_FIELDS = [
    "id",
    "author",
    "title",
    "selftext",
    "url",
    "score",
    "num_comments",
    "created_utc",
]

COMMENT_FIELDS = [
    "id",
    "author",
    "body",
    "score",
    "created_utc",
    "parent_id",
    "link_id",
]

MAX_RETRIES = 5
DEFAULT_PAGE_LIMIT = 100


def utc_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)


def parse_arctic_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        return utc_datetime(float(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Missing Arctic Shift timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def format_arctic_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    value = value.replace(microsecond=0)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def within_lookback(created_utc: datetime, lookback_days: int) -> bool:
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    return created_utc >= cutoff


def strip_fullname(value: str | None, prefix: str) -> str:
    cleaned = (value or "").strip()
    if cleaned.startswith(prefix):
        return cleaned[len(prefix):]
    return cleaned


class ArcticShiftClient:
    def __init__(self) -> None:
        self.base_url = settings.arctic_shift_base_url.rstrip("/")
        self.timeout = settings.arctic_shift_request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.arctic_shift_user_agent})

    def iter_posts(
        self,
        subreddit_name: str,
        after: datetime,
        before: datetime,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        yield from self._iter_search(
            endpoint="posts/search",
            subreddit_name=subreddit_name,
            after=after,
            before=before,
            limit=limit,
            fields=POST_FIELDS,
        )

    def iter_comments(
        self,
        subreddit_name: str,
        after: datetime,
        before: datetime,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        yield from self._iter_search(
            endpoint="comments/search",
            subreddit_name=subreddit_name,
            after=after,
            before=before,
            limit=limit,
            fields=COMMENT_FIELDS,
        )

    def _iter_search(
        self,
        endpoint: str,
        subreddit_name: str,
        after: datetime,
        before: datetime,
        limit: int,
        fields: list[str],
    ) -> Iterator[dict[str, Any]]:
        cursor = after
        seen_ids: set[str] = set()
        normalized_subreddit = normalize_subreddit_name(subreddit_name)

        while cursor < before:
            items = self._search_page(
                endpoint=endpoint,
                subreddit_name=normalized_subreddit,
                after=cursor,
                before=before,
                limit=limit,
                fields=fields,
            )
            if not items:
                break

            max_created = cursor
            yielded = 0
            for item in items:
                item_id = str(item.get("id") or "").strip()
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                yielded += 1
                item_created = parse_arctic_datetime(item["created_utc"])
                if item_created > max_created:
                    max_created = item_created
                yield item

            if yielded == 0:
                break
            if len(items) < limit:
                break

            next_cursor = max_created + timedelta(seconds=1)
            if next_cursor <= cursor:
                break
            cursor = next_cursor

    def _search_page(
        self,
        endpoint: str,
        subreddit_name: str,
        after: datetime,
        before: datetime,
        limit: int,
        fields: list[str],
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"{self.base_url}/api/{endpoint}",
            params={
                "subreddit": normalize_subreddit_name(subreddit_name),
                "after": format_arctic_datetime(after),
                "before": format_arctic_datetime(before),
                "sort": "asc",
                "limit": self._normalize_limit(limit),
                "fields": ",".join(fields),
            },
        )

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise RuntimeError(f"Unexpected Arctic Shift payload shape for {endpoint}: {type(payload)!r}")

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = RuntimeError(f"Arctic Shift returned {response.status_code} for {url}")
                    time.sleep(self._retry_delay(response, attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(min(30.0, 2.0 ** attempt))
        if last_error:
            raise last_error
        raise RuntimeError(f"Arctic Shift request failed for {url}")

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        reset_after = response.headers.get("X-RateLimit-Reset")
        if reset_after:
            try:
                return max(1.0, float(reset_after))
            except ValueError:
                pass
        return min(60.0, 2.0 ** attempt)

    def _normalize_limit(self, limit: int) -> int:
        return max(1, min(limit, 100))
