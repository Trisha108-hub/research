"""Polite, shared rate-limit handling for public paper APIs."""

from __future__ import annotations

import threading
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import requests

_locks = {
    "openalex": threading.Lock(),
    "semantic_scholar": threading.Lock(),
    "crossref": threading.Lock(),
}
_last_request: dict[str, float] = {}


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def polite_get(
    service: str,
    url: str,
    *,
    params: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> requests.Response:
    """Serialize each provider's requests and retry temporary rate/server errors."""
    minimum_interval = 1.1 if service in {"semantic_scholar", "crossref"} else 0.2
    lock = _locks[service]
    with lock:
        for attempt in range(3):
            wait = minimum_interval - (time.monotonic() - _last_request.get(service, 0.0))
            if wait > 0:
                time.sleep(wait)

            response = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
            _last_request[service] = time.monotonic()
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                return response

            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            default_delay = 2.0 if service == "semantic_scholar" else 1.0
            delay = retry_after if retry_after is not None else default_delay * (2**attempt)
            # A daily quota reset is better surfaced than holding a web request open for minutes.
            if delay > 15:
                return response
            time.sleep(delay)

    return response
