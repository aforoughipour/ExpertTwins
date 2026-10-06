"""The network layer, and the switch that turns it off.

Two worlds, and nothing in between:

* **Connected** (today) — acquisition runs, documents flow into the library.
* **Air-gapped** (a compute cluster with no egress) — ``EXPERTWINS_OFFLINE=1``. Every
  network path raises :class:`OfflineError` rather than timing out, retrying,
  or degrading into a plausible-looking empty result.

The offline mode is a *supported configuration*, not a failure mode. Keeping
online acquisition and offline reading separate makes secure-cluster use a copy
operation rather than a port.

Everything here is also cached on disk. Re-running acquisition must be cheap,
because it will be re-run many times, and because a cache turns a rate-limited
API into a local one for the second pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .errors import Absence, OfflineError

# Identify ourselves. Crossref, Unpaywall and NCBI all ask for a contact
# address and give better rate limits to requests that provide one; sending an
# anonymous flood at a public science API is bad manners and gets you blocked.
#
# SET EXPERTWINS_CONTACT TO YOUR OWN ADDRESS before running acquisition. The default
# below is a placeholder and is not a mailbox anyone reads.
CONTACT = os.environ.get("EXPERTWINS_CONTACT", "nobody@example.org")
USER_AGENT = f"ExpertTwins/1.0 (literature acquisition; mailto:{CONTACT})"

# Per-host minimum seconds between requests. NCBI's documented ceiling is 3/s
# without an API key and 10/s with one; the rest are courtesy values chosen to
# stay far below anything that would get us throttled overnight.
_DEFAULT_DELAYS = {
    "eutils.ncbi.nlm.nih.gov": 0.34,
    "www.ebi.ac.uk": 0.15,
    "api.crossref.org": 0.12,
    "api.unpaywall.org": 0.12,
    "api.openalex.org": 0.12,
    "export.arxiv.org": 3.10,
    "arxiv.org": 3.10,
    "clinicaltrials.gov": 0.25,
}
_FALLBACK_DELAY = 0.5


def is_offline() -> bool:
    return os.environ.get("EXPERTWINS_OFFLINE", "").strip() in {"1", "true", "yes", "on"}


def require_online(what: str) -> None:
    """Guard placed at the top of every function that touches the network."""
    if is_offline():
        raise OfflineError(
            f"{what} requires network access but EXPERTWINS_OFFLINE is set. "
            "This is the air-gapped configuration: the library read path works, "
            "acquisition does not."
        )


class _HostThrottle:
    """Per-host politeness, safe across threads."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        delay = _DEFAULT_DELAYS.get(host, _FALLBACK_DELAY)
        with self._lock:
            previous = self._last.get(host, 0.0)
            now = time.monotonic()
            sleep_for = previous + delay - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._last[host] = now


_THROTTLE = _HostThrottle()


@dataclass
class Response:
    """A fetch result. ``ok`` is explicit so callers cannot skip checking."""

    ok: bool
    status: int
    content: bytes
    url: str
    from_cache: bool = False
    error: str = ""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class Fetcher:
    """Cached, throttled, retrying HTTP client.

    Deliberately small. It does one thing: get bytes from a URL without being
    rude, without repeating work, and without ever silently returning an empty
    body that a caller could mistake for a real answer.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout: int = 60,
        max_retries: int = 4,
        ncbi_api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.ncbi_api_key = ncbi_api_key or os.environ.get("NCBI_API_KEY") or None
        if self.ncbi_api_key:
            # With a key NCBI permits 10 requests/second.
            _DEFAULT_DELAYS["eutils.ncbi.nlm.nih.gov"] = 0.11
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self.stats: dict[str, int] = {
            "requests": 0, "cache_hits": 0, "retries": 0,
            "failures": 0, "bytes": 0,
        }

    # -- caching ----------------------------------------------------------

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "|" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.bin"

    # -- fetching ---------------------------------------------------------

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
        accept: str = "",
    ) -> Response:
        require_online(f"GET {url}")

        params = dict(params or {})
        host = urlparse(url).netloc
        if host == "eutils.ncbi.nlm.nih.gov" and self.ncbi_api_key:
            params.setdefault("api_key", self.ncbi_api_key)
            params.setdefault("email", CONTACT)
            params.setdefault("tool", "expertwins")

        cache_path = self._cache_path(url, params)
        if use_cache and cache_path.exists():
            self.stats["cache_hits"] += 1
            return Response(
                ok=True, status=200, content=cache_path.read_bytes(),
                url=url, from_cache=True,
            )

        headers = {"Accept": accept} if accept else {}
        last_error = ""
        for attempt in range(self.max_retries):
            _THROTTLE.wait(host)
            self.stats["requests"] += 1
            try:
                resp = self._session.get(
                    url, params=params, timeout=self.timeout, headers=headers
                )
            except requests.RequestException as exc:
                # Transport-level failure. Retryable, and never confusable with
                # a successful empty response.
                last_error = f"{type(exc).__name__}: {exc}"
                self.stats["retries"] += 1
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue

            if resp.status_code == 200:
                content = resp.content
                self.stats["bytes"] += len(content)
                if use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(content)
                return Response(ok=True, status=200, content=content, url=resp.url)

            if resp.status_code in (429, 500, 502, 503, 504):
                # Back off and try again — these are the provider asking us to
                # slow down, not telling us the document is absent.
                last_error = f"HTTP {resp.status_code}"
                self.stats["retries"] += 1
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 60))
                else:
                    time.sleep(min(2 ** attempt + random.random(), 30))
                continue

            # 404/403/406 etc: a definite answer. Not retried.
            self.stats["failures"] += 1
            return Response(
                ok=False, status=resp.status_code, content=resp.content,
                url=resp.url, error=f"HTTP {resp.status_code}",
            )

        self.stats["failures"] += 1
        return Response(ok=False, status=0, content=b"", url=url, error=last_error)

    def get_json(
        self, url: str, params: dict[str, Any] | None = None, **kw: Any
    ) -> Any | Absence:
        """JSON fetch that returns a typed :class:`Absence` rather than ``None``.

        The caller cannot accidentally treat "the server refused" as "there is
        nothing there" — they are different types with different truthiness.
        """
        resp = self.get(url, params, accept="application/json", **kw)
        if not resp.ok:
            return Absence(
                what="JSON response", method=f"GET {urlparse(url).netloc}",
                query=json.dumps(params or {}, sort_keys=True)[:400],
                examined=0, detail={"status": resp.status, "error": resp.error},
            )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            return Absence(
                what="parseable JSON", method=f"GET {urlparse(url).netloc}",
                query=json.dumps(params or {}, sort_keys=True)[:400],
                examined=len(resp.content), detail={"json_error": str(exc)},
            )
