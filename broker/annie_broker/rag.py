"""WeKnora retrieval gateway: one warm pool, one hard deadline, fail open.

The browser calls this on `user.final` and appends the returned context to the
live session. Retrieval is additive: every call settles within `rag_timeout_s`
and any failure returns an empty answer with a reason, so speech never waits on
the knowledge base.

Connection rules (measured 2026-11-12, see docs/contracts.md):
- One module-level client with keep-alive; a fresh connection costs ~284 ms more
  than a reused one on the 172.20.x services, which would eat the deadline.
- `trust_env=False`: the broker runs behind Clash (HTTP(S)_PROXY=127.0.0.1:7897)
  and Python's proxy bypass does not understand CIDR ranges, so internal calls
  must opt out explicitly. GPT-Live still goes through the proxy (voice.py).
- Vector-only by default: WeKnora 0.9's keyword branch fails upstream (see
  HYBRID_AVAILABLE below), and WeKnora's own rerank timeout is 60 s, so this
  deadline is the real bound.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .config import Settings

log = logging.getLogger("annie.rag")

MAX_QUERY_CHARS = 200
LOGIN_TTL_S = 1_800.0

# Measured 2026-11-13 on WeKnora 0.9.0 + ParadeDB 0.25.11 (the shipped compose
# pair): every keyword retrieval fails with SQLSTATE XX000 "Unsupported query
# shape" and WeKnora logs "partial results: keyword=0, vector=50", then answers
# 200 with vector-only hits. 43/43 queries reproduced it, so this is an
# upstream index/query mismatch, not a per-query data problem. Until that pair
# is fixed, the gateway reports vector-only instead of pretending to be hybrid.
HYBRID_AVAILABLE = False
HYBRID_REASON = "weknora_keyword_branch_unsupported_query_shape"

# Exhibits get a lot of small talk. On the shdata corpus "你好" scores 0.552 —
# just above the 0.55 threshold — and would inject 《2026数字中国》 into a
# greeting. The miss margin to a real hit (0.568) is only 0.016, so this cannot
# rely on the score; it has to be a robust whole-query match.
#
# Normalization removes whitespace, punctuation and emoji, lowercases English,
# and then requires the ENTIRE query to be greeting/social words plus optional
# sentence-final particles. That covers "你好呀", "您好！", "你 好", "HELLO",
# "在不在", and still lets "你好，请问随申办企业云有几个频道" reach retrieval.
_SMALLTALK_CORE = (
    "hellothere|hithere|heythere|hello there|hi there|hey there|"
    "你好|您好|哈喽|嗨|hello|hi|hey|yo|"
    "早上好|上午好|中午好|下午好|晚上好|早安|晚安|"
    "谢谢|多谢|感谢|再见|拜拜|bye|goodbye|"
    "在吗|在不在|有人吗|你是谁|你叫什么名字|你叫什么"
)
_SMALLTALK_FILLER = "呀啊哦呢吗哈嘛啦吧哇呐么"
_SMALLTALK_RE = re.compile(rf"^(?:(?:{_SMALLTALK_CORE})[{_SMALLTALK_FILLER}]*)+$")
_NORMALIZE_RE = re.compile(r"[\s\W_]+", re.UNICODE)
SMALLTALK_MAX_CHARS = 24


def _normalize_smalltalk(text: str) -> str:
    return _NORMALIZE_RE.sub("", (text or "").lower())


def _is_smalltalk(text: str) -> bool:
    """True only when the whole query is a greeting/social phrase."""
    normalized = _normalize_smalltalk(text)
    if not normalized or len(normalized) > SMALLTALK_MAX_CHARS:
        return False
    return bool(_SMALLTALK_RE.match(normalized))


@dataclass(frozen=True)
class RagAnswer:
    context: str
    sources: tuple[str, ...]
    score: float | None
    reason: str  # ok | empty | below_threshold | greeting | timeout | upstream_error | breaker_open | disabled | rate_limited
    latency_ms: float
    retrieval: str = "vector-only"
    # high: >= rag_low_confidence; low: [rag_threshold, rag_low_confidence);
    # none: nothing injected. The card is marked from this field; the context
    # carries no spoken hedge, the persona forbids "资料中提到".
    confidence: str = "none"
    # Populated for breaker_open so callers can retry without guessing.
    retry_after_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "context": self.context,
            "sources": list(self.sources),
            "score": self.score,
            "reason": self.reason,
            "confidence": self.confidence,
            "retry_after_ms": self.retry_after_ms,
            "latency_ms": self.latency_ms,
            "retrieval": self.retrieval,
            "hybrid_available": HYBRID_AVAILABLE,
            "hybrid_reason": None if HYBRID_AVAILABLE else HYBRID_REASON,
        }


def _empty(reason: str, latency_ms: float = 0.0, retry_after_ms: float = 0.0) -> RagAnswer:
    return RagAnswer("", (), None, reason, latency_ms, retry_after_ms=retry_after_ms)


# Overview questions ("公司有哪些重点项目") share almost no surface words with
# the corpus section that answers them (八大场景 / 20+ 政务智能体 / 城市治理),
# so the vector query lands on an unrelated department table. Adding the domain
# terms is enough to aim it; the reply is still generated from retrieved text.
_OVERVIEW_RE = re.compile(
    r"重点项目|重大项目|主要项目|哪些项目|重点业务|主营业务|核心业务"
    r"|公司.{0,4}(做什么|的业务|业务范围|有哪些业务)"
)
_OVERVIEW_HINT = "政务智能化 政务智能体 八大场景 城市治理 数字政法 公共服务"


def _expand_query(query: str) -> str:
    if _OVERVIEW_RE.search(query) and _OVERVIEW_HINT not in query:
        return f"{query} {_OVERVIEW_HINT}"[:MAX_QUERY_CHARS]
    return query


def _clean(text: str) -> str:
    """Collapse markdown whitespace so the snippet reads naturally when spoken."""
    lines = [line.strip().lstrip("#").strip() for line in text.splitlines()]
    return " ".join(part for part in lines if part)


class RagClient:
    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.s = settings
        self.clock = clock
        self._client = httpx.AsyncClient(
            base_url=settings.rag_base_url,
            trust_env=False,
            timeout=httpx.Timeout(settings.rag_timeout_s, connect=settings.rag_connect_timeout_s),
            limits=httpx.Limits(
                max_keepalive_connections=settings.rag_pool_size,
                max_connections=settings.rag_pool_size * 2,
                keepalive_expiry=120.0,
            ),
            transport=transport,
        )
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._login_lock = asyncio.Lock()
        self._cache: dict[str, tuple[float, RagAnswer]] = {}
        self._sessions: dict[str, list] = {}
        self.disabled = False  # set on 401/403; needs a restart with valid credentials
        # Breaker state: failures counts consecutive misses, open_until bounds
        # the cool-down, _backoff_level climbs 0 → 1 → 2 on failed probes.
        self.failures = 0
        self.open_until = 0.0
        self._backoff_level = 0
        self._probe_inflight = False

    # -- availability ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.s.rag_enabled and not self.disabled

    @property
    def breaker_open(self) -> bool:
        return self.failures >= self.s.rag_breaker_threshold and self.clock() < self.open_until

    @property
    def breaker_half_open(self) -> bool:
        """Cool-down elapsed: exactly one real request may probe the upstream."""
        return self.failures >= self.s.rag_breaker_threshold and self.clock() >= self.open_until

    @property
    def breaker_state(self) -> str:
        if self.breaker_open:
            return "open"
        if self.breaker_half_open:
            return "half_open"
        return "closed"

    def status(self) -> dict:
        return {
            "enabled": self.available,
            "configured": self.s.rag_enabled,
            "breaker_open": self.breaker_open,
            "breaker_state": self.breaker_state,
            "breaker_failures": self.failures,
            # Be explicit about retrieval quality: a vector-only answer is not
            # "hybrid", and callers must not read it as multi-hop capable.
            "retrieval": "vector-only",
            "hybrid_requested": self.s.rag_hybrid,
            "hybrid_available": HYBRID_AVAILABLE,
            "hybrid_reason": None if HYBRID_AVAILABLE else HYBRID_REASON,
            "tenant_scoped": bool(self.s.rag_tenant_id),
        }

    # -- credentials ----------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.s.rag_api_key:
            headers["X-API-Key"] = self.s.rag_api_key
        # WeKnora 0.8.3 platform keys are tenant-scoped; unset keeps older
        # deployments (and the 8881 channel) byte-identical.
        if self.s.rag_tenant_id:
            headers["X-Tenant-ID"] = self.s.rag_tenant_id
        return headers

    async def _login(self) -> str | None:
        if self.s.rag_api_key or not (self.s.rag_email and self.s.rag_password):
            return None
        async with self._login_lock:
            if self._token and self.clock() < self._token_expires_at:
                return self._token
            try:
                async with asyncio.timeout(self.s.rag_timeout_s):
                    resp = await self._client.post(
                        "/auth/login",
                        json={"email": self.s.rag_email, "password": self.s.rag_password},
                    )
                if resp.status_code in (401, 403):
                    self._disable(resp.status_code)
                    return None
                if resp.status_code != 200:
                    return None
                token = resp.json().get("token")
            except (TimeoutError, httpx.HTTPError, ValueError):
                return None
            if not isinstance(token, str) or not token:
                return None
            self._token = token
            self._token_expires_at = self.clock() + LOGIN_TTL_S
            return token

    async def _headers(self) -> dict[str, str]:
        headers = dict(self._auth_headers())
        token = await self._login()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _disable(self, status: int) -> None:
        if not self.disabled:
            log.error("weknora rejected authentication (status=%s); /rag disabled", status)
        self.disabled = True

    # -- queries --------------------------------------------------------------

    def _cached(self, query: str) -> RagAnswer | None:
        entry = self._cache.get(query)
        if entry is None:
            return None
        expires, answer = entry
        if self.clock() >= expires:
            self._cache.pop(query, None)
            return None
        return answer

    def _remember(self, query: str, answer: RagAnswer) -> None:
        if answer.reason not in ("ok", "below_threshold", "empty"):
            return
        if len(self._cache) >= self.s.rag_cache_max:
            self._cache.clear()
        self._cache[query] = (self.clock() + self.s.rag_cache_ttl_s, answer)

    def _cooldown_s(self) -> float:
        base = self.s.rag_breaker_cooldown_s
        return min(base * (2 ** self._backoff_level), self.s.rag_breaker_max_cooldown_s)

    def _fail(self, reason: str) -> None:
        """Record a real upstream failure and (re)open the breaker with backoff.

        The first open after `threshold` failures waits one base cool-down; each
        failed half-open probe doubles it, capped at `rag_breaker_max_cooldown_s`.
        """
        self.failures += 1
        if self.failures < self.s.rag_breaker_threshold:
            return
        cooldown = self._cooldown_s()
        self.open_until = self.clock() + cooldown
        self._backoff_level = min(self._backoff_level + 1, 2)
        log.warning(
            "rag circuit breaker open for %.0fs (failures=%d, state=open)",
            cooldown,
            self.failures,
        )

    def _succeed(self) -> None:
        if self.failures:
            log.info("rag circuit breaker closed")
        self.failures = 0
        self._backoff_level = 0
        self.open_until = 0.0

    def _breaker_reason(self) -> RagAnswer:
        retry_ms = max(0.0, (self.open_until - self.clock()) * 1000)
        return _empty("breaker_open", retry_after_ms=round(retry_ms, 1))

    async def query(self, text: str, *, session_key: str | None = None) -> RagAnswer:
        started = time.perf_counter()
        if not self.s.rag_enabled or self.disabled:
            return _empty("disabled")
        if self.breaker_open:
            return self._breaker_reason()
        query = " ".join((text or "").split())[:MAX_QUERY_CHARS]
        if len(query) < 2:
            return _empty("empty")
        # Small talk never reaches retrieval: it must not inject a source card.
        if _is_smalltalk(query):
            return _empty("greeting")
        cached = self._cached(query)
        if cached is not None:
            return cached

        if session_key is not None and not self._session_allows(session_key):
            return _empty("rate_limited")

        # Half-open: let exactly one request through to test the upstream.
        is_probe = self.breaker_half_open
        if is_probe:
            if self._probe_inflight:
                return self._breaker_reason()
            self._probe_inflight = True
        try:
            answer = await self._search(_expand_query(query), started)
        finally:
            if is_probe:
                self._probe_inflight = False

        if answer.reason in ("timeout", "upstream_error"):
            self._fail(answer.reason)
        elif answer.reason in ("ok", "empty", "below_threshold"):
            # A completed exchange — even an empty or below-threshold answer —
            # proves the upstream is healthy. Cache hits never get here, so a
            # probe always performs a real request.
            self._succeed()
        self._remember(query, answer)
        return answer

    def _session_allows(self, key: str) -> bool:
        # Retrieval is one call per user turn; the token already scopes it to a session.
        bucket = self._sessions.get(key)
        now = self.clock()
        if bucket is None:
            if len(self._sessions) > 4096:
                self._sessions.clear()
            self._sessions[key] = [now, 1]
            return True
        if now - bucket[0] >= 1.0:
            bucket[0], bucket[1] = now, 1
            return True
        if bucket[1] >= self.s.rag_per_session_rps:
            return False
        bucket[1] += 1
        return True

    async def _search(self, query: str, started: float) -> RagAnswer:
        body: dict[str, Any] = {
            "query_text": query,
            "match_count": self.s.rag_top_k,
            # Off by default: the keyword branch is broken upstream, and asking
            # for it only burns latency before WeKnora silently drops it.
            "disable_keywords_match": not self.s.rag_hybrid,
            "vector_threshold": 0.0,
        }
        if self.s.rag_rerank_model:
            body["rerank"] = {
                "model_id": self.s.rag_rerank_model,
                "threshold": self.s.rag_rerank_threshold,
            }
        path = f"/knowledge-bases/{self.s.rag_kb_id}/hybrid-search"
        try:
            async with asyncio.timeout(self.s.rag_timeout_s):
                resp = await self._client.post(path, json=body, headers=await self._headers())
        except (TimeoutError, httpx.TimeoutException):
            return _empty("timeout", self._elapsed(started))
        except httpx.HTTPError as exc:
            log.info("weknora transport error: %s", type(exc).__name__)
            return _empty("upstream_error", self._elapsed(started))

        if resp.status_code in (401, 403):
            self._disable(resp.status_code)
            return _empty("disabled", self._elapsed(started))
        if resp.status_code != 200:
            log.info("weknora status=%s", resp.status_code)
            return _empty("upstream_error", self._elapsed(started))
        try:
            payload = resp.json()
        except ValueError:
            return _empty("upstream_error", self._elapsed(started))

        hits = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(hits, list):
            return _empty("upstream_error", self._elapsed(started))
        return self._compose(hits, payload.get("meta") if isinstance(payload, dict) else None, started)

    def _compose(self, hits: list, meta: Any, started: float) -> RagAnswer:
        scored: list[tuple[float, dict]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            score = hit.get("score")
            content = hit.get("content") or hit.get("matched_content") or ""
            if isinstance(score, (int, float)) and isinstance(content, str) and content.strip():
                scored.append((float(score), hit))
        if not scored:
            return _empty("empty", self._elapsed(started))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score = scored[0][0]

        # rerank returns one merged score in meta.rerank; the per-hit score is the
        # vector recall score, which must not be compared against the rerank threshold.
        threshold = self.s.rag_threshold
        if self.s.rag_rerank_model and isinstance(meta, dict):
            rerank = meta.get("rerank") or {}
            if isinstance(rerank, dict) and isinstance(rerank.get("top_score"), (int, float)):
                top_score = float(rerank["top_score"])
                threshold = self.s.rag_rerank_threshold
        if top_score < threshold:
            return _empty("below_threshold", self._elapsed(started))

        # C10b: the band just above the threshold is injected, and `confidence`
        # lets the source card mark it. The prefix itself must stay neutral: any
        # "资料中提到"-style wording here would be read out loud by the live
        # model, which the persona forbids. Hedging, if any, is the voice
        # layer's instruction, not text the model can parrot back.
        confidence = "high" if top_score >= self.s.rag_low_confidence else "low"
        if confidence == "low":
            prefix = "展馆资料（不要编造资料外的事实）：\n"
        else:
            prefix = "展馆资料（可直接引用，不要编造资料外的事实）：\n"
        budget = self.s.rag_max_chars - len(prefix)  # whole block must fit the append limit
        parts: list[str] = []
        sources: list[str] = []
        seen: set[str] = set()
        used = 0
        for score, hit in scored:
            if len(parts) >= self.s.rag_max_snippets:
                break
            source = str(hit.get("knowledge_title") or hit.get("knowledge_filename") or "").strip()
            key = str(hit.get("knowledge_id") or hit.get("id") or source)
            if key in seen:
                continue
            text = _clean(str(hit.get("content") or ""))
            if not text:
                continue
            label = f"《{source}》" if source else ""
            room = budget - used - len(label)
            if room <= 0:
                break
            snippet = text[:room]
            seen.add(key)
            used += len(label) + len(snippet)
            sources.append(source or "展馆资料")
            parts.append(f"{label}{snippet}" if label else snippet)
        if not parts:
            return _empty("empty", self._elapsed(started))
        context = prefix + "\n".join(parts)
        return RagAnswer(
            context,
            tuple(sources),
            round(top_score, 4),
            "ok",
            self._elapsed(started),
            "vector-only",
            confidence,
        )

    def _elapsed(self, started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 1)

    # -- L0/L1 cards ----------------------------------------------------------

    def cards(self, area: str | None) -> dict:
        """Static knowledge cards for session instructions; empty when unset."""
        path = self.s.rag_cards_path
        if not path:
            return {"text": "", "count": 0}
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("rag cards unreadable: %s", path)
            return {"text": "", "count": 0}
        cards = raw if isinstance(raw, list) else raw.get("cards", [])
        picked = []
        for card in cards if isinstance(cards, list) else []:
            if not isinstance(card, dict) or not card.get("text"):
                continue
            areas = card.get("areas") or ([card["area"]] if card.get("area") else [])
            if area and area != "all" and areas and area not in areas:
                continue
            picked.append(card)
        lines: list[str] = []
        used = 0
        for card in picked:
            line = f"- {card.get('title', '展项')}：{card['text']}"
            if used + len(line) > self.s.rag_cards_max_chars:
                break
            used += len(line)
            lines.append(line)
        text = "展馆知识卡（只能依据这些卡片答题，卡片外的问题要说明不知道）：\n" + "\n".join(lines) if lines else ""
        return {"text": text, "count": len(lines)}

    async def aclose(self) -> None:
        await self._client.aclose()
