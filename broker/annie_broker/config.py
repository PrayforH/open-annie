"""Broker settings, read once from the environment."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field

VERSION = "0.1.0"

# Supplier price, not the product target. GPT-Live lists $0.05/min billed per
# second (docs/models/gpt-live-1, 2026-09-27); the measured 2026-11-12 bill of
# ¥0.36/min is that same number at ¥7.2/USD. The product's ¥0.05/min goal is a
# different figure and must never be used for accounting.
DEFAULT_VOICE_USD_PER_MIN = 0.05
# A WebRTC create bills 15 s up front, credited against the running session.
VOICE_INIT_SECONDS = 15
# Jev: $0.042 per million input tokens, output free (docs.typesafe.ai/models).
DEFAULT_JEV_USD_PER_MTOK = 0.042


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return (_env(name, "1" if default else "0") or "").lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in (_env(name, default) or "").split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    typesafe_api_key: str | None = None
    hmac_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    daily_budget_usd: float = 25.0
    # What the supplier actually charges per voice minute; accounting only.
    voice_usd_per_min: float = DEFAULT_VOICE_USD_PER_MIN
    jev_usd_per_mtok: float = DEFAULT_JEV_USD_PER_MTOK
    max_sessions: int = 20
    session_cap_s: int = 300
    idle_timeout_s: int = 30
    per_ip_daily: int = 2  # 0 disables the per-IP limit
    jev_model: str = "jev-latest"
    gptlive_model: str = "gpt-live-1"
    allowed_origins: tuple[str, ...] = ("http://localhost:*", "http://127.0.0.1:*")
    # The LAN gate: private networks may open a voice session; public origins stay refused.
    # Keep as config so the exhibit LAN can be widened without editing code.
    lan_cidrs: tuple[str, ...] = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
    lan_hosts: tuple[str, ...] = (".local",)
    # Speak a wrap-up line this long before the session cap closes the call.
    session_warn_s: int = 20
    # Jev allows 1,200 rpm per key; stay under it so bursts never hit 429.
    jev_rpm: int = 1000
    jev_burst: int = 20
    # The browser enforces its own 900 ms (reply) / 700 ms (listen) budgets; this
    # server deadline only bounds how long an upstream call may hold a slot.
    decide_timeout_s: float = 1.5
    decide_per_session_rps: float = 5.0
    token_grace_s: int = 30
    default_voice: str = "marin"
    default_instructions: str = ""
    # Restrict what the browser data channel may send; () = upstream default (all).
    client_events: tuple[str, ...] = ()
    sideband: bool = True
    hangup_fallback: bool = True
    safety_identifier: bool = False
    ip_header: str | None = None  # e.g. "x-forwarded-for" behind a trusted proxy
    openai_base_url: str = "https://api.openai.com"
    typesafe_base_url: str = "https://api.typesafe.ai"
    d1_base_url: str | None = None
    # WeKnora retrieval gateway. Off unless a knowledge base and credentials are set.
    rag_base_url: str = "http://127.0.0.1:8881/api/v1"
    rag_kb_id: str | None = None
    rag_api_key: str | None = None
    # WeKnora 0.8.3 platform keys are tenant-scoped: without X-Tenant-ID the
    # platform answers 409 TENANT_REQUIRED. Empty leaves the 8881 channel as-is.
    rag_tenant_id: str | None = None
    rag_email: str | None = None
    rag_password: str | None = None
    # The caller-side deadline is the real bound: WeKnora's own rerank timeout is 60 s.
    rag_timeout_s: float = 1.2
    rag_connect_timeout_s: float = 0.3
    rag_top_k: int = 3
    rag_max_snippets: int = 2
    rag_max_chars: int = 420  # session.instructions.append caps content at 500 tokens
    rag_threshold: float = 0.55
    # Scores in [rag_threshold, rag_low_confidence) are still injected and the
    # answer is marked `confidence: low` for the source card; no spoken hedge is
    # added to the context (the persona forbids saying "资料中提到"). Below
    # rag_threshold nothing is injected.
    rag_low_confidence: float = 0.62
    # Circuit breaker. After the cooldown the next request is a real probe: a
    # success closes immediately, a failure backs off 10 → 20 → 30 s (capped).
    rag_breaker_threshold: int = 3
    rag_breaker_cooldown_s: float = 10.0
    rag_breaker_max_cooldown_s: float = 30.0
    # Ask WeKnora for the keyword branch too. Leave off: the branch is broken
    # upstream (see the note in rag.py), so this flag only adds a request field.
    rag_hybrid: bool = False
    rag_rerank_model: str | None = None
    rag_rerank_threshold: float = 0.3
    rag_cache_ttl_s: float = 300.0
    rag_cache_max: int = 256
    rag_per_session_rps: float = 2.0
    rag_pool_size: int = 8
    rag_cards_path: str | None = None
    rag_cards_max_chars: int = 4000
    # Dev only: GET /dev-token hands out /decide tokens without a voice session.
    dev: bool = False
    dev_token_ttl_s: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        secret = _env("ANNIE_HMAC_SECRET")
        return cls(
            # fal secrets are shared by every app on an account, so the fal deploy uses
            # namespaced ANNIE_* names; plain names still work locally.
            openai_api_key=_env("ANNIE_OPENAI_API_KEY") or _env("OPENAI_API_KEY"),
            typesafe_api_key=_env("ANNIE_TYPESAFE_API_KEY") or _env("TYPESAFE_API_KEY"),
            hmac_secret=secret.encode() if secret else secrets.token_bytes(32),
            daily_budget_usd=_float("ANNIE_DAILY_BUDGET_USD", 25.0),
            voice_usd_per_min=_float("ANNIE_VOICE_USD_PER_MIN", DEFAULT_VOICE_USD_PER_MIN),
            jev_usd_per_mtok=_float(
                "ANNIE_JEV_USD_PER_MTOK", DEFAULT_JEV_USD_PER_MTOK
            ),
            max_sessions=_int("ANNIE_MAX_SESSIONS", 20),
            session_cap_s=_int("ANNIE_SESSION_CAP_S", 300),
            idle_timeout_s=_int("ANNIE_IDLE_TIMEOUT_S", 30),
            per_ip_daily=_int("ANNIE_PER_IP_DAILY", 2),
            lan_cidrs=_list(
                "ANNIE_LAN_CIDRS", "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,fc00::/7"
            ),
            lan_hosts=_list("ANNIE_LAN_HOSTS", ".local"),
            session_warn_s=_int("ANNIE_SESSION_WARN_S", 20),
            jev_model=_env("JEV_MODEL", "jev-latest"),
            gptlive_model=_env("GPTLIVE_MODEL", "gpt-live-1"),
            allowed_origins=_list(
                "ANNIE_ALLOWED_ORIGINS", "http://localhost:*,http://127.0.0.1:*"
            ),
            jev_rpm=_int("ANNIE_JEV_RPM", 1000),
            decide_timeout_s=_float("ANNIE_DECIDE_TIMEOUT_S", 1.5),
            default_voice=_env("ANNIE_VOICE", "marin"),
            default_instructions=_env("ANNIE_INSTRUCTIONS", ""),
            client_events=_list("ANNIE_CLIENT_EVENTS", ""),
            sideband=_bool("ANNIE_SIDEBAND", True),
            hangup_fallback=_bool("ANNIE_HANGUP_FALLBACK", True),
            safety_identifier=_bool("ANNIE_SAFETY_IDENTIFIER", False),
            ip_header=_env("ANNIE_IP_HEADER"),
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com"),
            typesafe_base_url=_env("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
            d1_base_url=_env("ANNIE_D1_BASE_URL"),
            rag_base_url=_env("ANNIE_RAG_BASE_URL", "http://127.0.0.1:8881/api/v1") or "",
            rag_kb_id=_env("ANNIE_RAG_KB_ID"),
            rag_api_key=_env("ANNIE_RAG_API_KEY"),
            rag_tenant_id=_env("ANNIE_RAG_TENANT_ID"),
            rag_email=_env("ANNIE_RAG_EMAIL"),
            rag_password=_env("ANNIE_RAG_PASSWORD"),
            rag_timeout_s=_float("ANNIE_RAG_TIMEOUT_S", 1.2),
            rag_connect_timeout_s=_float("ANNIE_RAG_CONNECT_TIMEOUT_S", 0.3),
            rag_top_k=_int("ANNIE_RAG_TOP_K", 3),
            rag_max_snippets=_int("ANNIE_RAG_MAX_SNIPPETS", 2),
            rag_max_chars=_int("ANNIE_RAG_MAX_CHARS", 420),
            rag_threshold=_float("ANNIE_RAG_THRESHOLD", 0.55),
            rag_low_confidence=_float("ANNIE_RAG_LOW_CONFIDENCE", 0.62),
            rag_breaker_threshold=_int("ANNIE_RAG_BREAKER_THRESHOLD", 3),
            rag_breaker_cooldown_s=_float("ANNIE_RAG_BREAKER_COOLDOWN_S", 10.0),
            rag_breaker_max_cooldown_s=_float("ANNIE_RAG_BREAKER_MAX_COOLDOWN_S", 30.0),
            rag_hybrid=_bool("ANNIE_RAG_HYBRID", False),
            rag_rerank_model=_env("ANNIE_RAG_RERANK_MODEL"),
            rag_rerank_threshold=_float("ANNIE_RAG_RERANK_THRESHOLD", 0.3),
            rag_cache_ttl_s=_float("ANNIE_RAG_CACHE_TTL_S", 300.0),
            rag_cache_max=_int("ANNIE_RAG_CACHE_MAX", 256),
            rag_per_session_rps=_float("ANNIE_RAG_PER_SESSION_RPS", 2.0),
            rag_pool_size=_int("ANNIE_RAG_POOL_SIZE", 8),
            rag_cards_path=_env("ANNIE_RAG_CARDS_PATH"),
            rag_cards_max_chars=_int("ANNIE_RAG_CARDS_MAX_CHARS", 4000),
            dev=_bool("ANNIE_DEV", False),
            dev_token_ttl_s=_int("ANNIE_DEV_TOKEN_TTL_S", 900),
        )

    @property
    def rag_enabled(self) -> bool:
        has_auth = bool(self.rag_api_key or (self.rag_email and self.rag_password))
        return bool(self.rag_base_url and self.rag_kb_id and has_auth)

    @property
    def voice_usd_per_second(self) -> float:
        return self.voice_usd_per_min / 60.0

    @property
    def jev_usd_per_input_token(self) -> float:
        return self.jev_usd_per_mtok / 1_000_000.0

    @property
    def session_cap_usd(self) -> float:
        return self.session_cap_s * self.voice_usd_per_second

    @property
    def sessions_per_budget(self) -> int:
        """How many full-cap sessions the daily budget can cover back to back?

        Every admission reserves a whole session at the cap, so this is the
        number the gate will actually allow: budget / (cap_s / 60 * $/min).
        Empty when the cap costs nothing (it would divide by zero).
        """
        if self.session_cap_usd <= 0:
            return 0
        return int(self.daily_budget_usd // self.session_cap_usd)
