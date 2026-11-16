"""Retrieval gateway: deadline, thresholds, breaker, auth and L0/L1 cards."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from annie_broker import tokens
from conftest import Upstream

RAG = {
    "rag_base_url": "http://weknora.test/api/v1",
    "rag_kb_id": "kb-1",
    "rag_api_key": "k-test",
    "rag_timeout_s": 0.2,
    "rag_threshold": 0.55,
}

HIT = {
    "data": [
        {
            "id": "chunk-1",
            "knowledge_id": "doc-1",
            "knowledge_title": "MY-KB-AC-001_BNM、电子货币.md",
            "content": "# Summary\n电子货币的发行与支付活动由马来西亚国家银行 BNM 监管。",
            "score": 0.7369,
        }
    ]
}


def token(session_id: str = "sess-1", ttl: int = 300) -> str:
    return tokens.mint(b"test-secret", session_id, int(time.time()) + ttl)


def rag_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=HIT)


class SlowTransport(httpx.AsyncBaseTransport):
    """Never answers within the deadline; proves the caller-side bound."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.delay)
        return httpx.Response(200, json=HIT)


def test_rag_returns_extracted_context(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    resp = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == "ok"
    assert body["context"].startswith("展馆资料")
    assert "BNM" in body["context"]
    assert body["sources"] == ["MY-KB-AC-001_BNM、电子货币.md"]
    assert body["score"] == 0.7369

    request = upstream.requests[0]
    assert request.url.path == "/api/v1/knowledge-bases/kb-1/hybrid-search"
    assert request.headers["x-api-key"] == "k-test"
    sent = json.loads(request.content)
    assert sent["disable_keywords_match"] is True  # keyword branch broken upstream
    assert sent["query_text"] == "谁来监管电子货币"
    # The answer must say what it actually is, not imply hybrid retrieval.
    assert body["retrieval"] == "vector-only"
    assert body["hybrid_available"] is False
    assert body["hybrid_reason"] == "weknora_keyword_branch_unsupported_query_shape"


def test_overview_question_is_expanded_for_retrieval_only(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    body = client.post("/rag", json={"token": token(), "query": "公司有哪些重点项目"}).json()

    sent = json.loads(upstream.requests[0].content)
    assert sent["query_text"].startswith("公司有哪些重点项目")
    assert "政务智能化" in sent["query_text"]
    # The injected context is the retrieved text, never the rewritten query.
    assert "政务智能化 政务智能体" not in body["context"]


def test_normal_question_is_not_expanded(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    client.post("/rag", json={"token": token(), "query": "随申办企业云有几个频道"})

    sent = json.loads(upstream.requests[0].content)
    assert sent["query_text"] == "随申办企业云有几个频道"


def test_rag_hybrid_flag_is_requested_but_still_reported_unavailable(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **{**RAG, "rag_hybrid": True})

    body = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"}).json()

    assert json.loads(upstream.requests[0].content)["disable_keywords_match"] is False
    assert body["retrieval"] == "vector-only"
    assert body["hybrid_available"] is False


def test_health_reports_hybrid_state(make_client):
    client = make_client(rag=Upstream(rag_ok), **RAG)
    rag = client.get("/health").json()["rag"]
    assert rag["retrieval"] == "vector-only"
    assert rag["hybrid_available"] is False
    assert rag["hybrid_reason"] == "weknora_keyword_branch_unsupported_query_shape"


def test_rag_rejects_scores_below_threshold(make_client):
    low = lambda request: httpx.Response(200, json={"data": [dict(HIT["data"][0], score=0.42)]})
    client = make_client(rag=Upstream(low), **RAG)

    body = client.post("/rag", json={"token": token(), "query": "红烧肉怎么做"}).json()

    assert body["context"] == ""
    assert body["reason"] == "below_threshold"


def test_rag_times_out_without_blocking(make_client):
    client = make_client(rag=SlowTransport(), **{**RAG, "rag_timeout_s": 0.05})

    body = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"}).json()

    assert body["context"] == ""
    assert body["reason"] == "timeout"


def test_rag_breaker_opens_after_repeated_upstream_errors(make_client):
    upstream = Upstream(lambda request: httpx.Response(500, text="boom"))
    client = make_client(rag=upstream, **RAG)
    broker = client.app.state.broker

    for _ in range(3):
        answer = client.portal.call(broker.rag.query, "谁来监管电子货币")
        assert answer.reason == "upstream_error"
    assert len(upstream.requests) == 3

    answer = client.portal.call(broker.rag.query, "另一个问题")
    assert answer.reason == "breaker_open"
    assert len(upstream.requests) == 3  # fail fast, no fourth call


def test_rag_caches_repeated_questions(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    for _ in range(3):
        body = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"}).json()
        assert body["reason"] == "ok"
    assert len(upstream.requests) == 1


def test_rag_requires_a_valid_session_token(make_client):
    client = make_client(rag=Upstream(rag_ok), **RAG)

    assert client.post("/rag", json={"token": "nope", "query": "测试问题"}).status_code == 401
    assert client.post("/rag", json={"token": token(ttl=-10), "query": "测试问题"}).status_code == 401
    assert client.post("/rag", json={"token": token()}).status_code == 400


def test_rag_refuses_foreign_origin(make_client):
    client = make_client(rag=Upstream(rag_ok), **RAG)

    resp = client.post(
        "/rag",
        json={"token": token(), "query": "谁来监管电子货币"},
        headers={"Origin": "http://evil.example"},
    )

    assert resp.status_code == 403


def test_rag_disabled_returns_empty_answer(make_client):
    client = make_client()

    body = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"}).json()

    assert body["context"] == ""
    assert body["reason"] == "disabled"
    assert client.get("/health").json()["providers"]["rag"] is False


def test_cards_filter_by_area(make_client, tmp_path):
    cards = tmp_path / "cards.json"
    cards.write_text(
        json.dumps(
            [
                {"title": "BNM", "text": "电子货币由 BNM 监管。", "areas": ["金融", "入口"]},
                {"title": "SToDS", "text": "数字服务税税率 8%。", "areas": ["税务"]},
            ]
        ),
        encoding="utf-8",
    )
    client = make_client(rag_cards_path=str(cards))

    body = client.get("/rag/cards", params={"area": "金融"}).json()

    assert body["count"] == 1
    assert "BNM" in body["text"] and "SToDS" not in body["text"]
    assert client.get("/rag/cards", params={"area": "all"}).json()["count"] == 2


def test_rag_sends_tenant_header_when_configured(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **{**RAG, "rag_tenant_id": "10000"})

    body = client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"}).json()

    assert body["reason"] == "ok"
    assert upstream.requests[0].headers["x-tenant-id"] == "10000"
    assert client.get("/health").json()["providers"]["rag"] is True


def test_rag_omits_tenant_header_by_default(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    client.post("/rag", json={"token": token(), "query": "谁来监管电子货币"})

    assert "x-tenant-id" not in upstream.requests[0].headers


def test_greeting_is_intercepted_before_retrieval(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    for phrase in ("你好", "您好呀！", "Hello", "谢谢", "你是谁？"):
        body = client.post("/rag", json={"token": token(), "query": phrase}).json()
        assert body["reason"] == "greeting", phrase
        assert body["context"] == "" and body["sources"] == [], phrase

    assert upstream.requests == []  # no source card can be injected


def test_greeting_word_inside_a_real_question_still_retrieves(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    body = client.post(
        "/rag", json={"token": token(), "query": "你好，请问随申办企业云的五大频道是什么"}
    ).json()

    assert body["reason"] == "ok"
    assert len(upstream.requests) == 1


GREETING_VARIANTS = (
    "你好",
    "您好",
    "你好呀",
    "早上好",
    "在吗",
    "哈喽",
    "hi",
    "你是谁",
    "谢谢",
    "再见",
    "在不在",
    "你好！！",
    "你 好",
    "HELLO",
    "hi there",
)


def test_greeting_variants_are_all_intercepted(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    for phrase in GREETING_VARIANTS:
        body = client.post("/rag", json={"token": token(), "query": phrase}).json()
        assert body["reason"] == "greeting", phrase
        assert body["context"] == "" and body["sources"] == [], phrase

    assert upstream.requests == []


def test_real_questions_that_start_with_a_greeting_still_retrieve(make_client):
    upstream = Upstream(rag_ok)
    client = make_client(rag=upstream, **RAG)

    for i, phrase in enumerate(
        (
            "你好，请问随申办企业云有几个频道",
            "早上好，城市视觉中枢是什么",
            "在吗，我想了解一下政务智能体",
        )
    ):
        body = client.post("/rag", json={"token": token(f"sess-{i}"), "query": phrase}).json()
        assert body["reason"] == "ok", phrase

    assert len(upstream.requests) == 3


def _scored(score: float):
    return lambda request: httpx.Response(200, json={"data": [dict(HIT["data"][0], score=score)]})


def test_low_confidence_band_is_injected_but_marked(make_client):
    client = make_client(rag=Upstream(_scored(0.58)), **RAG)

    body = client.post("/rag", json={"token": token(), "query": "城市视觉中枢是什么"}).json()

    assert body["reason"] == "ok"
    assert body["confidence"] == "low"
    assert body["sources"] == ["MY-KB-AC-001_BNM、电子货币.md"]
    assert body["context"].startswith("展馆资料（不要编造")
    for hedge in ("资料中提到", "根据资料", "知识库显示", "知识库中提到"):
        assert hedge not in body["context"]


def test_high_confidence_band_is_not_hedged(make_client):
    client = make_client(rag=Upstream(_scored(0.65)), **RAG)

    body = client.post("/rag", json={"token": token(), "query": "城市视觉中枢是什么"}).json()

    assert body["reason"] == "ok"
    assert body["confidence"] == "high"
    assert body["context"].startswith("展馆资料（可直接引用")
    assert "资料中提到" not in body["context"]


def test_score_below_threshold_is_not_injected(make_client):
    client = make_client(rag=Upstream(_scored(0.549)), **RAG)

    body = client.post("/rag", json={"token": token(), "query": "城市视觉中枢是什么"}).json()

    assert body["reason"] == "below_threshold"
    assert body["context"] == "" and body["sources"] == []
    assert body["confidence"] == "none"


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _install_clock(client):
    clock = FakeClock()
    client.app.state.broker.rag.clock = clock
    return clock


def test_breaker_half_open_probe_recovers_within_cooldown(make_client):
    upstream = Upstream(lambda request: httpx.Response(500, text="boom"))
    client = make_client(rag=upstream, **RAG)
    broker = client.app.state.broker
    clock = _install_clock(client)

    for query in ("问题一", "问题二", "问题三"):
        assert client.portal.call(broker.rag.query, query).reason == "upstream_error"
    assert broker.rag.breaker_state == "open"

    answer = client.portal.call(broker.rag.query, "问题四")
    assert answer.reason == "breaker_open"
    assert 9_000 <= answer.retry_after_ms <= 10_000
    assert len(upstream.requests) == 3  # fail fast: no call while open

    clock.advance(10.1)
    assert broker.rag.breaker_state == "half_open"
    upstream.handler = rag_ok

    probe = client.portal.call(broker.rag.query, "问题五")
    assert probe.reason == "ok"
    assert broker.rag.breaker_state == "closed"
    assert len(upstream.requests) == 4  # exactly one probe

    assert client.portal.call(broker.rag.query, "问题六").reason == "ok"
    assert len(upstream.requests) == 5


def test_failed_probe_backs_off_10_20_30_capped(make_client):
    upstream = Upstream(lambda request: httpx.Response(500, text="boom"))
    client = make_client(rag=upstream, **RAG)
    broker = client.app.state.broker
    clock = _install_clock(client)

    for query in ("问题一", "问题二", "问题三"):
        client.portal.call(broker.rag.query, query)
    assert broker.rag.breaker_state == "open"

    clock.advance(10.1)
    assert client.portal.call(broker.rag.query, "问题四").reason == "upstream_error"
    assert abs((broker.rag.open_until - clock.t) - 20.0) < 0.01

    clock.advance(20.1)
    assert client.portal.call(broker.rag.query, "问题五").reason == "upstream_error"
    assert abs((broker.rag.open_until - clock.t) - 30.0) < 0.01

    clock.advance(30.1)
    assert client.portal.call(broker.rag.query, "问题六").reason == "upstream_error"
    assert abs((broker.rag.open_until - clock.t) - 30.0) < 0.01  # capped


def test_breaker_answer_has_no_negative_script_in_body_or_logs(make_client, caplog):
    upstream = Upstream(lambda request: httpx.Response(500, text="boom"))
    client = make_client(rag=upstream, **RAG)
    broker = client.app.state.broker

    with caplog.at_level("WARNING", logger="annie.rag"):
        for query in ("问题一", "问题二", "问题三"):
            client.portal.call(broker.rag.query, query)

    answer = client.portal.call(broker.rag.query, "问题四")

    assert answer.reason == "breaker_open"
    assert answer.retry_after_ms > 0
    body = json.dumps(answer.as_dict(), ensure_ascii=False)
    logs = "\n".join(record.getMessage() for record in caplog.records)
    for text in (body, logs):
        assert "没资料" not in text
        assert "系统繁忙" not in text
