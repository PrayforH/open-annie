// Barge-in and LAN-gate unit tests. Run with: node --test packages/core/tests
import test from "node:test";
import assert from "node:assert/strict";

import { Bus } from "../src/bus.js";
import { isLocalStage, isPrivateHost } from "../src/net.js";
import {
  BREAKER_FALLBACK_LINE,
  DELEGATION_FALLBACK,
  DELEGATION_WATCHDOG_MS,
  GptLiveVoice,
  findL0Card,
  parseL0Cards,
} from "../src/voice-gptlive.js";

function makeVoice(opts = {}) {
  const bus = new Bus();
  const clock = { t: 0, now: () => clock.t };
  const voice = new GptLiveVoice({ bus, clock, brokerUrl: "http://127.0.0.1:8787", ...opts });
  const mic = { amp: 0 };
  const events = [];
  voice.micAnalyser = { sample: () => ({ amp: mic.amp }) };
  voice.analyser = { sample: () => ({ amp: 0.6, db: -20, vowels: { aa: 1 } }) };
  voice.el = {
    paused: false,
    playCalls: 0,
    pause() {
      this.paused = true;
    },
    play() {
      this.paused = false;
      this.playCalls++;
      return Promise.resolve();
    },
  };
  voice.asst = { open: true, text: "hello there", last: 0 };
  voice.lastAnnieActive = 0;
  bus.on("playback.stopped", (e) => events.push(e));
  bus.on("bargein", () => events.push({ type: "bargein" }));
  return { voice, mic, clock, events };
}

test("barge-in silences local playout inside 300 ms of detected voice", () => {
  const { voice, mic, clock, events } = makeVoice();
  mic.amp = 0.4;
  voice.tick(0, true); // voice onset, over Annie
  voice.tick(0.16, true); // past BARGEIN_DUCK_S (0.12)
  assert.equal(voice.el.paused, true, "element paused");
  assert.equal(voice.playout.stopped, true);
  const stopped = events.find((e) => e.reason === "user-voice");
  assert.ok(stopped, "playback.stopped emitted");
  assert.ok(stopped.latency_ms <= 300, `latency ${stopped.latency_ms} ms`);
});

test("a short blip resumes; a confirmed barge-in does not", () => {
  const short = makeVoice();
  short.mic.amp = 0.4;
  short.voice.tick(0, true);
  short.voice.tick(0.16, true);
  short.mic.amp = 0.0;
  short.voice.tick(0.2, true);
  short.voice.tick(0.5, true); // quiet > BARGEIN_CANCEL_S, never barged
  assert.equal(short.voice.el.paused, false, "short blip resumes the live stream");
  assert.ok(short.events.some((e) => e.type === "bargein") === false);

  const real = makeVoice();
  real.mic.amp = 0.4;
  real.voice.tick(0, true);
  real.voice.tick(0.4, true); // past BARGEIN_HOLD_S, confirmed
  assert.ok(real.events.some((e) => e.type === "bargein"), "barge-in confirmed");
  real.mic.amp = 0.0;
  real.voice.tick(0.45, true);
  real.voice.tick(0.8, true); // quiet after a confirmed cut
  assert.equal(real.voice.el.paused, true, "stays stopped until her next turn");
});

test("mouth stays closed while playout is paused", () => {
  const { voice, mic } = makeVoice();
  mic.amp = 0.4;
  voice.tick(0, true);
  voice.tick(0.2, true);
  const frame = voice.tick(0.25, true);
  assert.equal(frame.amp, 0);
  assert.equal(frame.visemes ?? null, null);
});

test("LAN gate accepts private hosts and refuses the public internet", () => {
  for (const host of ["localhost", "127.0.0.1", "172.20.109.60", "192.168.1.5", "10.2.3.4", "exhibit.local"]) {
    assert.equal(isPrivateHost(host), true, host);
  }
  for (const host of ["8.8.8.8", "exhibit.example.com", "1270.0.0.1"]) {
    assert.equal(isPrivateHost(host), false, host);
  }
  const params = new Map();
  const query = { get: (k) => params.get(k) ?? null };
  assert.equal(isLocalStage("172.20.109.60", query), true);
  assert.equal(isLocalStage("exhibit.example.com", query), false);
  params.set("lan", "1");
  assert.equal(isLocalStage("exhibit.example.com", query), true);
});

// ---- client delegation -------------------------------------------------------
// GPT-Live stops and waits after session.delegation.created. Every path must
// answer it, or the turn (and the session) stalls.
function makeDelegationVoice(opts = {}) {
  const h = makeVoice(opts);
  const sent = [];
  h.voice.dc = {
    readyState: "open",
    send(payload) {
      sent.push(JSON.parse(payload));
    },
  };
  h.voice.token = "test-token";
  h.voice.rag = { enabled: true, timeout_ms: 200 };
  return { ...h, sent };
}

const DELEGATION_EVENT = {
  type: "session.delegation.created",
  delegation: { id: "dlg_1", target: "client", type: "client" },
};

test("a delegation after retrieval is released with the cached context at once", () => {
  const { voice, sent } = makeDelegationVoice();
  voice.ragTurn = 1;
  voice.ragState = { query: "q", timer: null, started: true, done: true, attempts: 1, score: 0.71 };
  voice.lastContext = { turn: 1, content: "马来西亚的电子货币牌照由 BNM 监管。" };

  voice.handleEvent(DELEGATION_EVENT);

  assert.equal(sent.length, 1, "one append answers the delegation");
  assert.equal(sent[0].type, "session.thinking.append");
  assert.equal(sent[0].delegation_id, "dlg_1");
  assert.equal(sent[0].content, "马来西亚的电子货币牌照由 BNM 监管。");
});

test("a delegation with no retrieval is released with a no-material note at once", () => {
  const { voice, sent } = makeDelegationVoice();
  voice.ragState = { query: "", timer: null, started: false, done: false, attempts: 0, score: 0 };

  voice.handleEvent(DELEGATION_EVENT);

  assert.equal(sent.length, 1);
  assert.equal(sent[0].delegation_id, "dlg_1");
  assert.match(sent[0].content, /没有检索到额外资料/);
});

test("an in-flight retrieval answers the delegation with its context", async () => {
  const { voice, sent } = makeDelegationVoice({ delegationTimeoutMs: 40 });
  voice.user = { open: true, text: "马来西亚的电子货币牌照由谁监管", last: 0 };
  let resolveFetch;
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => new Promise((resolve) => (resolveFetch = resolve));
  try {
    voice.tick(2, false); // user.final starts the lookup
    await new Promise((r) => setTimeout(r, 0));
    voice.handleEvent(DELEGATION_EVENT);
    assert.equal(sent.length, 0, "does not release before the lookup returns");

    resolveFetch({ ok: true, json: async () => ({ context: "BNM 监管", score: 0.71, sources: [] }) });
    await new Promise((r) => setTimeout(r, 10));

    assert.equal(sent.length, 1);
    assert.equal(sent[0].delegation_id, "dlg_1");
    assert.equal(sent[0].content, "BNM 监管");

    await new Promise((r) => setTimeout(r, 60)); // past the shortened watchdog
    assert.equal(sent.length, 1, "watchdog does not append a second time");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("a retrieval miss releases a waiting delegation instead of freezing", async () => {
  const { voice, sent } = makeDelegationVoice({ delegationTimeoutMs: 500 });
  voice.user = { open: true, text: "今天北京天气怎么样", last: 0 };
  let resolveFetch;
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => new Promise((resolve) => (resolveFetch = resolve));
  try {
    voice.tick(2, false);
    await new Promise((r) => setTimeout(r, 0));
    voice.handleEvent(DELEGATION_EVENT);
    assert.equal(sent.length, 0);

    resolveFetch({ ok: true, json: async () => ({ context: "", reason: "below_threshold" }) });
    await new Promise((r) => setTimeout(r, 10));

    assert.equal(sent.length, 1);
    assert.equal(sent[0].delegation_id, "dlg_1");
    assert.match(sent[0].content, /没有检索到额外资料/);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("the watchdog releases an unanswered delegation exactly once", async () => {
  // The shipped fallback is 2 s; this test runs the same path with 30 ms.
  assert.equal(DELEGATION_WATCHDOG_MS, 2000);
  const { voice, sent } = makeDelegationVoice({ delegationTimeoutMs: 30 });
  voice.ragState = { query: "q", timer: null, started: true, done: false, attempts: 1, score: 0 };

  voice.handleEvent(DELEGATION_EVENT);
  assert.equal(sent.length, 0, "waits for the in-flight retrieval");
  await new Promise((r) => setTimeout(r, 60));

  assert.equal(sent.length, 1, "watchdog released it once");
  assert.equal(sent[0].delegation_id, "dlg_1");
  assert.match(sent[0].content, /没有检索到额外资料/);

  await new Promise((r) => setTimeout(r, 60));
  assert.equal(sent.length, 1, "never released twice");
});

// ---- RAG reason handling -----------------------------------------------------
// The broker answers with reason hit | below_threshold | greeting | breaker_open
// (and confidence high | low on a hit). Every reason must still answer the
// delegation; only breaker_open and greeting add a quiet note, and neither may
// mention an outage or a retrieval problem.
const NEGATIVE_TALK = /系统繁忙|没资料|没有资料|资料不足|服务不可用|网络异常|检索不到|系统故障|稍后再试/;

const L0_INSTRUCTIONS = [
  "# Annie",
  "展馆知识卡（只能依据这些卡片答题，卡片外的问题要说明不知道）：",
  "- 讲解点一：随申办企业云：随申办企业云是一网通办的企业版，设置五个频道。",
  "- 讲解点二：城市视觉中枢：统一汇聚全市视频资源，输出视频调阅和视频计算能力。",
].join("\n");

function captureEvents(voice) {
  const events = { miss: [], context: [], released: [], created: [] };
  voice.bus.on("rag.miss", (e) => events.miss.push(e));
  voice.bus.on("rag.context", (e) => events.context.push(e));
  voice.bus.on("delegation.released", (e) => events.released.push(e));
  voice.bus.on("delegation.created", (e) => events.created.push(e));
  return events;
}

// Starts one lookup on user.final, lets a delegation arrive while it is in
// flight, then returns the stub answer. This is the ordering that used to stall.
async function runLookup({ answer, text, instructions, l0Cards }) {
  const h = makeDelegationVoice({ instructions, l0Cards });
  const events = captureEvents(h.voice);
  const realFetch = globalThis.fetch;
  let resolveFetch;
  globalThis.fetch = () => new Promise((resolve) => (resolveFetch = resolve));
  try {
    h.voice.user = { open: true, text, last: 0 };
    h.voice.tick(2, false); // user.final starts the lookup
    await new Promise((r) => setTimeout(r, 0));
    h.voice.handleEvent(DELEGATION_EVENT);
    assert.equal(h.sent.length, 0, "nothing appended while the lookup is in flight");
    resolveFetch({ ok: true, json: async () => answer });
    await new Promise((r) => setTimeout(r, 20));
    return { ...h, events };
  } finally {
    globalThis.fetch = realFetch;
  }
}

test("breaker_open: 5/5 rounds answer without outage talk and release the delegation", async () => {
  for (let i = 0; i < 5; i++) {
    const { voice, sent, events } = await runLookup({
      text: "帮我订明天去北京的机票",
      answer: { context: "", reason: "breaker_open", retry_after_ms: 10000 },
    });
    assert.equal(sent.length, 1, `round ${i}: exactly one append`);
    assert.equal(sent[0].delegation_id, "dlg_1");
    assert.ok(sent[0].content.includes(BREAKER_FALLBACK_LINE), `round ${i}: fallback line`);
    assert.doesNotMatch(sent[0].content, NEGATIVE_TALK, `round ${i}: no outage talk`);
    assert.equal(events.context.length, 0, `round ${i}: no source card`);
    assert.equal(events.miss[0].fallback, "line");
    assert.equal(events.released.length, 1, `round ${i}: released once`);
    assert.equal(events.released[0].reason, "rag-breaker_open");
    assert.equal(voice.delegation.released, true, `round ${i}: nothing left pending`);
  }
});

test("breaker_open answers from a matching L0 card when there is one", async () => {
  for (let i = 0; i < 5; i++) {
    const { sent, events } = await runLookup({
      text: "随申办企业云有哪些频道",
      answer: { context: "", reason: "breaker_open", retry_after_ms: 10000 },
      instructions: L0_INSTRUCTIONS,
    });
    assert.equal(sent.length, 1);
    assert.ok(sent[0].content.includes("五个频道"), `round ${i}: card text used`);
    assert.ok(!sent[0].content.includes(BREAKER_FALLBACK_LINE), `round ${i}: no deflection line`);
    assert.doesNotMatch(sent[0].content, NEGATIVE_TALK);
    assert.equal(events.miss[0].fallback, "l0-card");
    assert.equal(events.released.length, 1);
  }
});

test("greeting: 5/5 rounds reply socially with no material and no source card", async () => {
  for (let i = 0; i < 5; i++) {
    const { voice, sent, events } = await runLookup({
      text: "你好",
      answer: { context: "", reason: "greeting" },
    });
    assert.equal(sent.length, 1, `round ${i}: exactly one append`);
    assert.equal(sent[0].delegation_id, "dlg_1");
    assert.match(sent[0].content, /打招呼/, `round ${i}: social note`);
    assert.doesNotMatch(sent[0].content, NEGATIVE_TALK);
    assert.equal(events.context.length, 0, `round ${i}: no source card`);
    assert.equal(events.miss[0].reason, "greeting");
    assert.equal(events.miss[0].fallback, "greeting");
    assert.equal(events.released.length, 1);
    assert.equal(voice.delegation.released, true);
  }
});

test("below_threshold: 5/5 rounds keep the shipped miss behaviour", async () => {
  for (let i = 0; i < 5; i++) {
    const { voice, sent, events } = await runLookup({
      text: "今天北京天气怎么样",
      answer: { context: "", reason: "below_threshold" },
    });
    assert.equal(sent.length, 1, `round ${i}: delegation still answered`);
    assert.equal(sent[0].delegation_id, "dlg_1");
    assert.equal(sent[0].content, DELEGATION_FALLBACK, `round ${i}: unchanged miss note`);
    assert.doesNotMatch(sent[0].content, NEGATIVE_TALK);
    assert.equal(events.context.length, 0);
    assert.equal(events.released[0].reason, "rag-miss");
    assert.equal(voice.delegation.released, true);
  }
});

const SPOKEN_QUALIFIERS = /资料中提到|资料显示|知识库里说|知识库中|检索显示|根据我查到的/;

test("confidence low: 5/5 rounds inject an instruction, never a spoken qualifier", async () => {
  for (let i = 0; i < 5; i++) {
    const { voice, sent, events } = await runLookup({
      text: "随申办企业云有几个频道",
      answer: {
        context: "随申办企业云设置五个频道。",
        reason: "hit",
        confidence: "low",
        score: 0.58,
        sources: ["随申办企业云介绍.pdf"],
      },
    });
    assert.equal(sent.length, 1);
    assert.equal(sent[0].delegation_id, "dlg_1");
    // Instruction form: parenthesised, imperative, scoped to the model.
    assert.match(sent[0].content, /^（本条材料相关性偏低/, `round ${i}: note leads`);
    assert.match(sent[0].content, /只讲材料里能直接确认的事实/, `round ${i}: fact-only instruction`);
    assert.match(sent[0].content, /这个我说不太准/, `round ${i}: uncertainty line is allowed`);
    assert.doesNotMatch(sent[0].content, SPOKEN_QUALIFIERS, `round ${i}: no spoken qualifier`);
    assert.doesNotMatch(sent[0].content, NEGATIVE_TALK);
    // The material itself still reaches the model unchanged.
    assert.ok(sent[0].content.includes("随申办企业云设置五个频道。"), `round ${i}: material kept`);
    assert.equal(events.context.length, 1);
    assert.equal(events.context[0].confidence, "low");
    assert.equal(events.context[0].reason, "hit");
    assert.equal(events.released[0].reason, "rag-context");
    assert.equal(voice.delegation.released, true);
  }
});

test("confidence high: no hedge is added, and the broker's ok is reported as hit", async () => {
  const { sent, events } = await runLookup({
    text: "随申办企业云有几个频道",
    answer: {
      context: "随申办企业云设置五个频道。",
      reason: "ok", // the broker's spelling
      confidence: "high",
      score: 0.89,
      sources: ["随申办企业云介绍.pdf"],
    },
  });
  assert.equal(sent.length, 1);
  assert.equal(sent[0].content, "随申办企业云设置五个频道。");
  assert.equal(events.context[0].confidence, "high");
  assert.equal(events.context[0].reason, "hit");
});

test("a delegation arriving after a breaker_open lookup still gets the fallback note", async () => {
  const h = makeDelegationVoice({ instructions: L0_INSTRUCTIONS });
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ context: "", reason: "breaker_open" }),
  });
  try {
    h.voice.user = { open: true, text: "帮我订明天去北京的机票", last: 0 };
    h.voice.tick(2, false);
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(h.sent.length, 1, "note appended without a delegation");
    h.voice.handleEvent(DELEGATION_EVENT);
    assert.equal(h.sent.length, 2, "delegation answered from the cached note");
    assert.equal(h.sent[1].delegation_id, "dlg_1");
    assert.ok(h.sent[1].content.includes(BREAKER_FALLBACK_LINE));
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("L0 cards parse from the persona block and match by overlap, not exact title", () => {
  const cards = parseL0Cards(L0_INSTRUCTIONS);
  assert.equal(cards.length, 2);
  assert.ok(cards[0].text.startsWith("随申办企业云："));
  assert.equal(findL0Card(cards, "随申办企业云有哪些频道")?.text.includes("五个频道"), true);
  assert.equal(findL0Card(cards, "城市视觉中枢是做什么的")?.text.includes("视频资源"), true);
  assert.equal(findL0Card(cards, "你好"), null);
});
