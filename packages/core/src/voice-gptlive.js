import { LiveAnalyser, LiveMouth } from "./lipsync.js";

// GPT-Live-1 over browser WebRTC. The broker does the SDP exchange with the server
// key (GPT-Live has no ephemeral client keys); audio flows browser <-> OpenAI and
// never touches fal. Exchange and event names: broker/UPSTREAM.md §1.3-1.4.
// GPT-Live sends no turn-done, speech-started, or audio-done events, so finals,
// audio start/end, and barge-in are inferred here. Measured on a live session:
// transcript deltas arrive in bursts up to ~2.6 s apart mid-sentence, and the user's
// input transcript lags their voice by ~1.4 s, longer than GPT-Live takes to yield.
// So turn ends use audio energy *and* transcript, and barge-in uses a local mic VAD.

const TURN_QUIET_S = 1.5; // no Annie audio and no transcript this long ends her turn
const BARGEIN_DUCK_S = 0.05; // two voiced frames over her speech silence her playout
const BARGEIN_HOLD_S = 0.35; // sustained this long confirms a barge-in (a cough does not)
const BARGEIN_CANCEL_S = 0.25; // quiet this long cancels a duck that never confirmed
const PLAYOUT_RESUME_AFTER_S = 0.4; // do not resume on the tail of the cut turn
const SESSION_WARN_S = 20; // fallback when the broker omits warn_s
// A kiosk left open in front of nobody still bills every second of the WebRTC
// call. The broker reaps idle sessions too, but only while its sideband attach
// is up; when that fallback runs on the wall clock, silence runs to the hard
// cap. So the browser closes the call after this much true silence: no user
// voice, no Annie speech, no transcript on either side.
export const IDLE_DISCONNECT_S = 90;
const MIC_GATE = 0.15; // mic aperture treated as voice (~ -37 dBFS)
// Retrieval starts on partial transcripts, ~1.5 s before user.final, so the
// appended facts can land before GPT-Live begins the answer.
const RAG_PARTIAL_DEBOUNCE_MS = 500;
// Too short a prefix ("马来西亚的") retrieves the wrong document; wait for enough
// words to be discriminative while still ahead of the model's reply.
const RAG_MIN_PARTIAL_CHARS = 10;
// A partial transcript can match a weaker document. At user.final, retry with the
// full text and append only if it scores better than what is already in context.
const RAG_FINAL_RETRY_SCORE = 0.7;
// GPT-Live uses client delegation: when it says "let me check", it emits
// session.delegation.created and waits. Nothing else answers that event, so an
// unanswered one freezes the turn. Every delegation is released with context, a
// "no extra material" note, or this watchdog, whichever comes first.
export const DELEGATION_WATCHDOG_MS = 2000;
export const DELEGATION_FALLBACK =
  "（本轮没有检索到额外资料。请直接依据你已经知道的信息和已注入的上下文回答；如果确实不确定，就坦率地说不知道。不要再说要去查、确认或稍等。）";

// Retrieval answers carry one of four reasons (hit, below_threshold, greeting,
// breaker_open) and a hit carries confidence (high, low). Each has exactly one
// voice behavior, and every one of them must still release a waiting delegation:
// an answered turn that mentions an outage is worse than a short deflection.
export const BREAKER_FALLBACK_LINE =
  "这个我这边一时说不太准，您可以问我随申办企业云、城市视觉中枢这些展厅业务。";
export const BREAKER_FALLBACK_NOTE = `（这一轮只说这一句，然后结束：「${BREAKER_FALLBACK_LINE}」）`;
export const GREETING_NOTE = "（访客只是在打招呼，直接自然地回应就好，不需要讲展项内容。）";
// An instruction for the model, never a line to speak: the persona answers
// directly, so a low-confidence hit is narrowed to what the material states
// rather than prefixed with a spoken hedge.
export const LOW_CONFIDENCE_NOTE =
  "（本条材料相关性偏低，只供内部参考：只讲材料里能直接确认的事实，不推断、不补数字、不扩写；把握不足就说「这个我说不太准」。直接给答案，不要提及材料来源、文档或检索过程。）";

// L0 cards ride in with the session instructions as "- title：text" lines. The
// title itself may contain a colon ("讲解点一：随申办企业云"), so the line is not
// split; the best-overlapping line is the match.
const L0_CARD_HEADER = "展馆知识卡";
const L0_MATCH_MIN_CHARS = 4;

function normalizeText(value) {
  return String(value ?? "").toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, "");
}

function longestCommonSubstring(a, b) {
  let best = 0;
  let prev = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1;
        if (cur[j] > best) best = cur[j];
      }
    }
    prev = cur;
  }
  return best;
}

export function parseL0Cards(instructions) {
  const text = String(instructions ?? "");
  const at = text.indexOf(L0_CARD_HEADER);
  if (at < 0) return [];
  const cards = [];
  for (const raw of text.slice(at).split("\n").slice(1)) {
    if (!raw.startsWith("- ")) {
      if (cards.length) break; // the card block ended
      continue;
    }
    const line = raw.slice(2).trim().replace(/^讲解点[一二三四五六七八九十\d]+[：:]\s*/, "");
    if (line) cards.push({ text: line });
  }
  return cards;
}

export function findL0Card(cards, query) {
  const q = normalizeText(query);
  if (!q) return null;
  let best = null;
  for (const card of cards ?? []) {
    const n = longestCommonSubstring(q, normalizeText(card.text));
    if (n >= L0_MATCH_MIN_CHARS && (!best || n > best.n)) best = { card, n };
  }
  return best?.card ?? null;
}

export class GptLiveVoice {
  // `mic` may be any MediaStream (a synthetic one drives scripted live recordings);
  // `onRaw` sees every data-channel event, for logging.
  constructor({ bus, brokerUrl, voice = "marin", instructions, clock, mic, onRaw, rag = null, delegationTimeoutMs, l0Cards, idleDisconnectS }) {
    Object.assign(this, { bus, brokerUrl: brokerUrl.replace(/\/$/, ""), voice, instructions, clock, mic, onRaw, rag });
    this.delegationTimeoutMs = delegationTimeoutMs ?? DELEGATION_WATCHDOG_MS;
    // null until the call is up: a session that never connected has nothing to bill.
    this.lastActivity = null;
    this.idleDisconnectS = idleDisconnectS ?? IDLE_DISCONNECT_S;
    this.closed = false;
    // Provided explicitly for tests/callers, otherwise lifted from the cards
    // that already ride in with the persona.
    this.l0Cards = l0Cards ?? parseL0Cards(instructions);
    this.user = { text: "", last: -Infinity, open: false };
    this.asst = { text: "", last: -Infinity, open: false };
    this.lastAnnieActive = -Infinity; // last time her audio or transcript showed activity
    this.userVoice = { since: null, overSince: null, quietSince: null, overAnnie: false, barged: false };
    this.usageSeconds = 0;
    this.ragTurn = 0; // only the newest turn may append; older replies are dropped
    // One retrieval per user utterance, started as early as the partial text allows.
    this.ragState = { query: "", timer: null, started: false, done: false, attempts: 0, score: 0 };
    // The last context appended for this utterance, kept so a delegation that
    // arrives after retrieval can be answered without a second lookup.
    this.lastContext = null; // { turn, content }
    // The delegation GPT-Live is currently waiting on (at most one is live).
    this.delegation = { id: null, since: 0, timer: null, released: false };
    // Local playout state: a live WebRTC stream has no queue, so pausing the element
    // is what stops audible speech, and resuming it rejoins the live stream.
    this.playout = { stopped: false, stoppedAt: null, lastStopMs: null, stops: 0 };
    this.sessionWarnS = SESSION_WARN_S;
    this.capTimer = null;
  }

  // The session response says whether the broker has a knowledge base wired.
  enableRag(rag) {
    this.rag = rag?.enabled ? rag : null;
  }

  async connect() {
    this.pc = new RTCPeerConnection();
    this.mic ??= await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    for (const t of this.mic.getTracks()) this.pc.addTrack(t, this.mic);
    this.dc = this.pc.createDataChannel("oai-events"); // must exist before the offer
    this.dc.onmessage = (m) => {
      const e = JSON.parse(m.data);
      this.onRaw?.(e);
      this.handleEvent(e);
    };
    this.audioCtx = new AudioContext();
    this.micAnalyser = new LiveAnalyser(this.audioCtx, this.audioCtx.createMediaStreamSource(this.mic));
    this.pc.ontrack = (e) => {
      // Audible playout stays on the element so the platform echo canceller sees it;
      // the analysers (energy + HeadAudio worklet) tap the same stream for the mouth.
      this.el = Object.assign(new Audio(), { autoplay: true, srcObject: e.streams[0] });
      this.remote = e.streams[0];
      this.analyser = new LiveMouth(this.audioCtx, this.audioCtx.createMediaStreamSource(e.streams[0]));
      // The SDP exchange outlives the click's user activation, so autoplay can be
      // refused. Ask explicitly and let the stage offer one tap to unblock it.
      try {
        const played = this.el.play();
        if (played && typeof played.catch === "function") {
          played.catch(() => this.bus.emit("playback.blocked", {}));
        }
      } catch {
        this.bus.emit("playback.blocked", {});
      }
    };
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await new Promise((res) => {
      if (this.pc.iceGatheringState === "complete") return res();
      this.pc.onicegatheringstatechange = () => this.pc.iceGatheringState === "complete" && res();
    });
    const r = await fetch(`${this.brokerUrl}/session`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sdp: this.pc.localDescription.sdp, voice: this.voice, instructions: this.instructions }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `session ${r.status}`);
    const s = await r.json();
    this.sessionId = s.session_id;
    this.token = s.token;
    this.expiresAt = s.expires_at;
    this.sessionWarnS = Number.isFinite(s.warn_s) ? s.warn_s : SESSION_WARN_S;
    // The call is up and billable from here; the idle countdown starts now.
    this.closed = false;
    this.lastActivity = this.clock.now();
    await this.pc.setRemoteDescription({ type: "answer", sdp: s.sdp });
    this.#armCapWarning();
    return s;
  }

  // Public so transports and tests can feed one data-channel event.
  handleEvent(e) {
    this.#event(e);
  }

  #event(e) {
    const now = this.clock.now();
    switch (e.type) {
      case "session.input_transcript.delta": {
        const u = this.user;
        if (!u.open) {
          (u.open = true), (u.text = "");
          this.#resetRagTurn(); // a new utterance gets its own retrieval
        }
        u.text += e.delta;
        u.last = now;
        const text = u.text.trim();
        this.bus.emit("user.partial", { text });
        if (text.length >= RAG_MIN_PARTIAL_CHARS) this.#scheduleRag(text);
        break;
      }
      case "session.output_transcript.delta": {
        const a = this.asst;
        if (!a.open) {
          a.open = true;
          a.text = "";
          // A stopped playout rejoins only when a new turn starts, never on the
          // tail of the turn we just cut.
          if (this.playout.stopped && now - this.playout.stoppedAt >= PLAYOUT_RESUME_AFTER_S) {
            this.#resumePlayout("new-turn");
          }
          this.bus.emit("assistant.audio.start", {});
        }
        a.text += e.delta;
        a.last = now;
        this.lastAnnieActive = now;
        this.bus.emit("assistant.partial", { text: a.text.trim() });
        break;
      }
      // Client delegation: the model has stopped and is waiting for us. It never
      // resolves on its own, so every path below ends in an append.
      case "session.delegation.created":
        this.#onDelegationCreated(e, now);
        break;
      case "session.usage.updated":
        this.usageSeconds = e.usage?.seconds ?? this.usageSeconds;
        this.bus.emit("usage", { seconds: this.usageSeconds });
        break;
      case "session.closed":
        this.#clearDelegation();
        this.#stopPlayout("session-closed", 0);
        this.bus.emit("session.closed", { reason: e.reason });
        break;
      case "error":
        this.bus.emit("voice.error", { code: e.error?.code, message: e.error?.message });
        break;
    }
  }

  // Called every frame: returns mouth features and runs the turn/barge-in inference.
  tick(now, mouthSpeaking) {
    this.speaking = mouthSpeaking;
    if (mouthSpeaking) this.lastAnnieActive = now;
    const u = this.user, a = this.asst;
    // Idle shutdown, checked on the same frame clock as everything else: the
    // countdown only runs while the call is up, and resets on any sign of life
    // on either side of the conversation.
    if (this.lastActivity != null && !this.closed) {
      const micAmp = this.micAnalyser?.sample()?.amp ?? 0;
      const isActive = mouthSpeaking || micAmp > MIC_GATE || u.open || a.open;
      if (isActive) this.lastActivity = now;
      else if (now - this.lastActivity >= this.idleDisconnectS) {
        const idleS = Math.round(now - this.lastActivity);
        this.bus.emit("session.idle", { idle_s: idleS });
        this.close("idle");
      }
    }
    if (u.open && now - u.last > TURN_QUIET_S) {
      u.open = false;
      const text = u.text.trim();
      this.bus.emit("user.final", { text });
      this.#ragForFinal(text);
    }
    if (a.open && now - this.lastAnnieActive > TURN_QUIET_S) {
      a.open = false;
      this.bus.emit("assistant.final", { text: a.text.trim() });
      this.bus.emit("assistant.audio.end", {});
    }
    // Local mic VAD: a sustained voice onset over her speech is a barge-in; a short
    // "mm-hmm" is not. Runs on the mic we send, so it beats the input transcript.
    const m = this.micAnalyser?.sample();
    this.micLevel = m?.amp ?? 0; // for the HUD waveform
    const v = this.userVoice;
    if (m && m.amp > MIC_GATE) {
      v.quietSince = null;
      if (v.since == null) {
        v.since = now;
        v.overAnnie = mouthSpeaking || now - this.lastAnnieActive < 0.2;
        v.overSince = v.overAnnie ? now : null;
      } else if (v.overAnnie && v.overSince == null) {
        v.overSince = now; // her speech began after the voice: count from here
      }
      if (v.overAnnie && v.overSince != null) {
        // Silence her after two voiced frames. The 350 ms hold still decides
        // whether it was a real barge-in or a cough, but the stop is immediate.
        if (!this.playout.stopped && now - v.overSince >= BARGEIN_DUCK_S) {
          this.#stopPlayout("user-voice", now - v.overSince);
        }
        if (!v.barged && now - v.overSince >= BARGEIN_HOLD_S) {
          v.barged = true;
          this.bus.emit("bargein", {});
          // She yields on her own; what she says next is a new turn, not the cut one.
          if (this.asst.open) {
            this.asst.open = false;
            this.bus.emit("assistant.final", { text: this.asst.text.trim() });
            this.bus.emit("assistant.audio.end", {});
          }
        }
      }
    } else if (v.since != null) {
      v.quietSince ??= now;
      if (now - v.quietSince > BARGEIN_CANCEL_S) {
        const wasBarged = v.barged;
        v.since = null;
        v.overSince = null;
        v.barged = false;
        // A short "mm-hmm" only ducked her; let the live stream resume.
        if (!wasBarged && this.playout.stopped) this.#resumePlayout("short-blip");
      }
    }
    this.userVoiced = v.since != null;
    const f = this.analyser?.sample() ?? null;
    // While paused the remote stream keeps producing samples; keep the mouth shut.
    return this.playout.stopped && f
      ? { ...f, amp: 0, db: -120, vowels: null, visemes: null, ha: false }
      : f;
  }

  #stopPlayout(reason, sustainedS) {
    if (!this.el) return;
    const t0 = performance.now();
    try {
      this.el.pause();
    } catch {}
    const pauseMs = performance.now() - t0;
    if (!this.playout.stopped) {
      this.playout.stopped = true;
      this.playout.stoppedAt = this.clock.now();
      this.playout.lastStopMs = Math.round(sustainedS * 1000 + pauseMs);
      this.playout.stops++;
      this.bus.emit("playback.stopped", {
        reason,
        latency_ms: this.playout.lastStopMs,
        pause_ms: Number(pauseMs.toFixed(2)),
      });
    }
  }

  #resumePlayout(reason) {
    if (!this.el || !this.playout.stopped) return;
    try {
      const played = this.el.play();
      if (played && typeof played.catch === "function") played.catch(() => {});
    } catch {}
    this.playout.stopped = false;
    this.playout.stoppedAt = null;
    this.bus.emit("playback.resumed", { reason });
  }

  // ---- delegation -----------------------------------------------------------
  // GPT-Live's client delegation has no other client: if we do not answer, the
  // turn stalls and later questions are ignored. Release is idempotent per id.
  #onDelegationCreated(e, now) {
    const d = e.delegation ?? {};
    if (!d.id) return;
    // A second delegation while one is pending must not orphan the first.
    if (this.#pendingDelegation()) this.#releaseDelegation("superseded", this.#releaseContent());
    this.delegation = { id: d.id, target: d.target ?? null, since: now, timer: null, released: false };
    this.bus.emit("delegation.created", { id: d.id, target: d.target ?? null, type: d.type ?? null });
    // Retrieval already answered this utterance: hand the context straight back.
    if (this.lastContext?.content) {
      this.#releaseDelegation("have-context", this.lastContext.content);
      return;
    }
    // A lookup is still running; let it answer. The watchdog is the net.
    if (this.ragState.started && !this.ragState.done) {
      this.#armDelegationWatchdog();
      return;
    }
    this.#releaseDelegation("no-context", null);
  }

  #pendingDelegation() {
    return this.delegation.id && !this.delegation.released ? this.delegation : null;
  }

  #releaseContent() {
    return this.lastContext?.content ?? DELEGATION_FALLBACK;
  }

  #armDelegationWatchdog() {
    const d = this.delegation;
    clearTimeout(d.timer);
    d.timer = setTimeout(() => {
      if (this.#pendingDelegation()?.id === d.id) this.#releaseDelegation("watchdog", this.#releaseContent());
    }, this.delegationTimeoutMs);
  }

  // A successful append that already carried the delegation id counts as the
  // release; only the bookkeeping and the event remain.
  #markDelegationReleased(reason) {
    const d = this.#pendingDelegation();
    if (!d) return;
    clearTimeout(d.timer);
    d.timer = null;
    d.released = true;
    this.bus.emit("delegation.released", {
      id: d.id,
      reason,
      ms: Math.max(0, Math.round((this.clock.now() - d.since) * 1000)),
      appended: true,
    });
  }

  #releaseDelegation(reason, content) {
    const d = this.#pendingDelegation();
    if (!d) return false;
    clearTimeout(d.timer);
    d.timer = null;
    d.released = true;
    const sent = this.#appendThinking(content || DELEGATION_FALLBACK, d.id);
    this.bus.emit("delegation.released", {
      id: d.id,
      reason,
      ms: Math.max(0, Math.round((this.clock.now() - d.since) * 1000)),
      appended: sent,
    });
    return sent;
  }

  #clearDelegation() {
    clearTimeout(this.delegation.timer);
    this.delegation = { id: null, since: 0, timer: null, released: false };
  }

  // The broker caps every session; say a wrap-up line before the cap closes it.
  #armCapWarning() {
    clearTimeout(this.capTimer);
    if (!this.expiresAt) return;
    const delayMs = (this.expiresAt - Date.now() / 1000 - this.sessionWarnS) * 1000;
    if (delayMs <= 0) return;
    this.capTimer = setTimeout(() => this.#announceEnding(), delayMs);
  }

  #announceEnding() {
    const seconds = this.sessionWarnS;
    const text = `本次体验还剩 ${seconds} 秒，感谢您的参与。`;
    this.bus.emit("session.warning", { text, seconds });
    try {
      if (typeof speechSynthesis !== "undefined" && typeof SpeechSynthesisUtterance !== "undefined") {
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "zh-CN";
        u.rate = 1;
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
      }
    } catch {}
    // Do not append instructions here: that can interrupt her speech. The local
    // voice line and the caption are the warning; the reaper closes at the cap.
  }

  #resetRagTurn() {
    // A delegation belongs to the utterance that created it; a new utterance
    // must not leave the old one waiting.
    if (this.#pendingDelegation()) this.#releaseDelegation("new-turn", this.#releaseContent());
    clearTimeout(this.ragState.timer);
    this.lastContext = null;
    this.ragState = { query: "", timer: null, started: false, done: false, attempts: 0, score: 0 };
  }

  #allowRagRetry() {
    // A partial can be too garbled to match; the final text gets one more try.
    if (this.ragState.attempts >= 2) return;
    this.ragState.started = false;
    this.ragState.query = "";
  }

  // Start retrieval on partial text instead of waiting for user.final: the final
  // event fires 1.5 s after silence, by which time GPT-Live is already answering.
  #scheduleRag(text) {
    const q = text.trim();
    if (this.ragState.started || this.ragState.done || !q || q === this.ragState.query) return;
    this.ragState.query = q;
    clearTimeout(this.ragState.timer);
    this.ragState.timer = setTimeout(() => this.#lookupRag(q), RAG_PARTIAL_DEBOUNCE_MS);
  }

  #ragForFinal(text) {
    clearTimeout(this.ragState.timer);
    const q = (text || "").trim();
    if (!q || q.length < 2) return;
    const state = this.ragState;
    if (!state.started && !state.done) {
      this.#lookupRag(q); // transcript arrived in one chunk; no partial ran
      return;
    }
    // The partial matched something weak; the fuller final text may be better.
    const grew = q.length >= state.query.length + 4;
    if (state.done && state.score < RAG_FINAL_RETRY_SCORE && grew && state.attempts < 2) {
      state.started = false;
      state.done = false;
      state.query = q;
      this.#lookupRag(q);
    }
  }

  // Append the snippet to the live session instructions. Additive by design: a
  // miss, a timeout or a broker without a knowledge base leave the call as it was.
  async #lookupRag(text) {
    if (this.ragState.started || !this.rag?.enabled || !text || text.length < 2 || !this.token) return;
    this.ragState.started = true;
    this.ragState.attempts++;
    const turn = ++this.ragTurn;
    const budgetMs = (this.rag.timeout_ms ?? 1200) + 300;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), budgetMs);
    try {
      const r = await fetch(`${this.brokerUrl}/rag`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ token: this.token, query: text }),
        signal: ctrl.signal,
      });
      if (!r.ok) {
        this.#allowRagRetry();
        return;
      }
      const answer = await r.json();
      if (turn !== this.ragTurn) return; // a newer user turn superseded this one
      // The broker spells a hit "ok"; the voice contract (and the source card)
      // calls it "hit".
      const rawReason = answer.reason ?? (answer.context ? "hit" : "empty");
      const reason = rawReason === "ok" ? "hit" : rawReason;
      if (!answer.context) {
        this.#handleNoContext(answer, reason, text, turn);
        return;
      }
      const confidence = answer.confidence === "low" ? "low" : "high";
      // The broker keeps low confidence in the `confidence` field only; the
      // voice layer adds an instruction, never a spoken qualifier, so the
      // answer stays direct and nothing about the lookup is read aloud.
      const content = confidence === "low" && !answer.context.includes("本条材料相关性偏低")
        ? `${LOW_CONFIDENCE_NOTE}\n${answer.context}`
        : answer.context;
      const score = Number(answer.score ?? 0);
      // Never let a weaker final hit replace a better partial one.
      const better = !this.ragState.score || score >= this.ragState.score;
      // Quiet context, not instructions: an instructions.append can interrupt
      // speech in progress, which is what made RAG answers stall or restart.
      // A pending delegation gets this context keyed to its id.
      const pending = this.#pendingDelegation();
      const sent = better ? this.#appendThinking(content, pending?.id ?? null) : false;
      if (sent) this.lastContext = { turn, content };
      // The append above was the release when it carried the delegation id.
      if (sent && pending) this.#markDelegationReleased("rag-context");
      // A weaker hit must still release a waiting delegation.
      if (pending && !sent) this.#releaseDelegation("weaker-context", this.#releaseContent());
      this.ragState.done = true;
      this.ragState.score = Math.max(this.ragState.score, score);
      this.bus.emit("rag.context", {
        query: text,
        sources: answer.sources ?? [],
        // `low` means the snippet sits in the 0.55–0.62 band: the source card
        // marks it for the front end, while the spoken answer stays direct.
        confidence,
        reason,
        score,
        bytes: content.length,
        appended: sent,
        at: this.clock.now(),
      });
    } catch {
      // Retrieval is never allowed to affect the call; a delegation still needs
      // an answer, so release it rather than leave the turn frozen.
      if (this.#pendingDelegation()) this.#releaseDelegation("rag-error", this.#releaseContent());
      this.#allowRagRetry();
    } finally {
      clearTimeout(timer);
    }
  }

  // A retrieval with no context still has to move the turn along and free any
  // delegation. `breaker_open` and `greeting` add a note; below_threshold and
  // the other misses keep the behaviour that was already shipped.
  #handleNoContext(answer, reason, text, turn) {
    const event = { query: text, reason, retry_after_ms: answer.retry_after_ms ?? 0 };
    if (reason === "breaker_open") {
      // The L0 cards are already in the session instructions; a matching card
      // is a safe answer even while retrieval is down.
      const card = findL0Card(this.l0Cards, text);
      const content = card
        ? `（请只依据这张展馆知识卡回答，用一两句口语说完就结束：\n${card.text}）`
        : BREAKER_FALLBACK_NOTE;
      this.#appendNote(content, turn, reason);
      this.bus.emit("rag.miss", { ...event, fallback: card ? "l0-card" : "line" });
      return;
    }
    if (reason === "greeting") {
      // No material, no source card: just answer the greeting socially.
      this.#appendNote(GREETING_NOTE, turn, reason);
      this.bus.emit("rag.miss", { ...event, fallback: "greeting" });
      return;
    }
    this.bus.emit("rag.miss", event);
    // A waiting delegation must hear the miss, not time out.
    if (this.#pendingDelegation()) this.#releaseDelegation("rag-miss", this.#releaseContent());
    this.#allowRagRetry();
  }

  // Append a quiet note, keyed to a pending delegation when there is one; that
  // append is the release, so the delegation is never answered twice.
  #appendNote(content, turn, reason) {
    const pending = this.#pendingDelegation();
    const sent = this.#appendThinking(content, pending?.id ?? null);
    if (sent) {
      this.lastContext = { turn, content };
      this.ragState.done = true;
      this.ragState.score = 0;
    }
    if (sent && pending) this.#markDelegationReleased(`rag-${reason}`);
    else if (pending) this.#releaseDelegation(`rag-${reason}`, content);
    return sent;
  }

  #appendThinking(content, delegationId = null) {
    if (!this.dc || this.dc.readyState !== "open") return false;
    try {
      this.dc.send(JSON.stringify({
        type: "session.thinking.append",
        content,
        delegation_id: delegationId,
        event_id: `rag_${this.ragTurn}_${Date.now()}`,
      }));
      return true;
    } catch {
      return false;
    }
  }

  // `reason` is set only when the call ends itself (idle, cap), so the stage can
  // reset the kiosk without waiting for an echo from the broker. A visitor-facing
  // hang-up passes none and the stage resets from the click it already handled.
  close(reason = null) {
    if (this.closed) return;
    this.closed = true;
    this.lastActivity = null;
    clearTimeout(this.capTimer);
    this.#clearDelegation();
    try {
      this.dc?.readyState === "open" && this.dc.send(JSON.stringify({ type: "session.close" }));
    } catch {}
    setTimeout(() => {
      this.#stopPlayout("closed", 0);
      this.pc?.close();
      this.mic?.getTracks().forEach((t) => t.stop());
    }, 1500);
    if (reason) setTimeout(() => this.bus.emit("session.closed", { reason }), 0);
  }
}
