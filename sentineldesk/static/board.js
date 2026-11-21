"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const lanes = {};
  document.querySelectorAll(".lane").forEach((l) => { lanes[l.dataset.lane] = l; });
  const state = { running: false, timer: null, cards: [], busy: 0 };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) }).then((r) => r.json());

  async function waitReady() {
    for (;;) {
      const s = await fetch("/api/board/status").then((r) => r.json());
      $("#mode-triage").textContent = "triage: " + (s.triage_source === "local_tfidf" ? "local TF-IDF" : "pipeline");
      if (s.error) { $("#feed-status").textContent = "feed error"; $("#feed-status").title = s.error; return; }
      if (s.ready) {
        $("#feed-status").textContent = "feed: KB val";
        $("#feed-status").title = s.feed;
        $("#btn-play").disabled = false; $("#btn-burst").disabled = false; $("#btn-edge").disabled = false;
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  async function tick() {
    if (state.busy > 2) return;
    state.busy++;
    try {
      const t = await fetch("/api/board/next").then((r) => r.json());
      addCard(await post("/api/board/ingest", t));
    } finally { state.busy--; }
  }

  function pct(x) { return Math.round(x * 100) + "%"; }

  function addCard(c) {
    state.cards.push(c);
    const lane = lanes[c.lane] || lanes.review;
    const el = document.createElement("article");
    el.className = "tcard new prio-" + (c.triage ? c.triage.priority.label : "none") + (c.incident ? " in-incident" : "") + (c.simulated ? " simulated" : "");
    el.dataset.id = c.id;
    const tri = c.triage;
    const match = c.truth && tri ? (c.truth.queue === tri.queue.label ? '<span class="ok" title="queue matches known label">match</span>' : '<span class="miss" title="known queue: ' + esc(c.truth.queue) + '">miss</span>') : "";
    el.innerHTML =
      '<div class="tc-top"><span class="tc-id">' + esc(c.id) + "</span>" +
      '<span class="tc-lang" title="' + esc(c.language_name) + '">' + esc(c.language_name) + "</span>" +
      (c.simulated ? '<span class="tag sim">SIMULATED</span>' : "") +
      (c.kind === "edge" ? '<span class="tag sim">EDGE CASE</span>' : "") +
      (c.kind === "manual" ? '<span class="tag manual">MANUAL</span>' : "") +
      (c.known_words !== null && c.known_words < 0.3 ? '<span class="tag noise" title="share of words found in the KB vocabulary">UNREADABLE ' + pct(c.known_words) + "</span>" : "") +
      (c.incident ? '<span class="tag inc">' + esc(c.incident) + "</span>" : "") + "</div>" +
      '<p class="tc-subj">' + esc(c.subject) + "</p>" +
      (tri ? '<div class="tc-chips">' +
        '<span class="chip sm">' + esc(tri.type.label) + "</span>" +
        '<span class="chip sm">' + esc(tri.queue.label) + "</span>" +
        '<span class="chip sm prio">' + esc(tri.priority.label) + "</span></div>" +
        '<div class="tc-foot"><span class="conf"><span style="width:' + pct(tri.queue.confidence) + '"></span></span>' +
        '<span class="muted small">queue ' + pct(tri.queue.confidence) + "</span>" + match + "</div>" : "");
    el.addEventListener("click", () => showDetail(c));
    lane.querySelector(".cards").prepend(el);
    setTimeout(() => el.classList.remove("new"), 900);
    const cards = lane.querySelectorAll(".tcard");
    if (cards.length > 40) cards[cards.length - 1].remove();
    if (c.incident) markIncident(c);
    renderStats();
  }

  function markIncident(c) {
    const inc = (c.incidents || []).find((i) => i.id === c.incident);
    if (!inc) return;
    inc.members.forEach((id) => document.querySelectorAll('.tcard[data-id="' + CSS.escape(id) + '"]').forEach((e) => {
      e.classList.add("in-incident");
      if (!e.querySelector(".tag.inc")) e.querySelector(".tc-top").insertAdjacentHTML("beforeend", '<span class="tag inc">' + esc(inc.id) + "</span>");
    }));
    const b = $("#incident-banner");
    b.hidden = false;
    $("#incident-text").textContent = inc.id + ": " + inc.count + " similar tickets in the last 10 minutes. First report: \"" + inc.subject + "\". Shared terms: " + inc.terms.join(", ") + ".";
  }

  function bars(el, counts, cls) {
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const max = Math.max(1, ...entries.map((e) => e[1]));
    el.innerHTML = entries.map(([k, v]) => '<div class="bar ' + (cls ? cls(k) : "") + '"><span class="bar-label">' + esc(k) + '</span><span class="bar-track"><span style="width:' + (v / max * 100) + '%"></span></span><span class="bar-val">' + v + "</span></div>").join("");
  }

  function renderStats() {
    const cs = state.cards;
    const count = (s) => cs.filter((c) => c.status === s).length;
    $("#k-total").textContent = cs.length;
    $("#k-escalate").textContent = count("escalate");
    $("#k-review").textContent = count("review");
    $("#k-draft").textContent = count("draft_ready");
    const judged = cs.filter((c) => c.truth && c.triage);
    $("#k-acc").textContent = judged.length ? pct(judged.filter((c) => c.truth.queue === c.triage.queue.label).length / judged.length) : "n/a";
    const ms = cs.map((c) => c.total_ms).sort((a, b) => a - b);
    $("#k-ms").textContent = ms.length ? Math.round(ms[Math.floor(ms.length / 2)]) : "n/a";
    Object.entries(lanes).forEach(([k, l]) => { l.querySelector(".count").textContent = cs.filter((c) => c.lane === k).length; });
    const q = {}, p = {};
    cs.forEach((c) => { if (c.triage) { q[c.triage.queue.label] = (q[c.triage.queue.label] || 0) + 1; p[c.triage.priority.label] = (p[c.triage.priority.label] || 0) + 1; } });
    if (cs.length) { bars($("#queue-bars"), q); bars($("#prio-bars"), p, (k) => "p-" + k); }
  }

  function showDetail(c) {
    const tri = c.triage;
    const row = (h) => "<tr><th>" + h + "</th><td>" + esc(tri[h].label) + " <span class='muted'>(" + pct(tri[h].confidence) + ")</span></td><td>" + esc(c.truth ? c.truth[h] : "n/a") + "</td></tr>";
    $("#detail").innerHTML =
      "<p><strong>" + esc(c.id) + "</strong> " + esc(c.subject) + "</p>" +
      "<p class='small body'>" + esc(c.body) + "</p>" +
      "<p class='small'>Decision: <strong>" + esc(c.status) + "</strong> to " + esc(c.destination) + "</p>" +
      (c.reasons.length ? "<p class='small'>Reasons: " + c.reasons.map(esc).join(", ") + "</p>" : "") +
      (tri ? "<table class='small'><tr><th></th><th>predicted</th><th>known</th></tr>" + row("type") + row("queue") + row("priority") + "</table>" : "") +
      "<p class='small'>Known words: " + (c.known_words === null ? "no words" : pct(c.known_words)) + ", language " + esc(c.language_name) + " (" + esc(c.language) + ")</p>" +
      "<p class='muted small'>" + c.total_ms + " ms, triage " + esc(c.triage_source) + ", retrieval " + esc(c.retrieval_mode) + "</p>";
  }

  $("#btn-play").addEventListener("click", () => {
    state.running = !state.running;
    $("#btn-play").textContent = state.running ? "Pause stream" : "Start stream";
    clearInterval(state.timer);
    if (state.running) { tick(); state.timer = setInterval(tick, +$("#speed").value); }
  });
  $("#speed").addEventListener("change", () => {
    if (!state.running) return;
    clearInterval(state.timer); state.timer = setInterval(tick, +$("#speed").value);
  });
  $("#btn-burst").addEventListener("click", async () => {
    $("#btn-burst").disabled = true;
    try {
      const items = await post("/api/board/simulate");
      for (const it of items) { addCard(await post("/api/board/ingest", it)); await new Promise((r) => setTimeout(r, 600)); }
    } finally { $("#btn-burst").disabled = false; }
  });
  $("#compose").addEventListener("submit", async (e) => {
    e.preventDefault();
    const subject = $("#c-subject").value, body = $("#c-body").value;
    addCard(await post("/api/board/ingest", { subject, body, kind: "manual" }));
    $("#c-body").value = ""; $("#c-subject").value = "";
  });
  $("#btn-edge").addEventListener("click", async () => {
    $("#btn-edge").disabled = true;
    try {
      const items = await fetch("/api/board/edge-cases").then((r) => r.json());
      for (const it of items) { addCard(await post("/api/board/ingest", it)); await new Promise((r) => setTimeout(r, 400)); }
    } finally { $("#btn-edge").disabled = false; }
  });
  $("#btn-reset").addEventListener("click", async () => {
    await post("/api/board/reset");
    state.cards = [];
    document.querySelectorAll(".cards").forEach((c) => { c.innerHTML = ""; });
    $("#incident-banner").hidden = true;
    $("#queue-bars").innerHTML = $("#prio-bars").innerHTML = '<p class="muted small">Waiting for tickets.</p>';
    renderStats();
  });

  waitReady();
})();
