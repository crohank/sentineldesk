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
    el.addEventListener("click", () => openInbox(c.lane, c.id));
    lane.querySelector(".cards").prepend(el);
    setTimeout(() => el.classList.remove("new"), 900);
    const cards = lane.querySelectorAll(".tcard");
    if (cards.length > 40) cards[cards.length - 1].remove();
    if (c.incident) markIncident(c);
    renderStats();
    if (inbox.open) renderInbox();
  }

  function markIncident(c) {
    const inc = (c.incidents || []).find((i) => i.id === c.incident);
    if (!inc) return;
    state.cards.forEach((x) => { if (inc.members.includes(x.id)) x.incident = inc.id; });
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

  const FOLDERS = [
    ["all", "All tickets", "grey"], ["escalate", "On-call", "red"], ["review", "Human review", "yellow"],
    ["draft_ready", "Draft ready", "green"], ["language_review", "Language / unreadable", "blue"],
    ["needs_clarification", "Needs detail", "grey"],
  ];
  const inbox = { open: false, folder: "all", selected: null };
  const label = (s) => String(s).replace(/_/g, " ");

  function inFolder(c, f) {
    if (f === "all") return true;
    if (f.startsWith("INC-")) return c.incident === f;
    return c.lane === f;
  }

  function setView(v) {
    inbox.open = v === "inbox";
    $("#inbox").hidden = !inbox.open;
    $("#board-view").hidden = inbox.open;
    document.querySelectorAll(".vt").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
    if (inbox.open) renderInbox();
  }

  function openInbox(folder, id) {
    inbox.folder = folder || "all";
    inbox.selected = id || null;
    setView("inbox");
  }

  function ago(t) {
    const s = Math.max(0, Math.round(Date.now() / 1000 - t));
    return s < 60 ? s + "s ago" : Math.round(s / 60) + "m ago";
  }

  function renderInbox() {
    const cs = state.cards;
    const incs = [...new Set(cs.map((c) => c.incident).filter(Boolean))];
    const folders = FOLDERS.concat(incs.map((i) => [i, "Incident " + i, "red"]));
    $("#folders").innerHTML = folders.map(([k, name, color]) =>
      '<button class="folder' + (k === inbox.folder ? " active" : "") + '" data-f="' + esc(k) + '"><span class="dot ' + color + '"></span>' +
      esc(name) + '<span class="count">' + cs.filter((c) => inFolder(c, k)).length + "</span></button>").join("");
    const list = cs.filter((c) => inFolder(c, inbox.folder)).slice().reverse();
    $("#mail-list").innerHTML = list.length ? list.map((c) =>
      '<button class="mail' + (c.id === inbox.selected ? " active" : "") + " prio-" + (c.triage ? c.triage.priority.label : "none") + '" data-id="' + esc(c.id) + '">' +
      '<span class="m-top"><span class="m-id">' + esc(c.id) + '</span><span class="m-time">' + ago(c.received) + "</span></span>" +
      '<span class="m-subj">' + esc(c.subject || "(no subject)") + "</span>" +
      '<span class="m-snip">' + esc(c.body.slice(0, 110)) + "</span>" +
      '<span class="m-tags">' + (c.triage ? '<span class="chip sm">' + esc(c.triage.queue.label) + '</span><span class="chip sm">' + esc(c.triage.priority.label) + "</span>" : "") +
      '<span class="chip sm">' + esc(c.language_name) + "</span>" + (c.incident ? '<span class="tag inc">' + esc(c.incident) + "</span>" : "") +
      (c.simulated ? '<span class="tag sim">SIMULATED</span>' : "") + "</span></button>").join("")
      : '<p class="muted small pad">No tickets in this folder yet.</p>';
    const sel = cs.find((c) => c.id === inbox.selected);
    if (sel) renderReader(sel); else $("#reader").innerHTML = '<p class="muted">Select a ticket.</p>';
  }

  function renderReader(c) {
    const tri = c.triage;
    const row = (h) => "<tr><th>" + h + "</th><td>" + esc(tri[h].label) + " <span class='muted'>" + pct(tri[h].confidence) + "</span></td><td>" + esc(c.truth ? c.truth[h] : "n/a") + "</td></tr>";
    $("#reader").innerHTML =
      '<header class="r-head"><h2>' + esc(c.subject || "(no subject)") + "</h2>" +
      '<p class="muted small">' + esc(c.id) + " · " + esc(c.language_name) + " · " + ago(c.received) + " · " + c.total_ms + " ms</p>" +
      '<p><span class="decision d-' + esc(c.status) + '">' + esc(label(c.status)) + "</span> to " + esc(label(c.destination)) +
      (c.incident ? ' <span class="tag inc">' + esc(c.incident) + "</span>" : "") + "</p></header>" +
      '<section class="r-body">' + esc(c.body || "(empty body)") + "</section>" +
      (c.reason_text.length ? "<h3>Why it landed here</h3><ul class='reasons'>" + c.reason_text.map((r) => "<li><code>" + esc(r.code) + "</code> " + esc(r.message) + "</li>").join("") + "</ul>" : "") +
      (tri ? "<h3>Triage</h3><table class='small'><tr><th></th><th>predicted</th><th>known label</th></tr>" + row("type") + row("queue") + row("priority") + "</table><p class='muted small'>Source: " + esc(label(c.triage_source)) + ". Confidences are uncalibrated.</p>"
           : "<p class='muted small'>No triage (" + esc(label(c.triage_source)) + ").</p>") +
      (c.draft ? "<h3>Draft reply</h3><section class='r-body draft'>" + esc(c.draft) + "</section>" : "") +
      (c.precedents.length ? "<h3>Precedents</h3><ul class='reasons'>" + c.precedents.map((p) => "<li><code>" + esc(p.ticket_id) + "</code> " + esc(p.subject) + " <span class='muted'>" + p.score + "</span></li>").join("") + "</ul>" : "");
  }

  $("#folders").addEventListener("click", (e) => {
    const b = e.target.closest(".folder"); if (!b) return;
    inbox.folder = b.dataset.f; inbox.selected = null; renderInbox();
  });
  $("#mail-list").addEventListener("click", (e) => {
    const b = e.target.closest(".mail"); if (!b) return;
    inbox.selected = b.dataset.id; renderInbox();
  });
  document.querySelectorAll(".vt").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll(".lane h2").forEach((h) => {
    h.classList.add("clickable"); h.title = "Open this lane in the inbox";
    h.addEventListener("click", () => openInbox(h.parentElement.dataset.lane));
  });
  Object.entries({ "k-total": "all", "k-escalate": "escalate", "k-review": "review", "k-draft": "draft_ready" }).forEach(([id, f]) => {
    const k = $("#" + id).parentElement; k.classList.add("clickable");
    k.addEventListener("click", () => openInbox(f));
  });
  $("#incident-banner").classList.add("clickable");
  $("#incident-banner").addEventListener("click", () => {
    const last = state.cards.filter((c) => c.incident).pop();
    if (last) openInbox(last.incident);
  });

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
    inbox.selected = null;
    if (inbox.open) renderInbox();
  });

  waitReady();
})();
