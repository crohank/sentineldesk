# D5 decision log (integration and release)

One entry per decision, newest at the bottom. Each entry records what was decided and why, so a judge or a teammate can be answered from here.

## 2026-11-21

### D5-01 One package with uv, and pip still works
**Decision:** Add `pyproject.toml` and `uv.lock` for a `sentineldesk` package with `serve` and `export_schemas` entry points. Generate `requirements.txt` from the lockfile with `uv export`.
**Why:** The plan asks for `uv run` commands. D1 installs with pip into a venv on Windows, and the exported requirements file keeps that path working with the same pinned versions. The export also pins pandas to 3.1.1, which closes one of D1's open items.

### D5-02 `requires-python = ">=3.12,<3.15"`, no `.python-version`
**Decision:** Allow Python 3.12 through 3.14 and commit no `.python-version` file. Local development uses a 3.12 venv.
**Why:** D1 must use system Python 3.14, because Windows Smart App Control blocks uv-managed interpreters on that laptop (D1-07). A committed `.python-version` would make uv try to fetch a managed interpreter there. Python 3.15 is excluded because `uv sync --python 3.15` failed: scikit-learn 1.8.0 has no 3.15 wheel and does not build from source. Without the cap, uv could pick Homebrew 3.15 on a Mac.

### D5-03 Shared contract in `sentineldesk/schemas.py`
**Decision:** Use Pydantic v2 models whose field names match PLAN.md exactly. Every model forbids extra fields. JSON Schema is exported to `docs/schemas/contract.schema.json`, and a test fails when the export is stale.
**Why:** D2, D3 and D4 code against the plan text, so the names must match it. Forbidding extra fields stops a client from smuggling labels into `TicketInput`. The schema snapshot makes every contract change visible in review.

### D5-04 Labels are strings; status, destination and reason codes are Literals
**Decision:** `type`, `queue` and `priority` labels are free strings. `DecisionStatus`, `Destination` and `ReasonCode` are closed sets that map to escalation rules 1–9.
**Why:** The plan says queue names are learned from the KB. Reason codes need to be closed so that the console, the logs and the scorer agree. The codes are **provisional until D4 confirms them**. Rule 2 has two codes (`HIGH_IMPACT_OUTAGE` and `PREDICTED_INCIDENT_HIGH`) so the console can say which mechanism caught an outage.

### D5-05 Empty body returns 200 with `needs_clarification`
**Decision:** `TicketInput.body` has no minimum length. `analyze()` checks for at least one Unicode letter. If there is none, it returns rule 1 and skips triage, retrieval and drafting.
**Why:** Rule 1 is a gate outcome, not a malformed request. The judge should see "Add a short description of the issue", not a 422 error.

### D5-06 Pipeline owns rules 1 and 9; the gate owns 2–8
**Decision:** `analyze()` handles insufficient input before triage, and it downgrades any draft that abstains or fails validation to `review`/`DRAFT_FAILED`. The gate callable returns `draft_ready` when no rule 2–8 fires.
**Why:** Rule 1 must run before the classifier. Rule 9 depends on the draft, which only exists after the gate. Keeping both in the orchestrator means the API and batch evaluation behave identically.

### D5-07 Citation contract check on every draft
**Decision:** The pipeline always checks three things: cited IDs are among the retrieved precedents, each claim's ID is cited, and each evidence span is a substring of its precedent's `answer_snippet`. A draft that fails is withheld. This runs in addition to D4's `validate()`.
**Why:** The console links citations to evidence cards, and a fabricated ID would produce a broken link or a false citation. The check costs microseconds.

### D5-08 Fixture mode is explicit
**Decision:** Until real components land, stubs return `fixtures/routine.json`, `retrieval_mode` and `generator_mode` are `"fixture"`, every fixture string starts with `[FIXTURE]`, and the console shows a banner. `/healthz` reports `degraded` while artifacts are missing. Fixture ticket IDs are in KB range (`TKT-00001`..`TKT-00015`).
**Why:** The plan forbids anything that looks like a hardcoded demo answer. KB-range IDs keep the fixtures from tripping the leakage checks.

### D5-09 Server-rendered console with no external assets
**Decision:** Use Jinja2 and a plain form POST in the same FastAPI process. Local CSS uses system fonts and automatic dark mode, and the page loads no JavaScript. Asset URLs are relative.
**Why:** The console must work offline on the demo laptop. A relative stylesheet URL also avoids mixed-content errors behind the Cloud Run HTTPS proxy, which `url_for` would cause by emitting `http://` URLs.

### D5-10 Language detection stub uses seeded langdetect, warmed at startup
**Decision:** Set `DetectorFactory.seed = 0` and make one warm-up call when components load.
**Why:** langdetect is non-deterministic without a seed. Its first call loads profiles and took 174 ms. D3 replaces this stub with the release detector.

### D5-11 Bind host from `$HOST`, port from `$PORT`
**Decision:** `uv run serve` defaults to `127.0.0.1:8000`. The container will set `HOST=0.0.0.0`, and Cloud Run sets `PORT`.
**Why:** Binding to loopback is safe on a shared venue network. Cloud Run needs `0.0.0.0` and `$PORT`.

### D5-12 `analyze(ticket, components=None, *, mode="live", k=5)`
**Decision:** Keep the plan's `analyze(ticket, *, mode)` call shape. Add an optional `components` argument; when it is omitted, the process-wide components are loaded once. `mode="batch"` is for frozen holdout prediction and disables live-only state such as burst detection.
**Why:** D1's prediction writer can call `analyze(ticket, mode="batch")` exactly as the plan says. Tests can pass stub components without patching globals.

### D5-13 Draft-stage exceptions become rule 9, not HTTP 500
**Decision:** If `draft()` or `validate()` raises, `analyze()` returns `review`/`DRAFT_FAILED` and keeps triage and precedents. The log records the exception type only.
**Why:** Rule 9 covers timeouts and failed generation. The operator still needs the evidence. Exception messages can quote ticket text, so they stay out of the logs.

### D5-14 Inline citation markers are `[TKT-12345]`
**Decision:** `reply_text` marks citations as `[TKT-12345]`. This is documented in the `Draft` docstring, which is exported into the JSON Schema.
**Why:** The console renders those markers as links to evidence cards. D4's generator needs one agreed format.

### D5-15 Log file location
**Decision:** This log is `decisions.md` at the repository root, next to `CLAUDE.md`. D1's log stays at `docs/D1_decisions.md`.
**Why:** The D5 owner asked for `decisions.md`.

### D5-16 Console styled after the event's "Build with AI" look (superseded by D5-20)
**Decision:** The console takes its style from the event page: a light gray canvas, flat panels with thick black outlines and offset shadows, pill badges, and the four Google colors. Decision states use those colors: blue for ready to draft, red for escalate to on-call, yellow for needs clarification, and neutral for review. Citations are green pills. The `{SentinelDesk}` wordmark uses the banner's cream braces. The page uses Google's palette values but no GDG logo or event artwork. Dark mode is automatic.
**Why:** The judges see the event theme all day, and a matching console reads as built for this event. Color-coded states make the routing outcome visible at a glance in the demo. Google Sans is listed first in the font stack, but the page bundles no font, so it falls back to system fonts offline.

### D5-17 Console demo aids
**Decision:**
- Five sample tickets from the PLAN.md demo kit fill the form through `/?sample=<key>`. They never show a result on their own.
- A pipeline stepper shows each stage's time, marks skipped stages, and ends in the routing outcome.
- A "How it works" panel fills the empty state.
- Cmd/Ctrl+Enter submits, and the Analyze button shows a loading state.
- A Copy button copies the draft.
- Clicking a citation opens and highlights its evidence card.
- Precedents show their rank and a relative score bar.
- The footer credits the dataset (CC BY-NC 4.0).
- The JavaScript lives in `static/console.js` and only enhances the page; the console works without it.
**Why:** The presenter has five minutes. One click per demo ticket removes typing errors on stage. A sample only fills the form and the result still comes from the live pipeline, which keeps the plan's rule against hardcoded demo answers. The dataset license requires attribution.

### D5-18 Result panels grouped into column stacks
**Decision:** Decision and triage share one grid cell as a vertical stack, and so do draft and run. The input form spans no rows.
**Why:** With per-panel grid areas, the tall input panel stretched row heights and left a large gap between Decision and Triage.

### D5-19 Full GCP access: GCP by default, offline fallback kept
**Decision:** The team has full Google Cloud access. Deploy to Cloud Run from Artifact Registry, and keep secrets in Secret Manager. Vertex AI is the first choice for Gemma, Gemini and embeddings. The local bundle stays as the fallback the plan requires.
**Why:** It is a GDG Cloud event with a Google engineer judging, so visible GCP use counts. Venue Wi-Fi can still fail, which is why the offline path stays.

### D5-20 Console rebuilt on Material Design 3
**Decision:** The thick black outlines are gone. The console now uses Material Design 3, Google's own design system, hand-built from M3 baseline tokens:
- tonal surfaces (`surface-container-*`) and soft M3 elevation in place of outlines;
- filled text fields and filled or tonal buttons with state layers;
- M3 corner radii and connected button-group shapes for the pipeline stepper and lists;
- emphasized easing on transitions;
- separate light and dark schemes.
Google's four brand colors appear only as small accents: the logo mark, panel dots and sample language tags.
**Why:** The user found the outlines too heavy and asked for the best CSS available. M3 is the design system Google uses on its own products, which suits a GDG event. I used no Material Web or Tailwind package: both need a build step or a CDN, and the console must work offline. A web search found no ready CSS token package for M3 Expressive, so the tokens are hand-written CSS custom properties.

### D5-21 Latest versions by default
**Decision:** `uv lock --upgrade` confirmed every dependency is already at its latest release. The local venv now runs Python 3.14, the newest version the dependencies support. The one deliberate pin stays: `scikit-learn==1.8.0`.
**Why:** The user wants the latest versions everywhere. scikit-learn 1.9.x is blocked by Windows Smart App Control on D1's laptop (D1-07), and 1.8.0 has no Python 3.15 wheel. That pin is why Python is capped below 3.15.

## Open

- D4 to confirm reason codes, statuses and destinations (D5-04).
- Wire `--fast` to the fast-mode artifact directory once D1/D2/D3 publish a manifest.
- Dockerfile, Cloud Run deploy script, and an offline startup test.
- Burst detection banner (optional).
