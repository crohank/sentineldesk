"""Minimal Gemini client over plain HTTP: JSON-schema calls, retries, disk cache, token totals.

Plain urllib instead of the Google SDK: no new compiled dependency to clear Smart
App Control on the team laptop (see D1-07).

Backends:
  gemini  Gemini API (AI Studio key in GEMINI_API_KEY, sent as a header, never in the URL)
  vertex  Vertex AI (GOOGLE_CLOUD_PROJECT, token from GOOGLE_ACCESS_TOKEN or gcloud)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://generativelanguage.googleapis.com/v1beta"
RETRY_CODES = (429, 500, 502, 503, 504)

# USD per 1M tokens, standard tier, from docs/D1_gemini_classifier_plan.md (read 2026-11-21).
# Output price includes thinking tokens. Used only for the running budget check.
PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-4-carbon": (2.00, 10.00),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}


class GeminiError(RuntimeError):
    pass


class BudgetExceeded(GeminiError):
    pass


def post_json(url: str, body: dict, headers: dict, timeout: float = 60, attempts: int = 6) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == attempts - 1:
                raise GeminiError(f"HTTP {e.code}: {e.read()[:400]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts - 1:
                raise GeminiError(f"request failed: {e}") from e
        time.sleep(min(2 ** attempt, 30))
    raise GeminiError("unreachable")


def get_json(url: str, headers: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise GeminiError(f"HTTP {e.code}: {e.read()[:400]!r}") from e


def vertex_token() -> str:
    if os.environ.get("GOOGLE_ACCESS_TOKEN"):
        return os.environ["GOOGLE_ACCESS_TOKEN"]
    try:
        return subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                              text=True, check=True, shell=sys.platform == "win32").stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise GeminiError("set GOOGLE_ACCESS_TOKEN or install and log in to gcloud") from e


class GeminiClient:
    def __init__(self, backend: str, model: str, *, location: str = "global",
                 cache_dir: Path | None = None, thinking_level: str | None = None,
                 budget_usd: float | None = None, timeout: float = 60):
        if not model:
            raise GeminiError(f"a model ID is required for backend {backend}")
        self.backend, self.model, self.timeout = backend, model, timeout
        self.thinking_level, self.budget_usd = thinking_level, budget_usd
        self.cache_dir = cache_dir
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
        self._lock = threading.Lock()
        if backend == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise GeminiError("GEMINI_API_KEY is not set")
            self.url = f"{API}/models/{model}:generateContent"
            self._headers = {"x-goog-api-key": key}
        elif backend == "vertex":
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise GeminiError("GOOGLE_CLOUD_PROJECT is not set")
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            self.url = (f"https://{host}/v1/projects/{project}/locations/{location}"
                        f"/publishers/google/models/{model}:generateContent")
            self._headers = {"Authorization": f"Bearer {vertex_token()}"}
        else:
            raise GeminiError(f"unknown backend {backend!r}")

    def cost_usd(self) -> float:
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        u = self.usage
        return (u["prompt_tokens"] * p_in + (u["output_tokens"] + u["thought_tokens"]) * p_out) / 1e6

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def generate_json(self, prompt: str, schema: dict, *, extra_config: dict | None = None) -> dict:
        config = {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema}
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        config.update(extra_config or {})
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}

        key = hashlib.sha256(f"{self.backend}|{self.model}|{json.dumps(body, sort_keys=True)}".encode()).hexdigest()
        path = self._cache_path(key)
        if path and path.exists():
            with self._lock:
                self.usage["cached"] += 1
            return json.loads(path.read_text(encoding="utf-8"))

        with self._lock:
            if self.budget_usd is not None and self.cost_usd() >= self.budget_usd:
                raise BudgetExceeded(f"spent about ${self.cost_usd():.2f}, budget ${self.budget_usd:.2f}")
        t0 = time.perf_counter()
        resp = post_json(self.url, body, self._headers, timeout=self.timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("usageMetadata", {})
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += meta.get("promptTokenCount", 0)
            self.usage["output_tokens"] += meta.get("candidatesTokenCount", 0)
            self.usage["thought_tokens"] += meta.get("thoughtsTokenCount", 0)
        try:
            cand = resp["candidates"][0]
            parsed = json.loads(cand["content"]["parts"][-1]["text"])
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise GeminiError(f"unparseable response: {json.dumps(resp)[:400]}") from e
        out = {"json": parsed, "latency_ms": latency_ms, "usage": meta,
               "avg_logprobs": cand.get("avgLogprobs"), "finish_reason": cand.get("finishReason")}
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out), encoding="utf-8")
        return out


def list_models() -> list[dict]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set")
    out, token = [], None
    while True:
        url = f"{API}/models?pageSize=1000" + (f"&pageToken={token}" if token else "")
        page = get_json(url, {"x-goog-api-key": key})
        out += page.get("models", [])
        token = page.get("nextPageToken")
        if not token:
            return out
