"""Vertex AI RAG Engine retrieval (D3-16), with the local retriever as fallback.

    rag = RagEngineRetriever.from_manifest("artifacts/rag_engine/full.json", local=local_retriever)
    # the KB is split over several corpora (one per import part); each query hits all of them
    # in parallel and the hits are merged by score
    precedents, mode = rag.retrieve_with_mode({"subject": None, "body": "..."}, k=5)
    # mode: "rag_engine", or the local retriever's mode if RAG Engine failed or timed out

The corpus holds one file per KB ticket, named `<TICKET_ID>.txt`, containing the same cleaned
subject + body the local indexes use (D3-03, D3-04). RAG Engine returns the source URI, which
maps back to the ticket ID; the answer then comes from the local evidence store, so the
precedent carries the exact historical answer (D3-14). Holdout IDs are rejected.

RAG Engine scores are cosine distance (lower is closer); Precedent.score is 1 - distance so
that higher is better in every mode. Thresholds are still per mode.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sentineldesk.retrieval import HOLDOUT_FIRST_ID, Precedent, Retriever
from sentineldesk.textprep import clean_for_search

_ID_IN_URI = re.compile(r"(TKT-\d{5})\.txt$")
API = "v1beta1"


class GcloudToken:
    """Access token from `gcloud auth print-access-token`, refreshed every 45 minutes."""

    def __init__(self) -> None:
        self._tok: tuple[str, float] | None = None

    def __call__(self) -> str:
        if self._tok is None or time.time() - self._tok[1] > 45 * 60:
            out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                 text=True, timeout=20, check=True).stdout.strip()
            self._tok = (out, time.time())
        return self._tok[0]


def rest(method: str, url: str, project: str, token: GcloudToken, body: dict | None = None,
         timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token()}", "x-goog-user-project": project,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:  # surface Google's error message, not just the status
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
    return json.loads(raw) if raw else {}


class RagEngineRetriever:
    def __init__(self, project: str, location: str, corpora: list[str], local: Retriever,
                 timeout: float = 3.0) -> None:
        self.project, self.location, self.corpora = project, location, list(corpora)
        self.local = local  # evidence store + fallback search
        self.timeout = timeout
        self.token = GcloudToken()
        self.mode = "rag_engine"
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.corpora)))

    @classmethod
    def from_manifest(cls, path: str | Path, local: Retriever, timeout: float = 3.0) -> "RagEngineRetriever":
        m = json.loads(Path(path).read_text())
        corpora = [m["corpora"][k] for k in sorted(m["corpora"], key=int)] if m.get("corpora") else [m["corpus"]]
        return cls(m["project"], m["location"], corpora, local, timeout)

    def _query(self, corpus: str, text: str, top_k: int) -> list[tuple[str, float]]:
        url = f"https://{self.location}-aiplatform.googleapis.com/{API}/projects/{self.project}/locations/{self.location}:retrieveContexts"
        out = rest("POST", url, self.project, self.token, {
            "vertex_rag_store": {"rag_resources": [{"rag_corpus": corpus}]},
            "query": {"text": text, "rag_retrieval_config": {"top_k": top_k}},
        }, timeout=self.timeout)
        hits = []
        for c in out.get("contexts", {}).get("contexts", []):
            m = _ID_IN_URI.search(c.get("sourceUri", "") or c.get("sourceDisplayName", ""))
            if m:
                hits.append((m.group(1), round(1.0 - float(c.get("score", c.get("distance", 1.0))), 4)))
        return hits

    def search(self, query_text: str, k: int = 5, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (ticket_id, 1 - cosine_distance) across all corpora. Raises on any API failure."""
        top_k = k + (len(exclude) if exclude else 0)
        futures = [self._pool.submit(self._query, c, query_text, top_k) for c in self.corpora]
        merged = [h for f in futures for h in f.result()]
        merged.sort(key=lambda h: -h[1])
        out, seen = [], set()
        for tid, score in merged:
            if tid in seen or (exclude and tid in exclude):
                continue
            assert tid < HOLDOUT_FIRST_ID, f"holdout ID {tid} returned by RAG Engine"
            seen.add(tid)
            out.append((tid, score))
        return out[:k]

    def retrieve_with_mode(self, ticket: Any, k: int = 5) -> tuple[list[Precedent], str]:
        body = ticket.get("body") if isinstance(ticket, dict) else getattr(ticket, "body", None)
        subject = ticket.get("subject") if isinstance(ticket, dict) else getattr(ticket, "subject", None)
        text = clean_for_search(subject, body)
        if not text or not (body or "").strip():
            return [], "none"
        try:
            hits = self.search(text, k)
        except Exception:  # network down, auth expired, quota: fall back to the local stack
            return self.local.retrieve_with_mode(ticket, k)
        ev = self.local.evidence
        out = [Precedent(t, s, str(ev.at[t, "subject"]), str(ev.at[t, "answer"]), str(ev.at[t, "answer_language"]))
               for t, s in hits if t in ev.index]
        return out, self.mode

    def retrieve(self, ticket: Any, k: int = 5) -> list[Precedent]:
        return self.retrieve_with_mode(ticket, k)[0]
