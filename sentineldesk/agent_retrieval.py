"""Gemini Embedding 2 and Google Cloud Agent Retrieval, shared by indexing and serving."""
import math
import os
import subprocess
import threading
import time

import httpx
from langdetect import detect, LangDetectException

from .safety.pii import mask_pii
from .schemas import Precedent

PROJECT = "sentineldesk-ajin-ze3gmb"
COLLECTION = "tickets-fit-ge2-v1"


class AgentRetrieval:
    def __init__(self):
        self.parent = f"projects/{PROJECT}/locations/us-central1/collections/{COLLECTION}"
        self.objects = f"https://vectorsearch.googleapis.com/v1/{self.parent}/dataObjects"
        self.embedding = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/"
                          "publishers/google/models/gemini-embedding-2:embedContent")
        self.client = httpx.Client(timeout=60)
        self.lock = threading.Lock()
        self.token = ""
        self.refresh_at = 0

    def headers(self):
        with self.lock:
            if time.monotonic() >= self.refresh_at:
                if os.getenv("SENTINELDESK_AUTH") == "gcloud":
                    self.token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
                else:
                    import google.auth
                    from google.auth.transport.requests import Request
                    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                    creds.refresh(Request())
                    self.token = creds.token
                self.refresh_at = time.monotonic() + 2700
        return {"Authorization": f"Bearer {self.token}", "x-goog-user-project": PROJECT}

    def post(self, url, body, *, attempts=6):
        for attempt in range(attempts):
            try:
                response = self.client.post(url, json=body, headers=self.headers())
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
            else:
                if response.is_success:
                    return response.json()
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                    raise RuntimeError(f"{response.status_code}: {response.json()['error']['message'][:400]}")
            time.sleep(2 ** attempt)

    def embed(self, text, *, attempts=6):
        values = self.post(self.embedding, {"content": {"parts": [{"text": text}]},
                                            "outputDimensionality": 768}, attempts=attempts)["embedding"]["values"]
        if len(values) != 768 or not all(map(math.isfinite, values)) or not any(values):
            raise ValueError("Invalid Gemini embedding")
        return values

    def retrieve(self, ticket, k=5):
        text = mask_pii(ticket.text).text.strip()
        if not text:
            return []
        vector = self.embed(f"task: search result | query: {text}", attempts=2)
        query = {"searchField": "embedding", "vector": {"values": vector}, "topK": k,
                 "distanceMetric": "COSINE_DISTANCE", "searchHint": {"knnHint": {}},
                 "outputFields": {"dataFields": ["ticket_id", "subject", "answer"]}}
        hits = self.post(self.objects + ":search", {"vectorSearch": query}, attempts=2).get("results", [])
        results = []
        for hit in hits:
            row = hit["dataObject"]["data"]
            try:
                language = detect(row["answer"])
            except LangDetectException:
                language = "und"
            results.append(Precedent(ticket_id=row["ticket_id"], subject=row["subject"],
                                     answer_snippet=row["answer"], answer_language=language,
                                     score=1 - hit.get("distance", 0.0)))
        return results
