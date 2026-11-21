"""Classify tickets using the fit cases returned by Agent Retrieval."""
import json
import math
from html import escape

import pandas as pd

from .classifier_prompt import HEADS, build_prompt, confidence, load_labels, schema
from .safety.pii import mask_pii
from .schemas import Prediction, Triage


class GeminiTriage:
    def __init__(self, store, data_dir, model):
        self.store = store
        self.url = f"{store.embedding.split('/models/')[0]}/models/{model}:generateContent"
        kb = pd.read_parquet(data_dir / "kb_train.parquet",
                             columns=["ticket_id", "subject", "body", *HEADS])
        manifest = pd.read_parquet(data_dir / "kb_partitions.parquet")
        self.fit = kb.merge(manifest, on="ticket_id", validate="one_to_one").query(
            "partition == 'fit'").set_index("ticket_id")
        self.labels = load_labels(self.fit)
        self.config = {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema(self.labels),
            "thinkingConfig": {"thinkingLevel": "low"},
        }

    def predict(self, ticket, precedents):
        if not precedents:
            raise ValueError("No training precedents returned")
        examples = self.fit.loc[[p.ticket_id for p in precedents], HEADS + ["subject", "body"]]
        examples["text"] = [
            escape(mask_pii(f"{s or ''}\n{b}").text.strip(), quote=False)
            for s, b in zip(examples.subject, examples.body)
        ]
        prompt = build_prompt(escape(ticket.text, quote=False), examples, self.labels)
        response = self.store.post(self.url, {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": self.config,
        }, attempts=2)
        candidate = response["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
        answer = json.loads(text)
        predictions = {}
        for head in HEADS:
            label = answer[head]
            probabilities = answer[f"{head}_probs"]
            if (set(probabilities) != set(self.labels[head]) or label not in probabilities
                    or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                    or not any(probabilities.values())):
                raise ValueError(f"Invalid Gemini probabilities for {head}")
            score, probabilities = confidence(answer, head)
            predictions[head] = Prediction(label=label, confidence=score, probabilities=probabilities,
                                           calibrated=False, source="Gemini")
        return Triage(**predictions)
