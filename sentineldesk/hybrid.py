import json
from .schemas import Prediction, Triage
from .triage import TARGETS, TriagePredictor


class HybridTriage:
    def __init__(self, gemini, bundle):
        self.gemini = gemini
        self.tfidf = TriagePredictor.load(bundle)
        self.cutoffs = json.loads((bundle / "selection.json").read_text())

    def predict(self, ticket, precedents):
        local = self.tfidf.predict(ticket.subject, ticket.body)
        remote = self.gemini.predict(ticket, precedents)
        selected = {}
        for head in TARGETS:
            if local[head]["confidence"] >= self.cutoffs[head]:
                selected[head] = Prediction(**local[head], calibrated=True, source="TF-IDF")
            else:
                selected[head] = getattr(remote, head)
        return Triage(**selected)
