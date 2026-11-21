"""DEV-ONLY stand-ins for D2's triage and D3's retrieval.

D4's scripts use these until the real components are available. They are not part of
the package and never ship.

- TriageStandIn: TF-IDF (word 1-2 + char 3-5) and one logistic regression per head,
  fit on the KB `fit` partition. Its probabilities are NOT calibrated.
- RetrievalStandIn: TF-IDF cosine over subject + body of a KB subset. `answer_snippet`
  holds the full masked answer, as D4 asks D3 to do.
"""
from __future__ import annotations

import pickle
import time

from _common import DEV, guess_language, load_kb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from sentineldesk.safety.contracts import Precedent, Prediction, Triage
from sentineldesk.safety.pii import mask_pii

HEADS = ("type", "queue", "priority")


def _texts(subjects, bodies) -> list[str]:
    return [f"{s or ''}\n{b or ''}" for s, b in zip(subjects, bodies)]


class TriageStandIn:
    def __init__(self, features, models):
        self.features, self.models = features, models

    @classmethod
    def fit(cls, df) -> "TriageStandIn":
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=200_000)),
        ])
        X = features.fit_transform(_texts(df["subject"], df["body"]))
        models = {h: LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced").fit(X, df[h]) for h in HEADS}
        return cls(features, models)

    def predict(self, subjects, bodies) -> list[Triage]:
        X = self.features.transform(_texts(subjects, bodies))
        probs = {h: self.models[h].predict_proba(X) for h in HEADS}
        out = []
        for i in range(X.shape[0]):
            heads = {}
            for h in HEADS:
                classes, row = self.models[h].classes_, probs[h][i]
                j = int(row.argmax())
                heads[h] = Prediction(label=str(classes[j]), confidence=float(row[j]),
                                      probabilities={str(k): float(v) for k, v in zip(classes, row)})
            out.append(Triage(**heads))
        return out


class RetrievalStandIn:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.X = self.vec.fit_transform(_texts(self.df["subject"], self.df["body"]))

    def search(self, subject: str, body: str, k: int = 5) -> list[Precedent]:
        q = self.vec.transform(_texts([subject], [body]))
        scores = (self.X @ q.T).toarray().ravel()
        top = scores.argsort()[::-1][:k]
        out = []
        for i in top:
            row = self.df.iloc[int(i)]
            answer = mask_pii(row["answer"]).text
            out.append(Precedent(ticket_id=row["ticket_id"], score=round(float(scores[i]), 4),
                                 subject=row["subject"], answer_snippet=answer,
                                 answer_language=guess_language(answer)))
        return out


def load_triage() -> TriageStandIn:
    path = DEV / "triage_standin.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    kb = load_kb()
    t0 = time.time()
    model = TriageStandIn.fit(kb[kb["partition"] == "fit"])
    DEV.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    print(f"fit triage stand-in on the fit partition in {time.time() - t0:.0f}s -> {path}")
    return model


def load_retrieval(partitions=("fit", "calib", "val")) -> RetrievalStandIn:
    kb = load_kb()
    return RetrievalStandIn(kb[kb["partition"].isin(partitions)])


if __name__ == "__main__":
    load_triage()
