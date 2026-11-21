"""Triage: shared TF-IDF (word 1-2 + char_wb 3-5) -> one logistic-regression head per target,
each calibrated with a single temperature.

Output shape follows PLAN.md's interface:
    Prediction = {label, confidence, probabilities}
    Triage     = {type: Prediction, queue: Prediction, priority: Prediction}
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

TARGETS = ("type", "queue", "priority")

# Leading salutations / pleasantries that dominate the opening of most bodies (see docs/EDA.md).
_GREETING = re.compile(
    r"""^\s*(?:
        (?:dear|hello|hi|greetings|respected|to|good\s+(?:morning|afternoon|day))?\s*
            (?:customer\s+)?(?:support|service|care)(?:[\s-]+(?:team|department))?\s*[,:!.]
      | (?:sehr\s+geehrte[rs]?|geehrte[rs]?|liebe[rs]?|hallo|guten\s+tag)\s+[^,\n:]{0,40}[,:!]
      | (?:dear|hello|hi|greetings)\s*[,:!]
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)
_PLEASANTRY = re.compile(
    r"^\s*(?:i\s+hope\s+(?:this|you)[^.!\n]{0,60}[.!]|ich\s+hoffe[^.!\n]{0,60}[.!])\s*",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


def normalize(subject: str | None, body: str | None) -> str:
    """Subject (may be missing) + body with leading boilerplate removed and whitespace collapsed."""
    subject = (subject or "").strip()
    body = (body or "").strip()
    body = _GREETING.sub("", body, count=1)
    body = _PLEASANTRY.sub("", body, count=1)
    text = f"{subject} . {body}" if subject else body
    return _WS.sub(" ", text).strip()


def build_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Single temperature minimizing multiclass NLL."""
    def nll(t: float) -> float:
        return float(-log_softmax(logits / t, axis=1)[np.arange(len(y_idx)), y_idx].mean())
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


@dataclass
class TriagePredictor:
    features: FeatureUnion
    heads: dict[str, LogisticRegression]
    temperatures: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TARGETS})
    thresholds: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ---- training ----
    @classmethod
    def fit(cls, df: pd.DataFrame, C: float = 4.0) -> "TriagePredictor":
        text = [normalize(s, b) for s, b in zip(df["subject"], df["body"])]
        feats = build_features()
        X = feats.fit_transform(text)
        heads = {}
        for t in TARGETS:
            heads[t] = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(X, df[t])
        return cls(features=feats, heads=heads)

    def calibrate(self, df: pd.DataFrame) -> dict[str, float]:
        logits = self.logits(df)
        for t in TARGETS:
            classes = list(self.heads[t].classes_)
            y_idx = np.array([classes.index(v) for v in df[t]])
            self.temperatures[t] = fit_temperature(logits[t], y_idx)
        return dict(self.temperatures)

    # ---- inference ----
    def _transform(self, subjects, bodies):
        return self.features.transform([normalize(s, b) for s, b in zip(subjects, bodies)])

    def logits(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._transform(df["subject"], df["body"])
        return {t: self.heads[t].decision_function(X) for t in TARGETS}

    def proba(self, df: pd.DataFrame, calibrated: bool = True) -> dict[str, pd.DataFrame]:
        """Per-target DataFrame of class probabilities (columns = class names), indexed like df."""
        out = {}
        for t, z in self.logits(df).items():
            temp = self.temperatures[t] if calibrated else 1.0
            out[t] = pd.DataFrame(softmax(z / temp, axis=1), columns=self.heads[t].classes_, index=df.index)
        return out

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """ticket rows -> label / confidence per target plus p_<target>_<class> columns."""
        cols = {}
        for t, p in self.proba(df).items():
            cols[f"{t}_pred"] = p.idxmax(axis=1)
            cols[f"{t}_conf"] = p.max(axis=1)
            for c in p.columns:
                cols[f"p_{t}_{c}"] = p[c]
        res = pd.DataFrame(cols, index=df.index)
        if "ticket_id" in df:
            res.insert(0, "ticket_id", df["ticket_id"])
        return res

    def predict(self, subject: str | None, body: str) -> dict:
        df = pd.DataFrame({"subject": [subject or ""], "body": [body or ""]})
        result = {}
        for t, p in self.proba(df).items():
            row = p.iloc[0]
            result[t] = {
                "label": str(row.idxmax()),
                "confidence": float(row.max()),
                "probabilities": {str(k): float(v) for k, v in row.items()},
            }
        return result

    # ---- persistence ----
    def save(self, path: str | Path) -> dict:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_file = path / "model.joblib"
        joblib.dump({"features": self.features, "heads": self.heads}, model_file, compress=3)
        manifest = {
            **self.meta,
            "classes": {t: [str(c) for c in self.heads[t].classes_] for t in TARGETS},
            "temperatures": self.temperatures,
            "thresholds": self.thresholds,
            "sklearn_version": sklearn.__version__,
            "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "TriagePredictor":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        model_file = path / "model.joblib"
        sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
        if sha != manifest["model_sha256"]:
            raise ValueError(f"model.joblib hash {sha[:12]} does not match manifest")
        blob = joblib.load(model_file)
        meta = {k: v for k, v in manifest.items()
                if k not in {"classes", "temperatures", "thresholds", "sklearn_version", "model_sha256"}}
        return cls(features=blob["features"], heads=blob["heads"], temperatures=manifest["temperatures"],
                   thresholds=manifest["thresholds"], meta=meta)
