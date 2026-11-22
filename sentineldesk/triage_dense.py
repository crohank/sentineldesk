"""Dense triage variants, mirroring sentineldesk/triage.py (same text preprocessing, same heads,
same temperature calibration, same output schema).

features="embed"   : EmbeddingGemma vector (L2-normalized) -> LR heads           (variant B)
features="stacked" : [TF-IDF word+char | alpha * EmbeddingGemma] -> LR heads     (variant C)

Output of predict(): {type|queue|priority: {label, confidence, probabilities}} — identical to
TriagePredictor.predict, so D4/D5 can swap predictors without code changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (pickled inside FeatureUnion)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from sentineldesk.embed import Embedder
from sentineldesk.triage import TARGETS, build_features, fit_temperature, normalize

FEATURE_MODES = ("embed", "stacked", "tfidf")
HEAD_TYPES = ("lr", "svm", "lgbm")


def make_head(C: float) -> LogisticRegression:
    # same head as triage.py apart from C (multinomial is lbfgs' default for multiclass)
    return LogisticRegression(max_iter=2000, C=C, class_weight="balanced")


def stack(tfidf_X, dense: np.ndarray, alpha: float):
    return sp.hstack([tfidf_X, sp.csr_matrix(alpha * dense)], format="csr")


def head_logits(head, X) -> np.ndarray:
    """Per-class scores fed to the temperature softmax: LR/SVM margins, LightGBM raw scores."""
    if hasattr(head, "booster_"):  # LightGBM
        return np.asarray(head.predict(X, raw_score=True))
    return head.decision_function(X)


@dataclass
class DenseTriagePredictor:
    features: str
    embedder_cfg: dict
    heads: dict[str, LogisticRegression]
    tfidf: FeatureUnion | None = None
    alpha: float = 1.0
    C: dict[str, float] = field(default_factory=dict)
    temperatures: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TARGETS})
    thresholds: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    head_type: str = "lr"
    scaler: object | None = None  # StandardScaler fitted on fit-partition embeddings (linear heads only)
    _embedder: Embedder | None = field(default=None, repr=False)

    # ---- features ----
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            c = self.embedder_cfg
            self._embedder = Embedder(model_id=c["model_id"], revision=c["revision"], prompt_name=c["prompt_name"],
                                      max_len=c["max_len"], dim=c["dim"])
        return self._embedder

    def attach_embedder(self, emb: Embedder) -> None:
        cfg = emb.config()
        for k in ("model_id", "revision", "prompt_name", "max_len", "dim"):
            if cfg[k] != self.embedder_cfg[k]:
                raise ValueError(f"embedder {k}={cfg[k]!r} differs from trained {self.embedder_cfg[k]!r}")
        self._embedder = emb

    def matrix(self, df: pd.DataFrame, dense: np.ndarray | None = None):
        """Feature matrix for df. `dense` (aligned with df rows) skips re-encoding, e.g. from the cache."""
        texts = [normalize(s, b) for s, b in zip(df["subject"], df["body"])]
        if self.features == "tfidf":  # TF-IDF-only heads (LinearSVC / LightGBM); no embedder involved
            return self.tfidf.transform(texts)
        if dense is None:
            dense = self.embedder.encode(texts)  # L2-normalized
        if self.scaler is not None:
            dense = self.scaler.transform(dense)  # scale before alpha
        if self.features == "embed":
            return dense
        return stack(self.tfidf.transform(texts), dense, self.alpha)

    # ---- inference ----
    def logits(self, df: pd.DataFrame, dense: np.ndarray | None = None) -> dict[str, np.ndarray]:
        X = self.matrix(df, dense)
        return {t: head_logits(self.heads[t], X) for t in TARGETS}

    def calibrate(self, df: pd.DataFrame, dense: np.ndarray | None = None) -> dict[str, float]:
        """`dense` must be the raw L2-normalized vectors; the scaler (if any) is applied inside matrix()."""
        z = self.logits(df, dense)
        for t in TARGETS:
            classes = list(self.heads[t].classes_)
            self.temperatures[t] = fit_temperature(z[t], np.array([classes.index(v) for v in df[t]]))
        return dict(self.temperatures)

    def proba(self, df: pd.DataFrame, dense: np.ndarray | None = None, calibrated: bool = True) -> dict[str, pd.DataFrame]:
        out = {}
        for t, z in self.logits(df, dense).items():
            temp = self.temperatures[t] if calibrated else 1.0
            out[t] = pd.DataFrame(softmax(z / temp, axis=1), columns=self.heads[t].classes_, index=df.index)
        return out

    def predict_batch(self, df: pd.DataFrame, dense: np.ndarray | None = None) -> pd.DataFrame:
        cols = {}
        for t, p in self.proba(df, dense).items():
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
            result[t] = {"label": str(row.idxmax()), "confidence": float(row.max()),
                         "probabilities": {str(k): float(v) for k, v in row.items()}}
        return result

    # ---- persistence ----
    def save(self, path: str | Path) -> dict:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_file = path / "model.joblib"
        joblib.dump({"heads": self.heads, "tfidf": self.tfidf, "scaler": self.scaler}, model_file, compress=3)
        manifest = {
            **self.meta,
            "features": self.features,
            "head_type": self.head_type,
            "scaled": self.scaler is not None,
            "embedder": self.embedder_cfg,
            "alpha": self.alpha,
            "C": self.C,
            "classes": {t: [str(c) for c in self.heads[t].classes_] for t in TARGETS},
            "temperatures": self.temperatures,
            "thresholds": self.thresholds,
            "sklearn_version": sklearn.__version__,
            "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "DenseTriagePredictor":
        path = Path(path)
        m = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        model_file = path / "model.joblib"
        sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
        if sha != m["model_sha256"]:
            raise ValueError(f"model.joblib hash {sha[:12]} does not match manifest")
        blob = joblib.load(model_file)
        known = {"features", "embedder", "alpha", "C", "classes", "temperatures", "thresholds", "sklearn_version",
                 "model_sha256", "head_type", "scaled"}
        obj = cls(features=m["features"], embedder_cfg=m["embedder"], heads=blob["heads"], tfidf=blob["tfidf"],
                  alpha=m["alpha"], C=m["C"], temperatures=m["temperatures"], thresholds=m["thresholds"],
                  meta={k: v for k, v in m.items() if k not in known},
                  head_type=m.get("head_type", "lr"), scaler=blob.get("scaler"))
        if m.get("scaled") and obj.scaler is None:
            raise ValueError("manifest says scaled but model.joblib has no scaler")
        if embedder is not None:
            obj.attach_embedder(embedder)
        return obj


def embed_frame(emb: Embedder, df: pd.DataFrame, show_progress: bool = True) -> tuple[np.ndarray, dict]:
    """Cached classification-prompt vectors for df rows (keyed by ticket_id + normalized text)."""
    texts = [normalize(s, b) for s, b in zip(df["subject"], df["body"])]
    return emb.cached_encode(df["ticket_id"].tolist(), texts, show_progress=show_progress)


def fit_tfidf(train: pd.DataFrame) -> tuple[FeatureUnion, sp.csr_matrix]:
    feats = build_features()
    X = feats.fit_transform([normalize(s, b) for s, b in zip(train["subject"], train["body"])])
    return feats, X
