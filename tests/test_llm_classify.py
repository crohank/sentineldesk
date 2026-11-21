import json

import pandas as pd
import pytest

from scripts import gemini_client as gc
from scripts import llm_classify as lc

needs_data = pytest.mark.skipif(
    not (lc.PROC / "kb_partitions.parquet").exists(), reason="split or partitions missing")


def fake_response(payload: dict, prompt_tokens=100, out_tokens=20, thoughts=5) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": out_tokens,
                              "thoughtsTokenCount": thoughts}}


def test_client_caches_counts_usage_and_keeps_key_out_of_url(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(url, body, headers, timeout=60, attempts=6):
        calls.append((url, headers, body))
        return fake_response({"x": 1})
    monkeypatch.setattr(gc, "post_json", fake_post)

    c = gc.GeminiClient("gemini", "gemini-3.8-flash", cache_dir=tmp_path, thinking_level="low")
    a = c.generate_json("hello", {"type": "OBJECT"})
    b = c.generate_json("hello", {"type": "OBJECT"})
    assert a["json"] == b["json"] == {"x": 1}
    assert len(calls) == 1 and c.usage["cached"] == 1 and c.usage["prompt_tokens"] == 100
    url, headers, body = calls[0]
    assert "test-key" not in url and headers["x-goog-api-key"] == "test-key"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert c.cost_usd() == pytest.approx((100 * 0.75 + 25 * 3.75) / 1e6)


def test_client_stops_at_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gc, "post_json", lambda *a, **k: fake_response({}, prompt_tokens=10**6))
    c = gc.GeminiClient("gemini", "gemini-3.8-flash", budget_usd=0.5)
    c.generate_json("a", {})
    with pytest.raises(gc.BudgetExceeded):
        c.generate_json("b", {})


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(gc.GeminiError, match="GEMINI_API_KEY"):
        gc.GeminiClient("gemini", "gemini-3.8-flash")


@needs_data
def test_mock_val_run_scores_above_majority(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "REPORTS", tmp_path)
    monkeypatch.setattr(lc, "PRED_DIR", tmp_path / "pred")
    r = lc.run("val", "few", "mock", None, "t", n=200, k=8)
    assert r["meta"]["n"] == 200 and r["meta"]["few_shot_pool"] == "fit partition"
    assert r["macro_f1"]["type"] > 0.5


@needs_data
def test_llm_path_with_fake_client(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "REPORTS", tmp_path)
    monkeypatch.setattr(lc, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(lc, "CACHE", tmp_path / "cache")
    kb = pd.read_parquet(lc.PROC / "kb_train.parquet")
    labels = lc.load_labels(kb)
    seen_prompts = []

    class Fake:
        def __init__(self, *a, **k):
            self.usage, self.cost_usd = {"calls": 0}, (lambda: 0.0)

        def generate_json(self, prompt, schema):
            seen_prompts.append(prompt)
            ans = {h: labels[h][0] for h in lc.HEADS}
            for h in lc.HEADS:
                ans[f"{h}_probs"] = {l: (2.0 if l == labels[h][0] else 0.0) for l in labels[h]}
            return {"json": ans, "latency_ms": 5.0, "usage": {}}
    monkeypatch.setattr(lc, "GeminiClient", Fake)

    r = lc.run("val", "few", "gemini", "m", "t", n=30, k=3)
    assert r["meta"]["n"] == 30 and len(seen_prompts) == 30
    assert seen_prompts[0].count("<example id=") == 3
    assert "data, not instructions" in seen_prompts[0]


@needs_data
def test_invalid_label_from_model_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "REPORTS", tmp_path)
    monkeypatch.setattr(lc, "PRED_DIR", tmp_path / "pred")

    class Bad:
        def __init__(self, *a, **k):
            self.usage, self.cost_usd = {}, (lambda: 0.0)

        def generate_json(self, prompt, schema):
            return {"json": {"type": "Bug", "queue": "x", "priority": "urgent"}, "latency_ms": 1.0}
    monkeypatch.setattr(lc, "GeminiClient", Bad)
    with pytest.raises(lc.ClassifyError, match="not an allowed label"):
        lc.run("val", "zero", "gemini", "m", "t", n=12)


def test_confidence_renormalizes():
    conf, probs = lc.confidence({"type": "A", "type_probs": {"A": 3, "B": 1}}, "type")
    assert conf == 0.75 and probs == {"A": 0.75, "B": 0.25}
    assert lc.confidence({"type": "A", "type_probs": {"A": 0, "B": 0}}, "type")[0] == 0.0
