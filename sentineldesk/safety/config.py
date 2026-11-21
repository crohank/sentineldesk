"""Gate and drafting settings.

Values come from reports/d4/thresholds.md (KB validation partition, D4-18). They were
measured with dev-only stand-ins and are rerun with D2's calibrated probabilities and D3's
retrieval before release. The outage pair is D2's to select.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SafetyConfig:
    # Rule 7: minimum calibrated confidence of the selected label, per head.
    head_thresholds: dict[str, float] = field(
        default_factory=lambda: {"type": 0.45, "queue": 0.45, "priority": 0.75}
    )
    # Rule 6: P(Incident) and P(high) both at or above these values.
    outage_pair: tuple[float, float] = (0.35, 0.35)
    # Rule 8: minimum similarity of the best precedent, kept separate per retrieval backend.
    evidence_min: dict[str, float] = field(
        default_factory=lambda: {"sparse": 0.28, "dense": 0.80, "fixture": 0.0}
    )
    retrieval_backend: str = "sparse"
    # Drafting.
    max_actions: int = 3
    max_reply_words: int = 120
    model_budget_s: float = 6.0
    # Gemma via the shared Gemini API answered in 13-31 s at 12:25-12:35, so its live budget is longer.
    gemma_budget_s: float = 40.0

    def evidence_threshold(self) -> float:
        return self.evidence_min.get(self.retrieval_backend, self.evidence_min["sparse"])


DEFAULT_CONFIG = SafetyConfig()
