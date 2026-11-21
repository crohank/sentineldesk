"""D4: PII masking, the escalation gate, grounded drafting and draft validation.

Pipeline stages (signatures from PLAN.md, wired by D5 into `Components`):
    mask=mask_ticket, gate=gate, draft=draft, validate=validate
`draft_with_mode` also returns which generator produced the draft.
"""
from .config import DEFAULT_CONFIG, SafetyConfig
from .drafting import draft, draft_template, draft_with_mode
from .gate import gate
from .pii import MaskResult, mask_pii, mask_ticket
from .validation import validate

__all__ = [
    "DEFAULT_CONFIG",
    "MaskResult",
    "SafetyConfig",
    "draft",
    "draft_template",
    "draft_with_mode",
    "gate",
    "mask_pii",
    "mask_ticket",
    "validate",
]
