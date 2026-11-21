"""D4: PII masking, the escalation gate, grounded drafting and draft validation.

Pipeline stages (signatures from PLAN.md, wired by D5 into `Components`):
    mask=mask_ticket, gate=gate, draft=draft, validate=validate
"""
from .config import DEFAULT_CONFIG, SafetyConfig
from .gate import gate
from .pii import MaskResult, mask_pii, mask_ticket

__all__ = ["DEFAULT_CONFIG", "MaskResult", "SafetyConfig", "gate", "mask_pii", "mask_ticket"]
