"""Deterministic verifiers.

Critics that never consult a model. Objective questions - does this file exist,
does this claim resolve to evidence, does this command record support this
assertion - are answered mechanically, because the components of MIMIR that are
already deterministic are the only ones that never flip between runs.
"""

from mimir.verify.claims import (
    AnswerSupport,
    ClaimKind,
    ClaimSupport,
    check_answer,
    check_claim,
)

__all__ = [
    "AnswerSupport",
    "ClaimKind",
    "ClaimSupport",
    "check_answer",
    "check_claim",
]
