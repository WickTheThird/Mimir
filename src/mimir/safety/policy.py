"""Policy engine: the single gate between a proposal and execution (ADR 13)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.command import (
    PolicyViolation,
    ProposedCommand,
    RiskAssessment,
    RiskClass,
)
from mimir.safety.risk import RiskClassifier

log = get_logger(__name__)


class Verdict(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(slots=True)
class PolicyDecision:
    verdict: Verdict
    assessment: RiskAssessment
    reason: str = ""
    violations: list[PolicyViolation] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW

    @property
    def needs_approval(self) -> bool:
        return self.verdict == Verdict.REQUIRE_APPROVAL

    @property
    def denied(self) -> bool:
        return self.verdict == Verdict.DENY


class PolicyEngine:
    """Deterministic. Never calls a model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.safety
        self.classifier = RiskClassifier(self.config)

    # -- main entry point -------------------------------------------------

    def evaluate(self, command: ProposedCommand) -> PolicyDecision:
        assessment = self.classifier.classify(command)
        command.assessment = assessment
        violations = list(assessment.violations)

        violations.extend(self._check_context_allowlists(command))
        violations.extend(self._check_capability_bindings(command))

        fatal = [v for v in violations if v.fatal]
        if fatal or assessment.forbidden:
            reason = fatal[0].message if fatal else "risk class is on the forbidden list"
            decision = PolicyDecision(Verdict.DENY, assessment, reason, violations)
        elif assessment.requires_approval:
            decision = PolicyDecision(
                Verdict.REQUIRE_APPROVAL,
                assessment,
                self._approval_reason(assessment),
                violations,
            )
        else:
            decision = PolicyDecision(Verdict.ALLOW, assessment, "within auto-execute policy",
                                      violations)

        log.info(
            "policy_decision",
            command_id=command.id,
            verdict=decision.verdict.value,
            risk=assessment.risk.value,
            binary=command.binary,
            production=assessment.production_target,
        )
        return decision

    # -- checks -----------------------------------------------------------

    def _approval_reason(self, assessment: RiskAssessment) -> str:
        bits = [f"risk {assessment.risk.value}"]
        if assessment.production_target:
            bits.append("production target")
        if not assessment.reversible:
            bits.append("not automatically reversible")
        ceiling = self.config.auto_execute_max_risk
        bits.append(f"auto-execute ceiling is {ceiling}")
        return "; ".join(bits)

    def _check_context_allowlists(self, command: ProposedCommand) -> list[PolicyViolation]:
        out: list[PolicyViolation] = []
        kube = self.settings.kubernetes
        context = command.context.cluster_context
        if context:
            if kube.denied_contexts and any(
                re.search(p, context) for p in kube.denied_contexts
            ):
                out.append(
                    PolicyViolation(
                        rule="denied_context",
                        message=f"cluster context '{context}' is denied by configuration",
                    )
                )
            if kube.allowed_contexts and not any(
                re.search(p, context) for p in kube.allowed_contexts
            ):
                out.append(
                    PolicyViolation(
                        rule="context_not_allowed",
                        message=(
                            f"cluster context '{context}' is not in the configured allow list"
                        ),
                    )
                )

        sdm = self.settings.sdm
        resource = command.context.sdm_resource
        if resource:
            if sdm.denied_resource_patterns and any(
                re.search(p, resource) for p in sdm.denied_resource_patterns
            ):
                out.append(
                    PolicyViolation(
                        rule="denied_sdm_resource",
                        message=f"SDM resource '{resource}' is denied by configuration",
                    )
                )
            if sdm.allowed_resource_patterns and not any(
                re.search(p, resource) for p in sdm.allowed_resource_patterns
            ):
                out.append(
                    PolicyViolation(
                        rule="sdm_resource_not_allowed",
                        message=f"SDM resource '{resource}' is not in the configured allow list",
                    )
                )
        return out

    def _check_capability_bindings(self, command: ProposedCommand) -> list[PolicyViolation]:
        """ADR 16.5: privileged surfaces stay local."""
        origin = command.metadata.get("origin")
        if origin not in {"facade", "public"}:
            return []
        if self.settings.api.expose_privileged_routes_publicly:
            return []
        return [
            PolicyViolation(
                rule="privileged_tool_via_public_endpoint",
                message=(
                    "command execution is not available through the public inference facade "
                    "(ADR 16.5); run it from the local CLI or web UI"
                ),
            )
        ]

    # -- convenience ------------------------------------------------------

    def classify_only(self, command: ProposedCommand) -> RiskAssessment:
        return self.classifier.classify(command)

    def auto_execute_ceiling(self) -> RiskClass:
        return RiskClass(self.config.auto_execute_max_risk)


_engine: PolicyEngine | None = None


def get_policy_engine(settings: Settings | None = None) -> PolicyEngine:
    global _engine
    if _engine is None or settings is not None:
        _engine = PolicyEngine(settings)
    return _engine


def reset_policy_engine() -> None:
    global _engine
    _engine = None
