"""Built-in hooks that ship enabled (ADR 10.4, 13.4, 13.5)."""

from __future__ import annotations

import time

from mimir.config import get_settings
from mimir.hooks.manager import HookContext, HookEvent, HookManager, HookVerdict
from mimir.logging import get_correlation_id, get_logger
from mimir.redaction import find_secrets
from mimir.safety.injection import scan

log = get_logger(__name__)


async def audit_metadata(ctx: HookContext) -> HookVerdict:
    """Stamp every tool call with correlation and timing metadata (ADR 20)."""
    return HookVerdict(
        metadata={
            "correlation_id": get_correlation_id(),
            "observed_at": time.time(),
        }
    )


async def validate_kube_target(ctx: HookContext) -> HookVerdict:
    """Refuse a mutation whose namespace or context was never resolved."""
    payload = ctx.payload
    command = str(payload.get("command", ""))
    if "kubectl" not in command:
        return HookVerdict()
    context = payload.get("context") or {}
    if not context.get("cluster_context"):
        return HookVerdict.deny(
            "refusing a kubectl mutation with no resolved cluster context; "
            "run 'kubectl config current-context' and re-issue with an explicit --context"
        )
    if not context.get("namespace") and "--all-namespaces" not in command:
        return HookVerdict.deny(
            "refusing a kubectl mutation with no resolved namespace; pass -n explicitly"
        )
    return HookVerdict()


async def block_protected_namespace(ctx: HookContext) -> HookVerdict:
    settings = ctx.settings or get_settings()
    namespace = (ctx.payload.get("context") or {}).get("namespace")
    if namespace and namespace in settings.safety.protected_namespaces:
        return HookVerdict.deny(
            f"namespace '{namespace}' is protected by configuration and cannot be mutated "
            "through MIMIR"
        )
    return HookVerdict()


async def flag_secrets_in_output(ctx: HookContext) -> HookVerdict:
    """Note when a tool result still looks like it carries credentials."""
    summary = str(ctx.payload.get("summary", ""))
    hits = find_secrets(summary)
    if hits:
        log.warning("secret_pattern_in_tool_summary", tool=ctx.tool_name, rules=hits)
        return HookVerdict(metadata={"secret_patterns": hits})
    return HookVerdict()


async def flag_web_injection(ctx: HookContext) -> HookVerdict:
    """Record injection-shaped web content (ADR 13.5, 17.1)."""
    url = str(ctx.payload.get("url", ""))
    report = scan(str(ctx.payload.get("content", "")))
    if report.suspicious:
        log.warning("web_injection_suspected", url=url, summary=report.summary())
        return HookVerdict(
            metadata={
                "injection_severity": report.severity.value,
                "injection_rules": sorted({f.rule for f in report.findings}),
            }
        )
    return HookVerdict()


async def require_verification_for_promotion(ctx: HookContext) -> HookVerdict:
    """ADR 11.6 and NG4: nothing is promoted to trusted memory silently."""
    status = str(ctx.payload.get("verification_status", "unverified"))
    if status == "unverified":
        return HookVerdict(
            require_extra_approval=True,
            metadata={"promotion_note": "unverified note requires explicit user review"},
        )
    return HookVerdict()


def register_builtin_hooks(manager: HookManager) -> None:
    manager.register(HookEvent.BEFORE_TOOL, audit_metadata, name="audit_metadata")
    manager.register(HookEvent.AFTER_TOOL, flag_secrets_in_output, name="flag_secrets")
    manager.register(HookEvent.BEFORE_MUTATION, validate_kube_target, name="validate_kube_target")
    manager.register(
        HookEvent.BEFORE_MUTATION, block_protected_namespace, name="block_protected_namespace"
    )
    manager.register(HookEvent.ON_WEB_INGEST, flag_web_injection, name="flag_web_injection")
    manager.register(
        HookEvent.ON_MEMORY_PROMOTION,
        require_verification_for_promotion,
        name="require_verification",
    )
