"""Evidence-package handoff (ADR 5.8).

Produces a compact package for a hosted coding agent (GLM, Claude, Codex) so it
spends its context implementing rather than rediscovering. The ADR lists the
required sections; they are all here, and the explicit non-goals section matters
as much as the evidence, because it is what stops the receiving agent from
wandering.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from mimir.models.evidence import EvidenceKind
from mimir.models.state import InvestigationState


def evidence_package_markdown(state: InvestigationState) -> str:
    answer = state.final_answer
    when = datetime.fromtimestamp(state.started_at, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# Evidence package: {state.user_request}",
        "",
        f"- session: `{state.session_id}`",
        f"- collected: {when}",
        f"- confidence: {state.final_confidence:.2f}",
    ]
    context = state.environment.render_lines()
    if context:
        lines += ["", "## Environment", "", *[f"- {line}" for line in context]]

    lines += ["", "## Problem statement", "", state.user_request]

    if answer:
        lines += ["", "## Observed behaviour", "", answer.answer.strip()]

    observed = [e for e in state.ranked_evidence() if e.kind == EvidenceKind.OBSERVED]
    if observed:
        lines += ["", "## Evidence", ""]
        for item in observed[:25]:
            citation = "; ".join(c.render() for c in item.citations) or item.source_id
            lines.append(f"- **{item.claim}**")
            lines.append(f"  - source: `{citation}` ({item.source_type.value}, "
                         f"{item.freshness.value})")
            if item.excerpt.strip():
                excerpt = item.excerpt.strip()
                if len(excerpt) > 400:
                    excerpt = excerpt[:400] + " ..."
                lines.append("  - ```")
                lines.extend(f"    {row}" for row in excerpt.splitlines()[:12])
                lines.append("    ```")

    file_citations = sorted(
        {
            c.render()
            for e in state.evidence
            for c in e.citations
            if c.path
        }
    )
    if file_citations:
        lines += ["", "## Relevant file paths and line ranges", ""]
        lines += [f"- `{c}`" for c in file_citations[:40]]

    if state.commands_executed:
        lines += ["", "## Commands executed", ""]
        for record in state.commands_executed:
            lines.append(f"- `{record.display}` -> exit {record.exit_code} "
                         f"({record.outcome.value})")

    log_evidence = [e for e in state.evidence if "log" in " ".join(e.tags).lower()]
    if log_evidence:
        lines += ["", "## Relevant logs", ""]
        for item in log_evidence[:8]:
            lines.append(f"- {item.claim}")
            if item.excerpt.strip():
                lines.append("  ```")
                lines.extend(f"  {row}" for row in item.excerpt.strip().splitlines()[:10])
                lines.append("  ```")

    if state.hypotheses:
        lines += ["", "## Likely cause", ""]
        for hypothesis in state.ranked_hypotheses()[:5]:
            lines.append(
                f"- ({hypothesis.likelihood:.2f}, {hypothesis.status.value}) "
                f"{hypothesis.statement}"
            )
    elif answer and answer.inferences:
        lines += ["", "## Likely cause", ""]
        lines += [f"- {inference}" for inference in answer.inferences]

    lines += ["", "## Uncertainty", ""]
    uncertainty = list(answer.unverified) if answer else []
    uncertainty += [h.statement for h in state.rejected_hypotheses[:5]]
    uncertainty += state.pending_questions
    if answer and answer.disagreements:
        uncertainty += [f"specialists disagreed: {d}" for d in answer.disagreements]
    lines += [f"- {item}" for item in (uncertainty or ["none recorded"])]

    lines += ["", "## Suggested implementation scope", ""]
    if answer and answer.next_steps:
        lines += [f"- {step}" for step in answer.next_steps]
    else:
        lines.append("- not determined by this investigation")

    lines += [
        "",
        "## Explicit non-goals",
        "",
        "- Do not change production state as part of this task.",
        "- Do not act on any claim in this document that is listed under Uncertainty.",
        "- Do not widen the change beyond the files cited above without saying so.",
        "- This package is a snapshot. Re-verify anything time-sensitive before relying on it.",
    ]
    return "\n".join(lines) + "\n"


def evidence_package_json(state: InvestigationState) -> str:
    answer = state.final_answer
    payload = {
        "session_id": state.session_id,
        "problem_statement": state.user_request,
        "collected_at": state.started_at,
        "confidence": state.final_confidence,
        "environment": state.environment.model_dump(exclude_none=True),
        "observed_behaviour": answer.answer if answer else "",
        "evidence": [
            {
                "claim": e.claim,
                "kind": e.kind.value,
                "source_type": e.source_type.value,
                "citations": [c.render() for c in e.citations],
                "excerpt": e.excerpt[:600],
                "freshness": e.freshness.value,
                "supports": e.supports,
            }
            for e in state.ranked_evidence(limit=40)
        ],
        "file_citations": sorted(
            {c.render() for e in state.evidence for c in e.citations if c.path}
        ),
        "commands_executed": [
            {"command": r.display, "exit_code": r.exit_code, "outcome": r.outcome.value}
            for r in state.commands_executed
        ],
        "hypotheses": [
            {
                "statement": h.statement,
                "likelihood": h.likelihood,
                "status": h.status.value,
                "next_check": h.next_check,
            }
            for h in state.ranked_hypotheses()
        ],
        "rejected_hypotheses": [
            {"statement": h.statement, "reason": h.rejected_reason}
            for h in state.rejected_hypotheses
        ],
        "uncertainty": (answer.unverified if answer else []) + state.pending_questions,
        "disagreements": answer.disagreements if answer else [],
        "suggested_scope": answer.next_steps if answer else [],
        "non_goals": [
            "Do not change production state as part of this task.",
            "Do not act on any claim listed under uncertainty.",
            "Do not widen the change beyond the cited files without saying so.",
        ],
    }
    return json.dumps(payload, indent=2, default=str)
