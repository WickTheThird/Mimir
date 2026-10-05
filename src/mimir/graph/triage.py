"""Deterministic triage before the council runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_GREETING = re.compile(
    r"^(hi|hello|hey|yo|hiya|howdy|greetings|good\s+(morning|afternoon|evening|day))"
    r"[\s!.,]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(thanks|thank\s+you|ta|cheers|nice|great|cool|ok|okay)[\s!.,]*$", re.I)
_FAREWELL = re.compile(r"^(bye|goodbye|see\s+you|later|good\s?night|quit|exit)[\s!.,]*$", re.I)
_SMALL_TALK = re.compile(
    r"^(how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|"
    r"are\s+you\s+(there|ok|alive)|test|testing|ping)[\s!?.,]*$",
    re.IGNORECASE,
)

# Anything naming a concrete artefact is a real request regardless of how short
_CONCRETE = re.compile(
    r"""(
        [\w./-]+\.(?:py|ts|tsx|js|go|rs|java|rb|sql|ya?ml|toml|json|md)
      | \b(kubectl|psql|docker|systemctl|journalctl|curl|git|npm|pytest|helm|grep|awk)\b
      | https?://
      | \b(pod|deployment|namespace|cluster|service|queue|schema|migration|index|
           repo|repository|commit|branch|log|logs|error|errors|timeout|restart|
           crash|failure|incident)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


# A request to comment on material already in the prompt, or to weigh causes,
_MUTATION = re.compile(
    r"""\b(
        restart|rollout|scale|delete|remove|drain|cordon|uncordon|evict|
        apply|patch|edit|create|update|upgrade|downgrade|rollback|revert|
        raise|lower|increase|decrease|set|enable|disable|kill|terminate|
        deploy|redeploy|promote|failover|reset
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

_DELIBERATIVE = re.compile(
    r"""\b(
        why|root\s+cause|diagnose|investigate|explain|what\s+caused|
        compare|should\s+i|is\s+it\s+safe|what\s+is\s+wrong|troubleshoot|
        debug|analyse|analyze|recommend|rank|indicate|indicates|likely|
        which\s+side|what\s+does|these\s+logs\s+are|suggests?
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

class Triage(StrEnum):
    INVESTIGATE = "investigate"
    """Default. Run the council."""

    DIRECT = "direct"
    """An instruction whose target the operator already named."""

    GREETING = "greeting"
    CAPABILITY = "capability"
    """Asking what MIMIR is or can do. Answerable from static text."""

    ACKNOWLEDGEMENT = "acknowledgement"
    FAREWELL = "farewell"
    EMPTY = "empty"


@dataclass
class TriageResult:
    kind: Triage
    reply: str = ""

    @property
    def cheap(self) -> bool:
        """Whether the graph can be skipped entirely and a reply returned."""
        return self.kind not in (Triage.INVESTIGATE, Triage.DIRECT)


_REPLIES = {
    Triage.GREETING: (
        "Hello. Ask me about a repository, a cluster, a log, or a command and I "
        "will investigate it. Read-only work runs without asking; anything that "
        "changes state stops for approval."
    ),
    Triage.CAPABILITY: (
        "I investigate operational and repository questions locally: reading "
        "code, tracing flows, inspecting Kubernetes and containers, analysing "
        "logs, and constructing commands. Every answer is tied to evidence, and "
        "any command that changes state needs your approval first. Type /help "
        "for the command list."
    ),
    Triage.ACKNOWLEDGEMENT: "Noted. What would you like me to look at?",
    Triage.FAREWELL: "Goodbye.",
    Triage.EMPTY: "Ask me a question about a repository, cluster, log, or command.",
}


def _direct_or_investigate(text: str) -> Triage:
    """Both halves are required, and either doubt sends it to the council."""
    if _DELIBERATIVE.search(text) or _MUTATION.search(text):
        return Triage.INVESTIGATE

    from mimir.agent.request import parse_request

    request = parse_request(text)
    targeted = bool(
        request.namespace
        or request.context
        or request.context_contains
        or request.name_contains
    )
    if request.action and targeted:
        return Triage.DIRECT
    return Triage.INVESTIGATE


def triage(question: str) -> TriageResult:
    """Decide whether this input needs an investigation at all."""
    text = (question or "").strip()
    if not text:
        return TriageResult(Triage.EMPTY, _REPLIES[Triage.EMPTY])

    # A real request can be short and can open with a pleasantry.
    if _CONCRETE.search(text):
        return TriageResult(_direct_or_investigate(text))

    # Length bound: a long message is doing more than saying hello, even if it
    if len(text) > 64:
        return TriageResult(Triage.INVESTIGATE)

    for pattern, kind in (
        (_GREETING, Triage.GREETING),
        (_SMALL_TALK, Triage.CAPABILITY),
        (_THANKS, Triage.ACKNOWLEDGEMENT),
        (_FAREWELL, Triage.FAREWELL),
    ):
        if pattern.match(text):
            return TriageResult(kind, _REPLIES[kind])

    return TriageResult(Triage.INVESTIGATE)
