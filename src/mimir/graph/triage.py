"""Deterministic triage before the council runs.

MIMIR had no cheap path. Every input traversed context resolution, memory
recall, an LLM classification, skill selection, two or more specialists,
verification, safety review and synthesis. Typing ``hello`` at the prompt was
classified as ``command_construction``, assigned to the Kubernetes
investigator, and took over a minute.

That is the memo's governing rule violated in its most literal form: do not
make every query traverse the whole model, all stored memory, all tools, and
all specialists. A greeting needs none of them.

This module answers one question mechanically, with no model call: is this
input something to investigate at all? It is the cheapest possible instance of
compiled procedure versus deliberative search.

**The asymmetry is deliberate.** Failing to triage a greeting wastes a minute
and some heat. Wrongly triaging a real question means refusing to work, which
is far worse. So every rule here is narrow, anchored, and biased toward
investigating. When in doubt, run the council.
"""

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
# it looks. This is checked first and overrides every pattern above.
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


class Triage(StrEnum):
    INVESTIGATE = "investigate"
    """Default. Run the council."""

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
        return self.kind is not Triage.INVESTIGATE


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


def triage(question: str) -> TriageResult:
    """Decide whether this input needs an investigation at all.

    Bounded on purpose: only inputs that are *entirely* conversational are
    diverted, and only when they name nothing concrete. ``hello`` is a greeting;
    ``hello, why is the api pod restarting`` is a question.
    """
    text = (question or "").strip()
    if not text:
        return TriageResult(Triage.EMPTY, _REPLIES[Triage.EMPTY])

    # A real request can be short and can open with a pleasantry. Anything
    # naming a file, command, resource or operational noun is investigated,
    # whatever else it looks like.
    if _CONCRETE.search(text):
        return TriageResult(Triage.INVESTIGATE)

    # Length bound: a long message is doing more than saying hello, even if it
    # happens to start with a greeting word.
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
