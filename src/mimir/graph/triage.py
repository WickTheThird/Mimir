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


# An action the operator can point at a resource and have carried out. These
# are the verbs that make a request an instruction rather than a question.
_RETRIEVAL = re.compile(
    r"""\b(
        logs?|tail|describe|events?|status|restarts?|top|usage|
        rollout|manifest|yaml|image|env|endpoints?
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Something concrete enough to act on without asking which one is meant.
#
# The loose form of the resource branch used to be "kind followed by a word",
# which matched "checkout service. The" in a corpus case about reading supplied
# logs and would have routed a reasoning question to the retrieval loop. A kind
# followed by any word is not a reference to a resource. So a name must arrive
# in one of the shapes a name actually takes: after -n, after namespace or
# context, in kind/name form, or as a hyphenated DNS label, which English words
# are not.
_TARGET = re.compile(
    r"""(
        -n\s+[a-z0-9][\w.-]*
      | \bnamespace\s+[a-z0-9][\w.-]*
      | \bcontext\s+[a-z0-9][\w.-]*
      | \b(?:pod|deployment|statefulset|daemonset|svc|service|node|job|
            cronjob|ingress|configmap|secret)s?\s*/\s*[a-z0-9][\w.-]*
      | \b(?:pod|deployment|statefulset|daemonset|svc|service|node|job|
            cronjob|ingress|configmap|secret)s?\s+(?:named\s+|called\s+)?
        [a-z0-9]+(?:-[a-z0-9]+)+
      | \bkubectl\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# A question about why, or a request to compare, weigh or explain, wants the
# council even when it names a target. The asymmetry from the module docstring
# applies here too: sending an investigation to the direct loop under-answers
# it, which is worse than sending an instruction to the council and being slow.
_DELIBERATIVE = re.compile(
    r"""\b(
        why|root\s+cause|diagnose|investigate|explain\s+why|what\s+caused|
        compare|should\s+i|is\s+it\s+safe|what\s+is\s+wrong|troubleshoot|
        debug|analyse|analyze|recommend
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


class Triage(StrEnum):
    INVESTIGATE = "investigate"
    """Default. Run the council."""

    DIRECT = "direct"
    """An instruction whose target the operator already named.

    "get the last 10 logs from the whatsapp pods in messaging-squad on a ch1
    dev cluster" states the namespace, the workload, the cluster filter, the
    action and the line count. There is nothing left to deliberate: it wants
    carrying out, not investigating. Routed to the operations loop, which reads
    what was named and shows what came back.
    """

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


def _direct_or_investigate(text: str) -> Triage:
    """Both halves are required, and either doubt sends it to the council.

    A retrieval verb alone ("check the logs") names no target. A target alone
    ("the payments namespace") names no action. Only the pair is an
    instruction, and even then a deliberative word takes it back to the
    council, because "why are the whatsapp pods restarting" names both and is
    still a question about cause.
    """
    if _DELIBERATIVE.search(text):
        return Triage.INVESTIGATE
    if _RETRIEVAL.search(text) and _TARGET.search(text):
        return Triage.DIRECT
    return Triage.INVESTIGATE


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
        return TriageResult(_direct_or_investigate(text))

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
