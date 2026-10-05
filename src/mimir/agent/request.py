"""What the operator asked for, extracted by rule."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# -- vocabulary --------------------------------------------------------------

KIND_NOUNS = (
    "pod", "pods", "deployment", "deployments", "statefulset", "statefulsets",
    "daemonset", "daemonsets", "service", "services", "workload", "workloads",
    "container", "containers", "job", "jobs", "cronjob", "cronjobs",
)

ENVIRONMENTS = ("dev", "development", "prod", "production", "staging", "stage",
                "qa", "uat", "sandbox", "test")

#: Words that can precede a kind noun without being part of a name.
_QUALIFIERS = frozenset({
    "any", "all", "some", "the", "a", "an", "every", "each", "one", "first",
    "last", "latest", "newest", "oldest", "my", "our", "that", "this", "those",
    "these", "which", "what", "of", "for", "from", "in", "on", "at", "to",
    "me", "us", "please", "can", "you", "tell", "show", "give", "get", "see",
    "check", "find", "list", "fetch", "pull", "read", "look", "want", "need",
    "curious", "about", "is", "are", "inside", "within", "running", "live",
    "single", "specific", "particular", "healthy", "unhealthy", "failing",
    "broken", "restarting", "crashing", "new", "old",
})

ACTIONS = {
    "logs": ("log", "logs", "tail", "output", "stdout", "stderr"),
    "events": ("event", "events"),
    "describe": ("describe", "description", "manifest", "yaml", "spec"),
    "status": ("status", "health", "ready", "readiness", "state"),
    "restarts": ("restart", "restarts", "crashloop", "crashlooping", "oomkill"),
    "usage": ("usage", "cpu", "memory", "top", "resources"),
}

# -- patterns ----------------------------------------------------------------

# Order matters.
_NAMESPACE = (
    re.compile(r"(?:-n|--namespace)\s+([a-z0-9][\w.-]*)", re.IGNORECASE),
    re.compile(r"\b([a-z0-9][\w.-]*)\s+namespace\b", re.IGNORECASE),
    re.compile(r"\bnamespace\s+(?:called\s+|named\s+)?([a-z0-9][\w.-]*)", re.IGNORECASE),
)

_CONTEXT_EXACT = re.compile(r"(?:--context|\bcontext)\s+([a-z0-9][\w.-]*)", re.IGNORECASE)

_CONTEXT_FRAGMENT = (
    # "a cluster that has ch1 inside of it", "a cluster with ch1 in it"
    re.compile(
        r"\bclusters?\s+(?:that\s+)?(?:has|have|with|containing|contains|including)\s+"
        r"(?:the\s+)?['\"]?([a-z0-9][\w.-]*)['\"]?",
        re.IGNORECASE,
    ),
    # "ch1 cluster", "the ch1 dev cluster".
    re.compile(
        r"\b([a-z0-9][\w.-]*)(?:\s+([a-z0-9][\w.-]*))?\s+clusters?\b", re.IGNORECASE
    ),
)

# A count, not a duration.
_TAIL = re.compile(
    r"\b(?:last|latest|final|most\s+recent|top)\s+(\d{1,5})\b"
    r"(?!\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|weeks?)\b)"
    r"|\b(\d{1,5})\s+(?:lines?|logs?|entries|events?|rows?)\b",
    re.IGNORECASE,
)

_SINCE = re.compile(
    r"\b(?:last|past|previous|within|over|since)\s+(\d{1,4})\s*"
    r"(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\b",
    re.IGNORECASE,
)

_SINCE_UNIT = {"m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m",
               "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
               "d": "d", "day": "d", "days": "d"}

#: Words that never belong to a workload name or a scope, even in the right
_NOT_A_NAME = frozenset(ENVIRONMENTS) | {
    "cluster", "clusters", "namespace", "namespaces", "context", "contexts",
    "over", "under", "the", "a", "an", "and", "or", "of", "in", "on", "at",
    "for", "from", "to", "with", "within", "inside", "this", "that", "last",
    "past", "any", "all", "some", "it", "its", "is", "are", "was", "were",
}


@dataclass
class ParsedRequest:
    """Only what was actually stated. Every field can be absent."""

    namespace: str = ""
    context: str = ""
    """A context named exactly."""

    context_contains: str = ""
    """A fragment the operator described the cluster by."""

    environment: str = ""
    name_contains: str = ""
    """The most specific name fragment stated."""

    name_candidates: list[str] = field(default_factory=list)
    """Fragments from most to least specific, so a caller can widen."""

    action: str = ""
    tail: int = 0
    since: str = ""

    @property
    def stated(self) -> dict[str, object]:
        """Only the fields that were found, for display and for binding."""
        out: dict[str, object] = {}
        for key in ("namespace", "context", "context_contains", "environment",
                    "name_contains", "action", "since"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.tail:
            out["tail"] = self.tail
        return out

    def render(self) -> str:
        if not self.stated:
            return ""
        pairs = ", ".join(f"{k}={v}" for k, v in self.stated.items())
        return f"The operator stated: {pairs}. Use these exactly; do not substitute."


def _first(patterns, text: str) -> str:
    """The first group of the first match whose value is a real name."""
    for pattern in patterns:
        for match in pattern.finditer(text):
            for value in match.groups():
                if value and value.strip().lower() not in _NOT_A_NAME:
                    return value.strip()
    return ""


def _name_fragments(text: str) -> list[str]:
    """Name fragments, most specific first."""
    lowered = text.lower()
    out: list[str] = []
    for match in re.finditer(r"\b(" + "|".join(KIND_NOUNS) + r")\b", lowered):
        before = re.findall(r"[a-z0-9][\w.-]*", lowered[: match.start()])
        words: list[str] = []
        for word in reversed(before):
            if word in _QUALIFIERS or word in _NOT_A_NAME:
                break
            words.insert(0, word)
            if len(words) == 3:
                break
        if not words:
            continue
        if len(words) > 1:
            out.append("-".join(words))
        # The last word is the distinctive one far more often than the first:
        out.append(words[-1])

    # A name written after the kind ("deployment api", "pod api-7c9") is stated too.
    for match in re.finditer(r"\b(?:" + "|".join(KIND_NOUNS) + r")\s+(?:named\s+|called\s+)?([a-z0-9][\w.-]*)", lowered):
        word = match.group(1)
        if word not in _QUALIFIERS and word not in _NOT_A_NAME and len(word) > 2:
            out.append(word)
    # A name written as kind/name, and a hyphenated token anywhere, are names
    explicit = [
        match.group(1)
        for match in re.finditer(
            r"\b(?:" + "|".join(KIND_NOUNS) + r")\s*/\s*([a-z0-9][\w.-]*)", lowered
        )
    ]
    explicit += [
        token for token in re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", lowered)
        if token not in _NOT_A_NAME
    ]
    ordered = explicit + out
    seen: set[str] = set()
    return [f for f in ordered if len(f) > 2 and not (f in seen or seen.add(f))]


def parse_request(text: str) -> ParsedRequest:
    """Extract the stated parameters. Never guesses."""
    text = text or ""
    lowered = text.lower()

    parsed = ParsedRequest()
    parsed.namespace = _first(_NAMESPACE, text)
    parsed.context = _first((_CONTEXT_EXACT,), text)
    if not parsed.context:
        parsed.context_contains = _first(_CONTEXT_FRAGMENT, text)

    for environment in ENVIRONMENTS:
        if re.search(rf"\b{environment}\b", lowered):
            parsed.environment = environment
            break

    for action, words in ACTIONS.items():
        if any(re.search(rf"\b{w}\b", lowered) for w in words):
            parsed.action = action
            break

    tail = _TAIL.search(text)
    if tail:
        parsed.tail = int(tail.group(1) or tail.group(2))

    since = _SINCE.search(text)
    if since and not (tail and since.group(1) == (tail.group(1) or tail.group(2))):
        parsed.since = f"{since.group(1)}{_SINCE_UNIT[since.group(2).lower()]}"

    candidates = [
        c for c in _name_fragments(text)
        if c != parsed.namespace.lower() and c != parsed.context_contains.lower()
    ]
    parsed.name_candidates = candidates
    parsed.name_contains = candidates[0] if candidates else ""
    return parsed


__all__ = ["ACTIONS", "ENVIRONMENTS", "KIND_NOUNS", "ParsedRequest", "parse_request"]
