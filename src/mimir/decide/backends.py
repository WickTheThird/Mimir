"""Where a typed decision actually gets scored.

Three shapes exist in the open implementations and MIMIR should not care which
is installed. Nimble runs in-process on Apple Silicon through MLX and wants
about eighteen gigabytes; Kev ships a local HTTP server and comes as small as
0.8B; Von is 395M on CPU. The differences are memory and accuracy, not
interface, so the backend is configuration.

That matters for more than tidiness. A decision model small enough to run
beside the generative one, or on a machine that cannot hold a generative one at
all, is the difference between MIMIR needing a workstation and MIMIR running on
whatever is spare.
"""

from __future__ import annotations

import json
from typing import Any

from mimir.decide.base import Choice, Verdict, clip
from mimir.logging import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0


class HttpDecider:
    """A decision server on the other end of a socket.

    Written against the shape the open servers expose: one context, a flat
    schema of enum fields, probabilities back. No generated JSON is parsed,
    because none is generated; the numbers are a softmax over the options.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        model: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._checked: bool | None = None

    @property
    def name(self) -> str:
        return f"http:{self.model or self.base_url}"

    @property
    def available(self) -> bool:
        """Asked once per process, and never allowed to raise.

        A decision model that is not running is a reason to fall back, not a
        reason to fail a turn.
        """
        if self._checked is None:
            self._checked = self._probe()
        return self._checked

    def _probe(self) -> bool:
        import httpx

        for path in ("/health", "/healthz", "/"):
            try:
                response = httpx.get(f"{self.base_url}{path}", timeout=3.0)
            except httpx.HTTPError:
                continue
            if response.status_code < 500:
                return True
        log.info("decider_unavailable", base_url=self.base_url)
        return False

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        import httpx

        if not fields:
            return {}
        text, truncated = clip(context)
        payload: dict[str, Any] = {
            "context": text,
            "schema": {f.name: {"enum": list(f.options), "description": f.description}
                       for f in fields},
        }
        if self.model:
            payload["model"] = self.model
        try:
            response = httpx.post(
                f"{self.base_url}/decide", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("decider_failed", error=str(exc))
            return {}
        return _verdicts(body, fields, truncated)


def _verdicts(
    body: dict[str, Any], fields: list[Choice], truncated: bool
) -> dict[str, Verdict]:
    """Read a distribution per field, refusing anything not offered.

    A probability over an option that was never in the schema is not a decision
    about this question, whatever it is. Dropping it is the only safe reading.
    """
    out: dict[str, Verdict] = {}
    for choice in fields:
        raw = (body.get(choice.name) or body.get("fields", {}).get(choice.name) or {})
        distribution = raw.get("probabilities") or raw.get("distribution") or raw
        if not isinstance(distribution, dict):
            continue
        allowed = {
            option: float(distribution[option])
            for option in choice.options
            if isinstance(distribution.get(option), int | float)
        }
        if not allowed:
            continue
        total = sum(allowed.values()) or 1.0
        allowed = {k: round(v / total, 4) for k, v in allowed.items()}
        best = max(allowed, key=lambda k: allowed[k])
        out[choice.name] = Verdict(
            field=choice.name,
            choice=best,
            probability=allowed[best],
            distribution=allowed,
            truncated=truncated,
        )
    return out


class NimbleDecider:
    """Nimble in-process, through MLX on Apple Silicon.

    Imported lazily and never at module scope: MLX is an optional dependency
    and a machine without it must still be able to load this module.
    """

    def __init__(self, model_path: str, *, adapter_path: str = "") -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._scorer: Any = None
        self._tried = False

    @property
    def name(self) -> str:
        return f"nimble:{self.model_path}"

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._tried:
            return self._scorer
        self._tried = True
        try:
            from nimble import ParallelScorer  # type: ignore[import-not-found]

            self._scorer = ParallelScorer(
                self.model_path, adapter_path=self.adapter_path or None
            )
        except Exception as exc:  # noqa: BLE001 - absence is not an error here
            log.info("nimble_unavailable", error=str(exc))
            self._scorer = None
        return self._scorer

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        scorer = self._load()
        if scorer is None or not fields:
            return {}
        text, truncated = clip(context)
        schema = {f.name: list(f.options) for f in fields}
        try:
            body = scorer.score(text, schema)
        except Exception as exc:  # noqa: BLE001 - a backend fault is a fallback
            log.warning("nimble_failed", error=str(exc))
            return {}
        return _verdicts(body, fields, truncated)


def build_decider(settings: Any) -> Any:
    """The configured backend, or one that answers nothing."""
    from mimir.decide.base import NoDecider

    config = getattr(settings, "decisions", None)
    if config is None or not getattr(config, "enabled", False):
        return NoDecider()
    kind = str(getattr(config, "backend", "http"))
    if kind == "nimble":
        return NimbleDecider(
            str(getattr(config, "model_path", "")),
            adapter_path=str(getattr(config, "adapter_path", "")),
        )
    return HttpDecider(
        str(getattr(config, "base_url", "http://127.0.0.1:8080")),
        model=str(getattr(config, "model", "")),
    )


__all__ = ["HttpDecider", "NimbleDecider", "build_decider"]
