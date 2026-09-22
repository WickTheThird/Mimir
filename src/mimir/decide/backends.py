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


class KevDecider:
    """Kev's local System One server.

    Written against the published API rather than guessed: POST /v1/systemone
    with a state and a map of questions, each a noul, choice or score, and the
    answers come back with probabilities and a confidence. Nothing is generated
    and nothing is parsed out of prose.

    Kev-0.8B is about three gigabytes, which is what makes this usable on a
    machine that is also running a generative model and has sixteen gigabytes
    in total.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8009",
        *,
        model: str = "kev-latest",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or "kev-latest"
        self.timeout = timeout
        self._checked: bool | None = None

    @property
    def name(self) -> str:
        return f"kev:{self.model}"

    @property
    def available(self) -> bool:
        """Asked once per process, and never allowed to raise.

        A decision server that is not running is a reason to fall back, not a
        reason to fail a turn.
        """
        if self._checked is None:
            self._checked = self._probe()
        return self._checked

    def _probe(self) -> bool:
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=3.0)
            try:
                models = response.json().get("models") or []
                temps = [float(m.get("temperature", 1.0)) for m in models]
                # Kev stores a fitted temperature per checkpoint and applies
                # it at load. 1.0 means raw logits: the Qwen3-revision
                # checkpoints report it, and their probabilities are then
                # not calibrated however confident they look.
                self._calibrated = any(t != 1.0 for t in temps) if temps else False
            except (ValueError, TypeError, AttributeError):
                self._calibrated = False
        except httpx.HTTPError as exc:
            log.info("kev_unavailable", base_url=self.base_url, error=str(exc))
            return False
        return response.status_code < 500

    def decide(self, context: str, fields: list[Choice]) -> dict[str, Verdict]:
        import httpx

        if not fields:
            return {}
        state, truncated = clip(context)
        payload = {
            "state": state,
            "model": self.model,
            "questions": {
                field.name: {
                    "type": "noul" if _is_boolean(field) else "choice",
                    "instructions": field.description or field.name,
                    "criteria": _criteria(field),
                }
                for field in fields
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/v1/systemone", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("kev_failed", error=str(exc))
            return {}
        verdicts = _from_kev(body.get("answers") or {}, fields, truncated)
        if not getattr(self, "_calibrated", False):
            verdicts = {
                k: Verdict(field=v.field, choice=v.choice, probability=v.probability,
                           distribution=v.distribution, truncated=v.truncated,
                           calibrated=False)
                for k, v in verdicts.items()
            }
        return verdicts


def _is_boolean(field: Choice) -> bool:
    return {o.lower() for o in field.options} in ({"yes", "no"}, {"true", "false"})


def _criteria(field: Choice) -> Any:
    """Kev wants a description per option; the name alone is an honest one."""
    if _is_boolean(field):
        ordered = sorted(field.options, key=lambda o: o.lower() in ("no", "false"))
        return {"true": ordered[0], "false": ordered[-1]}
    return dict.fromkeys(field.options)


def _from_kev(
    answers: dict[str, Any], fields: list[Choice], truncated: bool
) -> dict[str, Verdict]:
    """Read Kev's answers into verdicts, keeping only offered options."""
    out: dict[str, Verdict] = {}
    for field in fields:
        answer = answers.get(field.name)
        if not isinstance(answer, dict):
            continue
        if answer.get("type") == "noul" and isinstance(answer.get("noul"), int | float):
            probability = float(answer["noul"])
            ordered = sorted(field.options, key=lambda o: o.lower() in ("no", "false"))
            distribution = {ordered[0]: round(probability, 4),
                            ordered[-1]: round(1.0 - probability, 4)}
        else:
            raw = answer.get("probabilities") or {}
            distribution = {
                option: float(raw[option])
                for option in field.options
                if isinstance(raw.get(option), int | float)
            }
            if not distribution:
                continue
            total = sum(distribution.values()) or 1.0
            distribution = {k: round(v / total, 4) for k, v in distribution.items()}
        best = max(distribution, key=lambda k: distribution[k])
        out[field.name] = Verdict(
            field=field.name,
            choice=best,
            probability=distribution[best],
            distribution=distribution,
            truncated=truncated,
        )
    return out


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


def build_decider(settings: Any, *, router: Any = None) -> Any:
    """The configured backend, or one that answers nothing."""
    from mimir.decide.base import NoDecider

    config = getattr(settings, "decisions", None)
    if config is None or not getattr(config, "enabled", False):
        return NoDecider()
    kind = str(getattr(config, "backend", "kev"))
    if kind == "local":
        from mimir.decide.local import LocalDecider

        return LocalDecider(router) if router is not None else NoDecider()
    if kind == "nimble":
        return NimbleDecider(
            str(getattr(config, "model_path", "")),
            adapter_path=str(getattr(config, "adapter_path", "")),
        )
    return KevDecider(
        str(getattr(config, "base_url", "http://127.0.0.1:8009")),
        model=str(getattr(config, "model", "kev-latest")),
    )


__all__ = ["KevDecider", "NimbleDecider", "build_decider"]
