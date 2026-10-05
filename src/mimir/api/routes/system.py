"""Health, capability, knowledge, and skill routes (ADR 6.2 C1, 15, 20)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from mimir import __version__
from mimir.api.auth import Caller, require_local
from mimir.config import get_settings
from mimir.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready(caller: Caller = Depends(require_local)) -> dict[str, Any]:
    """Readiness, including whether the model runtime actually answers."""
    from mimir.llm.router import get_router

    settings = get_settings()
    report = await get_router(settings).health_report()
    models = {alias: {"ok": ok, "detail": detail} for alias, (ok, detail) in report.items()}
    return {
        "status": "ok" if any(m["ok"] for m in models.values()) else "degraded",
        "version": __version__,
        "models": models,
    }


@router.get("/capabilities")
async def capabilities(caller: Caller = Depends(require_local)) -> dict[str, Any]:
    """What this instance can do, for the UI to render honestly."""
    from mimir.tools.base import load_all_tools

    settings = get_settings()
    registry = load_all_tools()
    return {
        "version": __version__,
        "tools": [
            {
                "name": spec.name,
                "capability": spec.capability.value,
                "risk": spec.risk.value,
                "mutating": spec.mutating,
                "description": " ".join(spec.description.split()),
            }
            for spec in registry.all()
        ],
        "safety": {
            "auto_execute_max_risk": settings.safety.auto_execute_max_risk,
            "require_approval_for_mutations": settings.safety.require_approval_for_mutations,
            "protected_namespaces": settings.safety.protected_namespaces,
        },
        "models": {
            "public_alias": settings.models.public_alias,
            "profiles": {
                alias: {"runtime": p.runtime, "model": p.model, "context": p.context_window}
                for alias, p in settings.models.profiles.items()
            },
            "routing": settings.models.routing.model_dump(),
        },
    }


@router.get("/skills")
async def list_skills(caller: Caller = Depends(require_local)) -> dict[str, Any]:
    from mimir.skills.registry import get_skill_registry

    registry = get_skill_registry()
    return {
        "skills": [
            {
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "when_to_use": skill.when_to_use,
                "specialist": getattr(skill.specialist, "value", str(skill.specialist)),
                "max_risk": getattr(skill.max_risk, "value", str(skill.max_risk)),
                "allowed_tools": list(skill.allowed_tools),
                "tags": list(skill.tags),
            }
            for skill in registry.all()
        ],
        "catalogue_tokens": registry.catalogue_tokens(),
    }


@router.get("/skills/{name}")
async def get_skill(name: str, caller: Caller = Depends(require_local)) -> dict[str, Any]:
    from mimir.skills.registry import get_skill_registry
    from mimir.skills.runner import SkillRunner

    try:
        loaded = SkillRunner(get_skill_registry()).load(name)
    except Exception as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {
        "name": loaded.skill.name,
        "body": loaded.body,
        "resources": [r.name for r in loaded.skill.references],
        "scripts": [s.name for s in loaded.skill.scripts],
    }


@router.get("/memory/search")
async def search_memory(
    q: str,
    limit: int = 8,
    caller: Caller = Depends(require_local),
) -> dict[str, Any]:
    from mimir.knowledge.index import get_knowledge_index
    from mimir.knowledge.retrieval import MemoryRetriever

    result = MemoryRetriever(get_knowledge_index()).search(q, top_k=limit)
    return {
        "query": q,
        "results": [chunk.to_payload() for chunk in result.chunks],
        "conflicts": [conflict.render() for conflict in result.conflicts],
    }


@router.get("/memory/documents/{doc_id:path}")
async def read_memory(doc_id: str, caller: Caller = Depends(require_local)) -> dict[str, Any]:
    from mimir.knowledge.store import get_knowledge_store

    settings = get_settings()
    document = get_knowledge_store(settings).get(doc_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown document {doc_id}")
    return {
        "document_id": doc_id,
        "metadata": json.loads(json.dumps(document.metadata.to_frontmatter(), default=str)),
        "freshness": document.freshness(settings.knowledge.stale_after_days).value,
        "body": document.body,
    }


@router.get("/metrics")
async def metrics(caller: Caller = Depends(require_local)) -> dict[str, Any]:
    """Telemetry snapshot (ADR 20)."""
    from mimir.observability.metrics import snapshot

    return snapshot()
