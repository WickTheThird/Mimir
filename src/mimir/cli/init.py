"""First-run setup (``mimir init``)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from mimir.config import DEFAULT_HOME, get_settings, reset_settings_cache

CONFIG_TEMPLATE = """\
# MIMIR configuration. See docs/ADR-001.md for the reasoning behind each area.
# Every value here is a decision the ADR deliberately left open (section 25),
# so change them freely as you benchmark.

environment: local

models:
  # Point these at whatever runtime you settle on: Ollama, llama.cpp, MLX-LM,
  # vLLM, or LiteLLM. They all speak the OpenAI chat-completions API.
  profiles:
    fast:
      alias: fast
      runtime: ollama
      base_url: http://127.0.0.1:11434/v1
      model: qwen2.5-coder:7b
      context_window: 32768
    deep:
      alias: deep
      runtime: ollama
      base_url: http://127.0.0.1:11434/v1
      model: qwen2.5:32b
      context_window: 32768
    embed:
      alias: embed
      runtime: ollama
      base_url: http://127.0.0.1:11434/v1
      model: nomic-embed-text
      supports_tools: false
      supports_json_schema: false
  # Route task classes to profiles. Point everything at one alias to start.
  routing:
    default: deep
    fast_command: fast
    deep_investigation: deep
    classification: fast
    final_synthesis: deep
    embedding: embed
  public_alias: mimir-local-ops

repos:
  # Directories to scan for git checkouts.
  roots: []
  # Or name them explicitly:
  # entries:
  #   - name: billing
  #     path: ~/src/billing
  #     description: billing service

safety:
  # Highest risk class that may run with no human interaction.
  # R0 analysis, R1 read-only, R2 elevated inspection, R3 reversible mutation,
  # R4 high risk. Leave this at R1 unless you have a good reason.
  auto_execute_max_risk: R1
  require_approval_for_mutations: true
  command_timeout_s: 120
  # Anything matching these in a context, namespace, or resource name is
  # treated as production and never auto-executed.
  production_context_patterns: ["prod", "production", "live", "prd"]
  protected_namespaces: ["kube-system", "kube-public", "istio-system"]

kubernetes:
  enabled: true
  # default_context: <fill in with `kubectl config get-contexts`>
  # default_namespace: <fill in>
  # allowed_contexts: []   # empty means all contexts are permitted
  # denied_contexts: []

sdm:
  enabled: true
  container_cli: docker
  # allowed_resource_patterns: []
  # denied_resource_patterns: []

web:
  enabled: true
  search_provider: ddg      # ddg, brave, tavily, searxng, or none
  respect_robots: true
  max_results: 8

api:
  host: 127.0.0.1
  port: 8756
  allow_loopback_without_auth: true
  # Generate with `mimir keys create`. Required for anything non-loopback.
  api_keys: []
  # ADR 16.4: the Warp-facing endpoint is a model gateway, not a nested agent.
  facade_agent_mode: false
  # ADR 16.5: privileged helpers stay loopback-only. Do not enable this.
  expose_privileged_routes_publicly: false

knowledge:
  embeddings_enabled: true
  stale_after_days: 180

observability:
  log_level: INFO
  json_logs: false
"""

ENVIRONMENT_TEMPLATE = """\
---
title: Environment conventions
category: stable/environments
confidence: low
verification_status: unverified
tags: [conventions, environment, placeholder]
---

# Environment conventions

This file is a TEMPLATE. MIMIR will read it as fact once you fill it in, so
replace every placeholder below with real values and delete the ones that do not
apply. Leave it empty rather than guessing: a wrong namespace here is worse than
no namespace, because it will be treated as curated knowledge.

## Clusters

| Purpose | Context name | Notes |
| --- | --- | --- |
| production | `<fill in>` | |
| staging | `<fill in>` | |

Get the real names with `kubectl config get-contexts`.

## Namespaces

| Service or team | Namespace | Cluster |
| --- | --- | --- |
| `<fill in>` | `<fill in>` | `<fill in>` |

## Naming conventions

Describe how workloads, namespaces, and SDM resources are named here, for
example whether deployments are `<service>` or `<service>-api`. MIMIR uses this
to construct commands without guessing.

## SDM resources

The ADR is explicit that SDM resource naming is environment specific and must
not be invented. Record the real names here after running `sdm ls`.

| Resource | Type | What it reaches |
| --- | --- | --- |
| `<fill in>` | | |

## Who owns what

| Service | Repository | Team or owner |
| --- | --- | --- |
| `<fill in>` | | |
"""


def config_path() -> Path:
    return DEFAULT_HOME / "config.yaml"


def update_config(changes: dict[str, Any]) -> Path:
    """Deep-merge ``changes`` into the config file, creating it if needed."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for key, value in overlay.items():
            if key in out and isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
        return out

    merged = merge(existing, changes)
    # Comments are lost on rewrite; keep the original as a reference so the
    if path.is_file() and not (path.parent / "config.yaml.orig").exists():
        shutil.copyfile(path, path.parent / "config.yaml.orig")
    path.write_text(yaml.safe_dump(merged, sort_keys=False, width=88), encoding="utf-8")
    reset_settings_cache()
    return path


def initialise(force: bool = False) -> Iterator[str]:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and not force:
        yield f"[yellow]config already exists at {path}[/yellow] (use --force to overwrite)"
    else:
        path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        yield f"[green]wrote config[/green] {path}"

    reset_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    from mimir.knowledge.store import get_knowledge_store

    store = get_knowledge_store(settings)
    created = store.ensure_layout()
    yield f"[green]knowledge layout ready[/green] {settings.knowledge.root} ({len(created)} dirs)"

    seeded = _seed_knowledge(settings.knowledge.root)
    if seeded:
        yield f"[green]seeded[/green] {len(seeded)} starter document(s)"
        for name in seeded:
            yield f"  {name}"

    try:
        from mimir.knowledge.index import get_knowledge_index

        stats = get_knowledge_index(settings).reindex(force=True)
        yield f"[green]indexed[/green] {stats.summary()}"
    except Exception as exc:  # noqa: BLE001 - indexing is not fatal to setup
        yield f"[yellow]index build skipped:[/yellow] {exc}"

    yield ""
    yield "Next steps:"
    yield "  1. Fill in knowledge/stable/environments/README.md with your real contexts."
    yield "  2. Add your repository roots under 'repos:' in the config."
    yield "  3. Start a model runtime, then run: mimir doctor"


def _seed_knowledge(root: Path) -> list[str]:
    """Copy seed documents that ship with the package, plus the env template."""
    created: list[str] = []
    target = root / "stable" / "environments" / "README.md"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ENVIRONMENT_TEMPLATE, encoding="utf-8")
        created.append(str(target.relative_to(root)))

    # Seed content that ships in the repository, if MIMIR is running from a
    source_root = Path(__file__).resolve().parents[3] / "knowledge"
    if source_root.is_dir() and source_root != root:
        for source in source_root.rglob("*.md"):
            relative = source.relative_to(source_root)
            destination = root / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            created.append(str(relative))
    return created
