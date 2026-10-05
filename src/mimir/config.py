"""Layered configuration for MIMIR."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HOME = Path(os.environ.get("MIMIR_HOME", Path.home() / ".mimir"))


class ModelProfile(BaseModel):
    """A single addressable model behind a runtime."""

    alias: str
    runtime: Literal["openai_compat", "ollama", "mlx", "llamacpp", "litellm", "echo"] = (
        "openai_compat"
    )
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:32b"
    api_key: str | None = None
    context_window: int = 32768
    max_output_tokens: int = 4096
    temperature: float = 0.2
    supports_tools: bool = True
    supports_json_schema: bool = True
    request_timeout_s: float = 300.0
    extra_body: dict[str, Any] = Field(default_factory=dict)


class DecisionsConfig(BaseModel):
    """A discriminative model for typed decisions (ADR 18.5)."""

    enabled: bool = False
    backend: Literal["kev", "nimble", "local"] = "kev"
    """kev: the served System One model, calibrated."""
    base_url: str = "http://127.0.0.1:8009"
    model: str = "kev-latest"
    model_path: str = ""
    adapter_path: str = ""
    min_probability: float = 0.7
    """Below this the verdict is treated as no answer."""

    min_margin: float = 0.15
    """Required distance from the runner-up."""


class ModelRouting(BaseModel):
    """Task-class to model-alias routing (ADR 18.4)."""

    default: str = "deep"
    fast_command: str = "fast"
    deep_investigation: str = "deep"
    web_synthesis: str = "deep"
    evidence_verification: str = "deep"
    final_synthesis: str = "deep"
    classification: str = "fast"
    embedding: str = "embed"


class ModelsConfig(BaseModel):
    profiles: dict[str, ModelProfile] = Field(
        default_factory=lambda: {
            "fast": ModelProfile(
                alias="fast",
                runtime="ollama",
                base_url="http://127.0.0.1:11434/v1",
                model="qwen2.5-coder:7b",
                context_window=32768,
            ),
            "deep": ModelProfile(
                alias="deep",
                runtime="ollama",
                base_url="http://127.0.0.1:11434/v1",
                model="qwen2.5:72b",
                context_window=32768,
            ),
            "embed": ModelProfile(
                alias="embed",
                runtime="ollama",
                base_url="http://127.0.0.1:11434/v1",
                model="nomic-embed-text",
                supports_tools=False,
                supports_json_schema=False,
            ),
        }
    )
    routing: ModelRouting = Field(default_factory=ModelRouting)
    # Public alias advertised to Warp through the OpenAI-compatible facade (ADR 16.1).
    public_alias: str = "mimir-local-ops"


class RepositoryEntry(BaseModel):
    name: str
    path: Path
    description: str = ""
    default_branch: str = "main"
    tags: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(os.path.expanduser(str(v))).resolve()


class ReposConfig(BaseModel):
    roots: list[Path] = Field(default_factory=list)
    entries: list[RepositoryEntry] = Field(default_factory=list)
    max_search_results: int = 200
    max_file_bytes: int = 2_000_000
    ignore_globs: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/dist/**",
            "**/build/**",
            "**/__pycache__/**",
            "**/*.min.js",
            "**/*.lock",
        ]
    )

    @field_validator("roots")
    @classmethod
    def _expand_roots(cls, v: list[Path]) -> list[Path]:
        return [Path(os.path.expanduser(str(p))).resolve() for p in v]


class SafetyConfig(BaseModel):
    """Deterministic policy knobs (ADR 13)."""

    # Highest risk class that may execute with no human interaction.
    auto_execute_max_risk: Literal["R0", "R1", "R2", "R3", "R4"] = "R1"
    # Risk classes that are refused outright regardless of approval.
    forbidden_risk: list[str] = Field(default_factory=list)
    require_approval_for_mutations: bool = True
    approval_timeout_s: float = 900.0
    command_timeout_s: float = 120.0
    max_output_bytes: int = 4_000_000
    # Production-looking contexts get an extra confirmation and never auto-execute.
    production_context_patterns: list[str] = Field(
        default_factory=lambda: [r"prod", r"production", r"live", r"prd"]
    )
    protected_namespaces: list[str] = Field(
        default_factory=lambda: ["kube-system", "kube-public", "istio-system"]
    )
    allow_shell_operators: bool = False
    redact_secrets: bool = True
    # Deny-list applied to every argv before anything else runs.
    denied_binaries: list[str] = Field(
        default_factory=lambda: ["rm", "dd", "mkfs", "shutdown", "reboot", "halt", "chown"]
    )


class WebConfig(BaseModel):
    enabled: bool = True
    search_provider: Literal["ddg", "brave", "tavily", "searxng", "none"] = "ddg"
    search_api_key: str | None = None
    searxng_url: str = "http://127.0.0.1:8888"
    max_results: int = 8
    fetch_timeout_s: float = 25.0
    max_document_bytes: int = 3_000_000
    respect_robots: bool = True
    user_agent: str = "MIMIR/0.1 (local operations assistant; +https://localhost)"
    blocked_domains: list[str] = Field(default_factory=list)
    allowed_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    cache_ttl_s: float = 3600.0


class KubernetesConfig(BaseModel):
    enabled: bool = True
    kubectl_path: str = "kubectl"
    kubeconfig: Path | None = None
    default_context: str | None = None
    default_namespace: str | None = None
    allowed_contexts: list[str] = Field(default_factory=list)  # empty means all
    denied_contexts: list[str] = Field(default_factory=list)
    log_tail_lines: int = 2000
    command_timeout_s: float = 90.0
    regions: list[str] = Field(default_factory=list)
    """Region fragments a fan-out searches when the request names none; empty means every context."""
    fanout_timeout_s: float = 15.0
    """Per-context limit for a fan-out probe, so one unreachable cluster cannot hold the answer."""
    skip_unreachable_for_s: float = 1800.0
    """A context that failed this recently is skipped unless named explicitly."""


class SdmConfig(BaseModel):
    enabled: bool = True
    sdm_path: str = "sdm"
    command_timeout_s: float = 120.0
    # Docker/containerd CLI used once connected to a resource.
    container_cli: str = "docker"
    allowed_resource_patterns: list[str] = Field(default_factory=list)
    denied_resource_patterns: list[str] = Field(default_factory=list)


class DatabaseToolConfig(BaseModel):
    enabled: bool = True
    psql_path: str = "psql"
    statement_timeout_ms: int = 30000
    max_rows: int = 500
    command_timeout_s: float = 60.0


class SandboxConfig(BaseModel):
    """Restricted code runner (ADR 9.7)."""

    enabled: bool = True
    interpreter: str = "python3"
    timeout_s: float = 30.0
    max_output_bytes: int = 1_000_000
    memory_limit_mb: int = 1024
    # Credentials scrubbed from the child environment.
    scrub_env_patterns: list[str] = Field(
        default_factory=lambda: [
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "API_KEY",
            "APIKEY",
            "AWS_",
            "GCP_",
            "AZURE_",
            "KUBECONFIG",
            "SDM_",
            "GH_",
            "GITHUB_",
        ]
    )


class PersistenceConfig(BaseModel):
    """ADR 19."""

    url: str = ""  # empty means sqlite under MIMIR_HOME
    echo_sql: bool = False
    retention_days: int | None = None  # None means keep forever (open decision in ADR)


class ApiKeyEntry(BaseModel):
    """A named key stored as its SHA-256; the plaintext is shown once at creation."""

    label: str
    sha256: str
    created_at: float = 0.0
    revoked: bool = False


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8756
    # Tokens accepted on non-loopback requests and by the OpenAI facade (ADR 16).
    api_keys: list[str] = Field(default_factory=list)
    """Legacy plaintext keys. New keys go in ``keys`` as hashes."""
    keys: list[ApiKeyEntry] = Field(default_factory=list)
    max_request_bytes: int = 1_000_000
    allow_loopback_without_auth: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:5173"])
    # When true the OpenAI facade may run the full agent graph.
    facade_agent_mode: bool = False
    facade_rate_limit_per_minute: int = 120
    expose_privileged_routes_publicly: bool = False


class KnowledgeConfig(BaseModel):
    root: Path = Field(default_factory=lambda: DEFAULT_HOME / "knowledge")
    embeddings_enabled: bool = True
    embedding_dimensions: int = 768
    max_snippet_chars: int = 1200
    stale_after_days: int = 180
    top_k: int = 8

    @field_validator("root")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(os.path.expanduser(str(v)))


class SkillsConfig(BaseModel):
    roots: list[Path] = Field(default_factory=list)
    max_auto_selected: int = 3
    allow_scripts: bool = True

    @field_validator("roots")
    @classmethod
    def _expand(cls, v: list[Path]) -> list[Path]:
        return [Path(os.path.expanduser(str(p))) for p in v]


class LspConfig(BaseModel):
    """Language server integration."""

    enabled: bool = True
    timeout_s: float = 20.0
    index_grace_s: float = 2.0
    """Pause after opening a document before querying."""


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = False
    metrics_enabled: bool = True
    log_file: Path | None = None


class GraphConfig(BaseModel):
    max_iterations: int = 24
    max_tool_calls_per_turn: int = 8
    max_specialist_rounds: int = 3
    evidence_budget: int = 60
    recursion_limit: int = 80
    parallel_specialists: bool = True


class Settings(BaseSettings):
    """Root settings object. Access via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="MIMIR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    home: Path = DEFAULT_HOME
    environment: str = "local"

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    decisions: DecisionsConfig = Field(default_factory=DecisionsConfig)
    repos: ReposConfig = Field(default_factory=ReposConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    kubernetes: KubernetesConfig = Field(default_factory=KubernetesConfig)
    sdm: SdmConfig = Field(default_factory=SdmConfig)
    database_tool: DatabaseToolConfig = Field(default_factory=DatabaseToolConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    lsp: LspConfig = Field(default_factory=LspConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)

    @field_validator("home")
    @classmethod
    def _expand_home(cls, v: Path) -> Path:
        return Path(os.path.expanduser(str(v)))

    # -- derived paths ---------------------------------------------------

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def database_url(self) -> str:
        if self.persistence.url:
            return self.persistence.url
        return f"sqlite+pysqlite:///{self.home / 'mimir.db'}"

    @property
    def checkpoint_path(self) -> Path:
        return self.home / "checkpoints.sqlite"

    def skill_roots(self) -> list[Path]:
        roots = list(self.skills.roots)
        default = self.knowledge.root / "skills"
        if default not in roots:
            roots.append(default)
        return roots

    def ensure_directories(self) -> None:
        for path in (
            self.home,
            self.sessions_dir,
            self.artifacts_dir,
            self.cache_dir,
            self.knowledge.root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - operator error path
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping")
    return data


def config_file_candidates() -> list[Path]:
    return [DEFAULT_HOME / "config.yaml", Path.cwd() / "mimir.yaml"]


def load_settings(extra_overrides: dict[str, Any] | None = None) -> Settings:
    merged: dict[str, Any] = {}
    for candidate in config_file_candidates():
        merged = _deep_merge(merged, _load_yaml(candidate))
    if extra_overrides:
        merged = _deep_merge(merged, extra_overrides)
    return Settings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = load_settings()
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """Used by tests and by ``mimir config reload``."""
    clear = getattr(get_settings, "cache_clear", None)
    if clear is not None:
        clear()
