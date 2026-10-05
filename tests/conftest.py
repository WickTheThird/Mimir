"""Shared fixtures. Every test runs against a throwaway MIMIR home."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point MIMIR at a temporary home so tests never touch ~/.mimir."""
    home = tmp_path / "mimir-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MIMIR_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    import mimir.config as config

    monkeypatch.setattr(config, "DEFAULT_HOME", home, raising=False)
    config.reset_settings_cache()

    from mimir.safety.approvals import reset_approval_broker
    from mimir.safety.policy import reset_policy_engine
    from mimir.tools.artifacts import reset_artifact_store
    from mimir.tools.exec import reset_executor

    reset_artifact_store()
    reset_executor()
    reset_policy_engine()
    reset_approval_broker()

    settings = config.load_settings(
        {"home": str(home), "knowledge": {"root": str(home / "knowledge")}}
    )
    settings.ensure_directories()
    monkeypatch.setattr(config, "get_settings", lambda: settings)

    yield settings

    config.reset_settings_cache()
    reset_artifact_store()
    reset_executor()
    reset_policy_engine()
    reset_approval_broker()


@pytest.fixture
def settings(isolated_home):
    return isolated_home


@pytest.fixture
def artifacts(settings):
    from mimir.tools.artifacts import ArtifactStore

    return ArtifactStore(settings)


@pytest.fixture
def tool_context(settings, artifacts):
    from mimir.tools.base import ToolContext
    from mimir.tools.exec import CommandExecutor

    return ToolContext(
        settings=settings,
        artifacts=artifacts,
        executor=CommandExecutor(settings, artifacts=artifacts),
    )


@pytest.fixture
def echo_router(settings):
    """A router whose every profile is the deterministic echo runtime."""
    from mimir.config import ModelProfile
    from mimir.llm.router import ModelRouter

    for alias in list(settings.models.profiles):
        settings.models.profiles[alias] = ModelProfile(
            alias=alias, runtime="echo", model="echo"
        )
    return ModelRouter(settings)


@pytest.fixture
def repo_fixture(tmp_path: Path, settings):
    """A small git repository with a realistic timeout bug to investigate."""
    import subprocess

    repo = tmp_path / "billing"
    (repo / "internal" / "auth").mkdir(parents=True)
    (repo / "cmd").mkdir(parents=True)

    # Without go.mod the module prefix is unknown, so "billing/internal/auth"
    (repo / "go.mod").write_text("module billing\n\ngo 1.22\n", encoding="utf-8")

    (repo / "cmd" / "main.go").write_text(
        'package main\n\nimport "billing/internal/auth"\n\n'
        "func main() {\n\tauth.Verify()\n}\n",
        encoding="utf-8",
    )
    (repo / "internal" / "auth" / "client.go").write_text(
        "package auth\n\n"
        "import (\n\t\"context\"\n\t\"time\"\n)\n\n"
        "// DefaultTimeout bounds every call to the auth service.\n"
        "const DefaultTimeout = 2 * time.Second\n\n"
        "func Verify() error {\n"
        "\tctx, cancel := context.WithTimeout(context.Background(), DefaultTimeout)\n"
        "\tdefer cancel()\n"
        "\tif err := call(ctx); err != nil {\n"
        "\t\t// fail open when the dependency is unavailable\n"
        "\t\treturn nil\n"
        "\t}\n"
        "\treturn nil\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "internal" / "auth" / "client_test.go").write_text(
        "package auth\n\nfunc TestVerifyFailsOpen(t *testing.T) {}\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "initial"],
        cwd=repo,
        check=True,
    )

    settings.repos.roots = [tmp_path]
    from mimir.tools.repo import reset_repository_directory

    reset_repository_directory()
    return repo
