"""MIMIR skills subsystem (ADR 10, ADR 27 [S1]).

A skill is a versioned directory in the Agent Skills format: ``SKILL.md`` with
YAML frontmatter, plus optional ``references/``, ``scripts/``, and ``tests/``.

The subsystem is split so that progressive disclosure (ADR 10.2) is structural
rather than a convention:

* :mod:`mimir.skills.loader` parses and validates frontmatter into a
  :class:`~mimir.skills.loader.Skill`. The instruction body is not a field on
  that model.
* :mod:`mimir.skills.registry` discovers skills and hands out level-1 metadata
  and a ranked selection.
* :mod:`mimir.skills.runner` loads level 2 and, on request by name, level 3, and
  computes the tool permissions for a skill in code.
* :mod:`mimir.skills.testing` runs the declared test cases.

Permissions are never derived from skill prose. See the trust-boundary note in
:mod:`mimir.skills.loader`.
"""

from __future__ import annotations

from mimir.skills.loader import (
    PLANNED_TOOL_NAMES,
    SKILL_FILENAME,
    Skill,
    SkillExample,
    SkillIO,
    SkillLoadResult,
    SkillResource,
    SkillTestCase,
    SkillValidationError,
    estimate_tokens,
    known_tool_names,
    load_skill_file,
    parse_skill_file,
    read_body,
    split_frontmatter,
)
from mimir.skills.registry import (
    SkillRegistry,
    SkillSelection,
    get_skill_registry,
    reset_skill_registry,
)
from mimir.skills.runner import (
    SKILL_BANNER,
    LoadedResource,
    LoadedSkill,
    SkillRunError,
    SkillRunner,
    ToolPermissions,
)
from mimir.skills.testing import (
    CheckResult,
    SkillTestReport,
    SkillTestResult,
    SkillTranscript,
    TestOutcome,
    run_all_skill_tests,
    run_skill_tests,
    validate_skill_package,
)

__all__ = [
    "PLANNED_TOOL_NAMES",
    "SKILL_BANNER",
    "SKILL_FILENAME",
    "CheckResult",
    "LoadedResource",
    "LoadedSkill",
    "Skill",
    "SkillExample",
    "SkillIO",
    "SkillLoadResult",
    "SkillRegistry",
    "SkillResource",
    "SkillRunError",
    "SkillRunner",
    "SkillSelection",
    "SkillTestCase",
    "SkillTestReport",
    "SkillTestResult",
    "SkillTranscript",
    "SkillValidationError",
    "TestOutcome",
    "ToolPermissions",
    "estimate_tokens",
    "get_skill_registry",
    "known_tool_names",
    "load_skill_file",
    "parse_skill_file",
    "read_body",
    "reset_skill_registry",
    "run_all_skill_tests",
    "run_skill_tests",
    "split_frontmatter",
    "validate_skill_package",
]
