"""Deterministic risk classification (ADR 13.1, 13.2)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from mimir.config import SafetyConfig, get_settings
from mimir.models.command import (
    CommandKind,
    PolicyViolation,
    ProposedCommand,
    RiskAssessment,
    RiskClass,
)

# --------------------------------------------------------------------------

KUBECTL_READ_VERBS = {
    "get",
    "describe",
    "logs",
    "events",
    "top",
    "explain",
    "api-resources",
    "api-versions",
    "version",
    "cluster-info",
    "diff",
    "wait",
}

KUBECTL_R2_VERBS = {"exec", "port-forward", "cp", "attach", "proxy", "debug"}

KUBECTL_R3_VERBS = {
    "rollout",
    "scale",
    "patch",
    "set",
    "annotate",
    "label",
    "cordon",
    "uncordon",
    "expose",
    "autoscale",
    "rollout-restart",
}

KUBECTL_R4_VERBS = {
    "delete",
    "apply",
    "replace",
    "create",
    "drain",
    "taint",
    "edit",
    "certificate",
}

# Subcommand-aware refinements: (verb, subcommand) -> risk.
KUBECTL_SUBCOMMAND_RISK: dict[tuple[str, str], RiskClass] = {
    ("rollout", "status"): RiskClass.R1,
    ("rollout", "history"): RiskClass.R1,
    ("rollout", "restart"): RiskClass.R3,
    ("rollout", "undo"): RiskClass.R3,
    ("rollout", "pause"): RiskClass.R3,
    ("rollout", "resume"): RiskClass.R3,
    ("auth", "can-i"): RiskClass.R1,
    ("config", "current-context"): RiskClass.R1,
    ("config", "view"): RiskClass.R1,
    ("config", "get-contexts"): RiskClass.R1,
    ("config", "use-context"): RiskClass.R2,
    ("config", "set-context"): RiskClass.R3,
    ("set", "resources"): RiskClass.R3,
    ("set", "image"): RiskClass.R3,
    ("set", "env"): RiskClass.R3,
}

# --------------------------------------------------------------------------

READ_ONLY_BINARIES = {
    # Shell and coreutils that only read or print.
    "echo",
    "printf",
    "true",
    "false",
    "sleep",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "pwd",
    "id",
    "whoami",
    "hostname",
    "uptime",
    "free",
    "vmstat",
    "iostat",
    "netstat",
    "ss",
    "lsof",
    "top",
    "column",
    "tr",
    "diff",
    "cmp",
    "md5sum",
    "shasum",
    "sha256sum",
    "base64",
    "xxd",
    "od",
    "strings",
    "nl",
    "rev",
    "paste",
    "join",
    "comm",
    "seq",
    "test",
    "rg",
    "grep",
    "git",
    "cat",
    "head",
    "tail",
    "ls",
    "find",
    "fd",
    "wc",
    "sort",
    "uniq",
    "cut",
    "awk",
    "sed",
    "jq",
    "yq",
    "stat",
    "file",
    "which",
    "env",
    "date",
    "uname",
    "ps",
    "df",
    "du",
    "dig",
    "nslookup",
    "host",
    "curl",
    "wget",
    "ping",
    "traceroute",
    "nc",
    "openssl",
    "helm",
    "kubectx",
    "kubens",
    "docker",
    "sdm",
    "psql",
    "kubectl",
    "python3",
    "tree",
}

# git subcommands that write.
GIT_WRITE_SUBCOMMANDS = {
    "push",
    "commit",
    "merge",
    "rebase",
    "reset",
    "checkout",
    "switch",
    "clean",
    "rm",
    "mv",
    "apply",
    "am",
    "cherry-pick",
    "revert",
    "tag",
    "branch",
    "stash",
    "gc",
    "filter-branch",
    "config",
}

DOCKER_READ_SUBCOMMANDS = {"ps", "images", "inspect", "logs", "top", "stats", "version", "info",
                          "port", "diff", "history", "events"}
DOCKER_R2_SUBCOMMANDS = {"exec", "cp", "attach"}
DOCKER_R3_SUBCOMMANDS = {"restart", "stop", "start", "pause", "unpause", "update", "kill"}
DOCKER_R4_SUBCOMMANDS = {"rm", "rmi", "prune", "system", "volume", "network", "run", "build",
                         "push", "compose"}

HELM_READ_SUBCOMMANDS = {"list", "ls", "status", "get", "history", "show", "template", "search",
                         "diff", "version", "repo", "env"}
HELM_R3_SUBCOMMANDS = {"upgrade", "install", "rollback"}
HELM_R4_SUBCOMMANDS = {"uninstall", "delete"}

SDM_READ_SUBCOMMANDS = {"status", "ls", "list", "resources", "version", "whoami", "audit"}
SDM_R2_SUBCOMMANDS = {"connect", "ssh", "port", "forward"}
SDM_R3_SUBCOMMANDS = {"disconnect", "logout", "login"}

# --------------------------------------------------------------------------

SQL_READ_PREFIXES = ("select", "with", "explain", "show", "table", "values", "\\d", "\\l", "\\dt")
SQL_R3_PREFIXES = ("update", "insert", "delete", "upsert", "merge", "copy")
SQL_R4_PREFIXES = (
    "drop",
    "truncate",
    "alter",
    "create",
    "grant",
    "revoke",
    "vacuum full",
    "reindex",
    "cluster",
)

SQL_STATEMENT_SPLIT = re.compile(r";\s*(?=\S)")

# Shell metacharacters that would let an argv escape into a shell if one were
SHELL_OPERATORS = re.compile(r"(?<!\\)[;&|`$><]|\$\(|\|\||&&")

DANGEROUS_ARG_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("wildcard_all", re.compile(r"^--all$|^-A$"), "targets every resource in scope"),
    ("all_namespaces", re.compile(r"^--all-namespaces$"), "spans every namespace"),
    ("force", re.compile(r"^--force$"), "bypasses safety checks"),
    ("grace_zero", re.compile(r"^--grace-period=0$"), "skips graceful termination"),
    ("cascade", re.compile(r"^--cascade=(foreground|background)$"), "deletes dependents"),
    ("selector_empty", re.compile(r"^(-l|--selector)=?$"), "empty selector matches everything"),
]


@dataclass
class RiskRule:
    """A named rule that can raise (never lower) the risk of a command."""

    name: str
    applies: Callable[[ProposedCommand], bool]
    risk: RiskClass
    reason: str
    reversible: bool = True
    rollback: str | None = None


@dataclass
class ClassificationInput:
    command: ProposedCommand
    config: SafetyConfig = field(default_factory=lambda: get_settings().safety)


def _norm(value: str) -> str:
    return value.strip().lower()


# : Flags that consume the following token as their value.
VALUE_FLAGS: frozenset[str] = frozenset(
    {
        # kubectl
        "-n", "--namespace", "--context", "--cluster", "--kubeconfig", "-o", "--output",
        "-l", "--selector", "--field-selector", "-c", "--container", "--since", "--since-time",
        "--tail", "--user", "--token", "--server", "--as", "--as-group", "--template",
        "-f", "--filename", "-k", "--kustomize", "--type", "-p", "--patch", "--replicas",
        "--request-timeout", "--timeout", "--limit", "--grace-period", "--cascade",
        "--current-replicas", "--image", "--from-literal", "--from-file", "--overrides",
        "--subresource", "--field-manager", "--sort-by", "--chunk-size",
        # docker
        "--format", "--filter", "-e", "--env", "-v", "--volume", "-w", "--workdir",
        "-u", "--name", "--network", "--entrypoint", "--label",
        # git
        "-C", "--git-dir", "--work-tree", "-m", "--message", "--author", "--date",
        # helm
        "--set", "--values", "--version", "--repo",
        # psql
        "-h", "--host", "--port", "-U", "--username", "-d", "--dbname",
        "--command", "--file",
        # sdm
        "--resource",
    }
)


def _positional_args(argv: list[str]) -> list[str]:
    """argv entries after the binary that are neither flags nor flag values."""
    out: list[str] = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            break
        if arg.startswith("-"):
            if "=" not in arg and arg in VALUE_FLAGS:
                skip_next = True
            continue
        out.append(arg)
    return out


def _has_flag(argv: list[str], *names: str) -> bool:
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def _flag_value(argv: list[str], *names: str) -> str | None:
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name and index + 1 < len(argv):
                return argv[index + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------


def classify_kubectl(argv: list[str]) -> tuple[RiskClass, list[str]]:
    reasons: list[str] = []
    positional = _positional_args(argv)
    if not positional:
        return RiskClass.R1, ["no kubectl verb detected"]
    verb = _norm(positional[0])
    sub = _norm(positional[1]) if len(positional) > 1 else ""

    if (verb, sub) in KUBECTL_SUBCOMMAND_RISK:
        risk = KUBECTL_SUBCOMMAND_RISK[(verb, sub)]
        reasons.append(f"kubectl {verb} {sub}")
    elif verb in KUBECTL_READ_VERBS:
        risk = RiskClass.R1
        reasons.append(f"kubectl {verb} is read-only")
    elif verb in KUBECTL_R2_VERBS:
        risk = RiskClass.R2
        reasons.append(f"kubectl {verb} opens a session into a live workload")
    elif verb in KUBECTL_R3_VERBS:
        risk = RiskClass.R3
        reasons.append(f"kubectl {verb} changes live state")
    elif verb in KUBECTL_R4_VERBS:
        risk = RiskClass.R4
        reasons.append(f"kubectl {verb} can affect many resources")
    else:
        risk = RiskClass.R2
        reasons.append(f"unrecognised kubectl verb '{verb}', treated conservatively")

    # kubectl exec running a write command inside the container is not R2.
    if verb == "exec":
        tail = _exec_payload(argv)
        if tail:
            payload_risk, payload_reasons = classify_argv(tail)
            if payload_risk.rank > risk.rank:
                risk = payload_risk
                reasons.extend(f"exec payload: {r}" for r in payload_reasons)

    if verb == "delete":
        if _has_flag(argv, "--all", "-A", "--all-namespaces"):
            reasons.append("delete combined with a wildcard selector")
        risk = RiskClass.R4

    if verb in {"apply", "replace"} and _has_flag(argv, "-f", "--filename", "-k", "--kustomize"):
        reasons.append("applies a manifest that may touch many resources")
        risk = RiskClass.R4

    if verb == "get" and _has_flag(argv, "-o", "--output"):
        out = _flag_value(argv, "-o", "--output") or ""
        if "secret" in " ".join(argv).lower() and out in {"yaml", "json"}:
            risk = max(risk, RiskClass.R2, key=lambda r: r.rank)
            reasons.append("reads secret material")
    if "secrets" in positional or "secret" in positional:
        risk = max(risk, RiskClass.R2, key=lambda r: r.rank)
        reasons.append("reads sensitive operational output")

    return risk, reasons


def _exec_payload(argv: list[str]) -> list[str]:
    """Everything after ``--`` in a kubectl/docker exec invocation."""
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def classify_git(argv: list[str]) -> tuple[RiskClass, list[str]]:
    positional = _positional_args(argv)
    if not positional:
        return RiskClass.R1, ["git with no subcommand"]
    sub = _norm(positional[0])
    if sub in GIT_WRITE_SUBCOMMANDS:
        # git config --get and git branch --list are reads.
        if sub == "config" and _has_flag(argv, "--get", "--list", "-l"):
            return RiskClass.R1, ["git config read"]
        if sub == "branch" and (_has_flag(argv, "--list", "-l") or len(positional) == 1):
            return RiskClass.R1, ["git branch listing"]
        if sub == "stash" and len(positional) > 1 and _norm(positional[1]) in {"list", "show"}:
            return RiskClass.R1, ["git stash read"]
        risk = RiskClass.R4 if sub in {"push", "reset", "clean", "filter-branch"} else RiskClass.R3
        return risk, [f"git {sub} modifies repository state"]
    return RiskClass.R1, [f"git {sub} is read-only"]


def classify_docker(argv: list[str]) -> tuple[RiskClass, list[str]]:
    positional = _positional_args(argv)
    if not positional:
        return RiskClass.R1, ["docker with no subcommand"]
    sub = _norm(positional[0])
    if sub in DOCKER_READ_SUBCOMMANDS:
        return RiskClass.R1, [f"docker {sub} is read-only"]
    if sub in DOCKER_R2_SUBCOMMANDS:
        risk, reasons = RiskClass.R2, [f"docker {sub} enters a running container"]
        tail = _exec_payload(argv) or positional[2:]
        if sub == "exec" and tail:
            payload_risk, payload_reasons = classify_argv(tail)
            if payload_risk.rank > risk.rank:
                return payload_risk, [*reasons, *(f"exec payload: {r}" for r in payload_reasons)]
        return risk, reasons
    if sub in DOCKER_R3_SUBCOMMANDS:
        return RiskClass.R3, [f"docker {sub} changes container lifecycle"]
    if sub in DOCKER_R4_SUBCOMMANDS:
        return RiskClass.R4, [f"docker {sub} is destructive or creates workloads"]
    return RiskClass.R2, [f"unrecognised docker subcommand '{sub}'"]


def classify_helm(argv: list[str]) -> tuple[RiskClass, list[str]]:
    positional = _positional_args(argv)
    if not positional:
        return RiskClass.R1, ["helm with no subcommand"]
    sub = _norm(positional[0])
    if sub in HELM_READ_SUBCOMMANDS:
        return RiskClass.R1, [f"helm {sub} is read-only"]
    if sub in HELM_R3_SUBCOMMANDS:
        if _has_flag(argv, "--dry-run"):
            return RiskClass.R1, [f"helm {sub} --dry-run does not change state"]
        return RiskClass.R3, [f"helm {sub} deploys a release"]
    if sub in HELM_R4_SUBCOMMANDS:
        return RiskClass.R4, [f"helm {sub} removes a release"]
    return RiskClass.R2, [f"unrecognised helm subcommand '{sub}'"]


def classify_sdm(argv: list[str]) -> tuple[RiskClass, list[str]]:
    positional = _positional_args(argv)
    if not positional:
        return RiskClass.R1, ["sdm with no subcommand"]
    sub = _norm(positional[0])
    if sub in SDM_READ_SUBCOMMANDS:
        return RiskClass.R1, [f"sdm {sub} reports local state"]
    if sub in SDM_R2_SUBCOMMANDS:
        return RiskClass.R2, [f"sdm {sub} opens access to a managed resource"]
    if sub in SDM_R3_SUBCOMMANDS:
        return RiskClass.R3, [f"sdm {sub} changes local access state"]
    return RiskClass.R2, [f"unrecognised sdm subcommand '{sub}'"]


def classify_sql(statement: str) -> tuple[RiskClass, list[str]]:
    """Classify a SQL statement or script (ADR 9.4)."""
    text = statement.strip()
    if not text:
        return RiskClass.R0, ["empty statement"]

    statements = [s.strip() for s in SQL_STATEMENT_SPLIT.split(text) if s.strip()]
    worst = RiskClass.R1
    reasons: list[str] = []
    for stmt in statements:
        lowered = re.sub(r"\s+", " ", _norm(stmt))
        # Strip leading comments.
        lowered = re.sub(r"^(--[^\n]*\n|/\*.*?\*/\s*)+", "", lowered, flags=re.S).strip()
        if lowered.startswith(SQL_R4_PREFIXES):
            worst = RiskClass.R4
            reasons.append(f"destructive or schema-changing statement: {lowered[:60]}")
        elif lowered.startswith(SQL_R3_PREFIXES):
            risk = RiskClass.R3
            if not re.search(r"\bwhere\b", lowered) and lowered.startswith(
                ("update", "delete")
            ):
                risk = RiskClass.R4
                reasons.append(f"{lowered.split()[0]} with no WHERE clause affects every row")
            else:
                reasons.append(f"write statement: {lowered[:60]}")
            if risk.rank > worst.rank:
                worst = risk
        elif lowered.startswith(SQL_READ_PREFIXES):
            reasons.append("read-only query")
        else:
            worst = max(worst, RiskClass.R3, key=lambda r: r.rank)
            reasons.append(f"unrecognised statement, treated as a write: {lowered[:60]}")
    if len(statements) > 1:
        reasons.append(f"{len(statements)} statements in one submission")
    return worst, reasons


def classify_psql(argv: list[str]) -> tuple[RiskClass, list[str]]:
    inline = _flag_value(argv, "-c", "--command")
    if inline:
        risk, reasons = classify_sql(inline)
        return max(risk, RiskClass.R2, key=lambda r: r.rank), [
            "opens a database session",
            *reasons,
        ]
    if _has_flag(argv, "-f", "--file"):
        return RiskClass.R4, ["executes a SQL file whose contents are not inspected here"]
    return RiskClass.R2, ["opens an interactive database session"]


BINARY_CLASSIFIERS: dict[str, Callable[[list[str]], tuple[RiskClass, list[str]]]] = {
    "kubectl": classify_kubectl,
    "git": classify_git,
    "docker": classify_docker,
    "podman": classify_docker,
    "helm": classify_helm,
    "sdm": classify_sdm,
    "psql": classify_psql,
}


#: Binaries that destroy or overwrite state wherever they run, including inside
DESTRUCTIVE_BINARIES: dict[str, str] = {
    "rm": "removes files",
    "rmdir": "removes directories",
    "dd": "writes raw blocks",
    "mkfs": "formats a filesystem",
    "shred": "destroys file contents",
    "truncate": "truncates files",
    "chmod": "changes permissions",
    "chown": "changes ownership",
    "kill": "signals processes",
    "pkill": "signals processes by name",
    "killall": "signals processes by name",
    "shutdown": "halts the host",
    "reboot": "restarts the host",
    "halt": "halts the host",
    "systemctl": "changes service state",
    "service": "changes service state",
    "iptables": "changes packet filtering",
    "mount": "changes mounts",
    "umount": "changes mounts",
    "tee": "writes to files",
    "mv": "moves files",
    "cp": "overwrites files",
    "apt": "installs packages",
    "apt-get": "installs packages",
    "yum": "installs packages",
    "apk": "installs packages",
    "pip": "installs packages",
    "npm": "installs packages",
}

#: Mutating shell builtins and redirections that can appear as an exec payload.
_WRITE_HINTS = re.compile(r"(?:^|\s)(?:>|>>|\|\s*tee\b)")


def classify_argv(argv: list[str]) -> tuple[RiskClass, list[str]]:
    """Classify a bare argv without a full command envelope."""
    if not argv:
        return RiskClass.R0, ["empty command"]
    binary = _norm(argv[0].rsplit("/", 1)[-1])

    if binary in DESTRUCTIVE_BINARIES:
        return RiskClass.R4, [f"{binary} {DESTRUCTIVE_BINARIES[binary]}"]

    # A shell invocation hides its real payload in a string argument.
    if binary in {"sh", "bash", "zsh", "ash", "dash"}:
        payload = _flag_value(argv, "-c") or ""
        reasons = [f"{binary} -c wraps another command"]
        if _WRITE_HINTS.search(payload):
            return RiskClass.R3, [*reasons, "payload redirects output into a file"]
        first = payload.strip().split()[0] if payload.strip() else ""
        inner = _norm(first.rsplit("/", 1)[-1])
        if inner in DESTRUCTIVE_BINARIES:
            return RiskClass.R4, [*reasons, f"payload runs {inner}"]
        if inner and inner not in READ_ONLY_BINARIES:
            return RiskClass.R3, [*reasons, f"payload runs unrecognised '{inner}'"]
        return RiskClass.R2, [*reasons, "payload looks read-only but was not fully parsed"]

    classifier = BINARY_CLASSIFIERS.get(binary)
    if classifier:
        return classifier(argv)
    if binary in READ_ONLY_BINARIES:
        return RiskClass.R1, [f"{binary} is a read-only local tool"]
    return RiskClass.R2, [f"unknown binary '{binary}', treated conservatively"]


# --------------------------------------------------------------------------


def _sql_statement(command: ProposedCommand) -> str:
    """The SQL a command will actually execute."""
    if command.stdin:
        return command.stdin
    inline = _flag_value(command.argv, "-c", "--command")
    if inline:
        return inline
    positional = _positional_args(command.argv)
    return " ".join(positional) if positional else ""


class RiskClassifier:
    """Turns a :class:`ProposedCommand` into a :class:`RiskAssessment`."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or get_settings().safety
        self._prod_patterns = [
            re.compile(p, re.I) for p in self.config.production_context_patterns
        ]

    def classify(self, command: ProposedCommand) -> RiskAssessment:
        argv = command.argv
        violations: list[PolicyViolation] = []
        matched: list[str] = []

        if command.kind == CommandKind.SQL:
            statement = _sql_statement(command)
            risk, reasons = classify_sql(statement)
            matched.append("sql")
            risk = max(risk, RiskClass.R2, key=lambda r: r.rank)
        else:
            risk, reasons = classify_argv(argv)
            matched.append(f"binary:{argv[0]}")

        # Shell metacharacters anywhere in argv.
        if not self.config.allow_shell_operators:
            for arg in argv:
                if SHELL_OPERATORS.search(arg):
                    violations.append(
                        PolicyViolation(
                            rule="shell_operator",
                            message=(
                                "argument contains a shell operator and would be "
                                f"ambiguous: {arg!r}"
                            ),
                        )
                    )
                    matched.append("shell_operator")
                    break

        # Denied binaries.
        payload = _exec_payload(argv)
        for label, candidate in (
            ("outer command", argv[0]),
            ("exec payload", payload[0] if payload else ""),
        ):
            if not candidate:
                continue
            binary = candidate.rsplit("/", 1)[-1]
            if binary in self.config.denied_binaries:
                violations.append(
                    PolicyViolation(
                        rule="denied_binary",
                        message=(
                            f"binary '{binary}' in the {label} is on the deny list "
                            "and will never be executed"
                        ),
                    )
                )
                matched.append("denied_binary")

        # Dangerous argument shapes raise the class.
        for name, pattern, why in DANGEROUS_ARG_PATTERNS:
            if any(pattern.match(a) for a in argv):
                reasons.append(f"{name}: {why}")
                matched.append(name)
                if risk.rank >= RiskClass.R3.rank:
                    risk = RiskClass.R4

        # Production target detection.
        production = self._is_production(command)
        if production:
            reasons.append("target looks like production")
            matched.append("production_target")
            if risk.rank >= RiskClass.R3.rank:
                risk = RiskClass.R4

        # Protected namespaces.
        namespace = command.context.namespace
        if namespace and namespace in self.config.protected_namespaces:
            reasons.append(f"namespace '{namespace}' is protected")
            matched.append("protected_namespace")
            if risk.rank >= RiskClass.R2.rank:
                risk = RiskClass.R4

        # Unclear target for a mutating command (ADR 13.2 R4).
        if risk.rank >= RiskClass.R3.rank and not self._has_clear_target(command):
            reasons.append("mutating command with no clearly resolved target")
            matched.append("unclear_target")
            risk = RiskClass.R4

        reversible, rollback = self._reversibility(command, risk)

        requires_approval = self._requires_approval(risk, production)
        forbidden = any(v.fatal for v in violations) or risk.value in self.config.forbidden_risk

        return RiskAssessment(
            risk=risk,
            reasons=reasons,
            requires_approval=requires_approval or forbidden,
            forbidden=forbidden,
            violations=violations,
            matched_rules=matched,
            production_target=production,
            reversible=reversible,
            rollback_hint=rollback,
        )

    # -- helpers ---------------------------------------------------------

    def _requires_approval(self, risk: RiskClass, production: bool) -> bool:
        ceiling = RiskClass(self.config.auto_execute_max_risk)
        if risk.rank > ceiling.rank:
            return True
        if production and risk.rank >= RiskClass.R2.rank:
            return True
        return risk.rank >= RiskClass.R3.rank and self.config.require_approval_for_mutations

    def _is_production(self, command: ProposedCommand) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    command.context.cluster_context,
                    command.context.namespace,
                    command.context.sdm_resource,
                    command.context.database,
                    command.context.host,
                ],
            )
        )
        return any(p.search(haystack) for p in self._prod_patterns)

    @staticmethod
    def _has_clear_target(command: ProposedCommand) -> bool:
        ctx = command.context
        if ctx.targets:
            return True
        if command.kind == CommandKind.KUBECTL:
            return bool(ctx.namespace and (ctx.pod or _positional_args(command.argv)[2:]))
        if command.kind == CommandKind.SQL:
            return bool(ctx.database)
        return bool(ctx.host or ctx.container or ctx.sdm_resource or ctx.repo)

    @staticmethod
    def _reversibility(command: ProposedCommand, risk: RiskClass) -> tuple[bool, str | None]:
        argv_text = " ".join(command.argv).lower()
        if risk.rank <= RiskClass.R2.rank:
            return True, None
        if "rollout restart" in argv_text or ("rollout" in argv_text and "restart" in argv_text):
            target = command.context.targets[0] if command.context.targets else "<deployment>"
            # The target may already be qualified ("deployment/api").
            deploy = target.split("/", 1)[1] if "/" in target else target
            kind = target.split("/", 1)[0] if "/" in target else "deployment"
            ns = command.context.namespace or "<namespace>"
            context = command.context.cluster_context
            prefix = f"kubectl --context {context} " if context else "kubectl "
            return True, f"{prefix}-n {ns} rollout undo {kind}/{deploy}"
        if "scale" in argv_text:
            return True, "re-run scale with the previous replica count"
        if "set resources" in argv_text or ("set" in argv_text and "resources" in argv_text):
            return True, "kubectl rollout undo restores the previous resource block"
        if "delete pod" in argv_text or ("delete" in argv_text and "pod" in argv_text):
            return True, "the controller recreates the pod; verify with kubectl get pods -w"
        if "patch" in argv_text:
            return True, "capture the current manifest first, then re-apply it to revert"
        if any(word in argv_text for word in ("drop", "truncate", "drain", "rmi", "prune")):
            return False, "no automatic rollback; restore from backup"
        if command.kind == CommandKind.SQL:
            return True, "run inside an explicit transaction and verify before COMMIT"
        return True, None


_classifier: RiskClassifier | None = None


def get_classifier(config: SafetyConfig | None = None) -> RiskClassifier:
    global _classifier
    if _classifier is None or config is not None:
        _classifier = RiskClassifier(config)
    return _classifier


def classify(command: ProposedCommand) -> RiskAssessment:
    return get_classifier().classify(command)
