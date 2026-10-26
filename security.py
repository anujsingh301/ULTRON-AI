from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .redaction import redact_secrets

_MAX_FILES = 2500
_MAX_FILE_BYTES = 512_000
_MAX_FINDINGS = 500
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".ultron",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "vendor",
    "external-references",
    "site-packages",
}
_SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".xml",
    ".env",
}
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?P<name>
      OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|
      GEMINI_API_KEY|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|GITHUB_TOKEN|
      (?:[A-Z0-9_-]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|CLIENT[_-]?SECRET|
      PASSWORD|SECRET[_-]?KEY|PRIVATE[_-]?KEY))
    )\b\s*[:=]\s*
    (?P<quote>['"]?)(?P<value>[A-Za-z0-9_./+=:-]{8,})
    """
)
_SECRET_TOKEN_PATTERNS = (
    re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
_PLACEHOLDERS = {
    "change-me",
    "changeme",
    "example",
    "placeholder",
    "replace-me",
    "your-api-key",
    "your_key_here",
    "test-key",
}


@dataclass(frozen=True)
class SecurityFinding:
    severity: str
    rule: str
    file: str
    line: int
    evidence: str
    recommendation: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _is_secret_value(value: str) -> bool:
    normalized = value.strip().strip("'\"").casefold()
    if normalized in _PLACEHOLDERS or normalized.startswith(
        ("your_", "your-", "${", "$env:", "os.getenv", "getenv(")
    ):
        return False
    return len(normalized) >= 12


def _secret_findings(text: str, relative_path: str) -> Iterable[SecurityFinding]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _SECRET_TOKEN_PATTERNS):
            yield SecurityFinding(
                "CRITICAL",
                "possible_hardcoded_credential",
                relative_path,
                line_number,
                "Credential-shaped token detected; the value is intentionally hidden.",
                "Revoke/rotate the credential with its provider, remove it from source and history, and load it from a secret manager or environment.",
            )
            continue
        match = _SECRET_ASSIGNMENT.search(line)
        if match and _is_secret_value(match.group("value")):
            yield SecurityFinding(
                "CRITICAL",
                "possible_hardcoded_credential",
                relative_path,
                line_number,
                f"Credential-like value assigned to {match.group('name')}; the value is intentionally hidden.",
                "Revoke/rotate the credential with its provider, remove it from source and history, and load it from a secret manager or environment.",
            )


def _python_findings(text: str, relative_path: str) -> Iterable[SecurityFinding]:
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        yield SecurityFinding(
            "LOW",
            "python_syntax_not_analyzed",
            relative_path,
            exc.lineno or 1,
            "Python syntax could not be parsed; no AST checks were run for this file.",
            "Resolve syntax errors and rerun the local static review.",
        )
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name in {"eval", "exec"}:
                yield SecurityFinding(
                    "MEDIUM",
                    "dynamic_code_execution",
                    relative_path,
                    node.lineno,
                    f"Use of {function_name}() can execute dynamically supplied code.",
                    "Avoid dynamic evaluation; use explicit parsing and allowlisted operations.",
                )
            if function_name in {"load", "loads"} and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id == "pickle":
                    yield SecurityFinding(
                        "HIGH",
                        "unsafe_pickle_deserialization",
                        relative_path,
                        node.lineno,
                        "Pickle deserialization was found; untrusted pickle data can execute code.",
                        "Do not deserialize untrusted pickle data; use a non-executable data format such as JSON.",
                    )
            if function_name in {"request", "get", "post", "put", "delete", "patch"}:
                if any(
                    keyword.arg == "verify"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                    for keyword in node.keywords
                ):
                    yield SecurityFinding(
                        "HIGH",
                        "tls_certificate_verification_disabled",
                        relative_path,
                        node.lineno,
                        "A network request disables TLS certificate verification.",
                        "Remove verify=False and configure a trusted certificate authority where needed.",
                    )
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
            }:
                if any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    yield SecurityFinding(
                        "MEDIUM",
                        "shell_interpolation",
                        relative_path,
                        node.lineno,
                        "A subprocess call enables shell parsing.",
                        "Prefer an argument list with shell=False; validate any user-controlled arguments.",
                    )


def _configuration_findings(
    text: str, relative_path: str
) -> Iterable[SecurityFinding]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if re.search(r"(?i)\bDEBUG\s*[:=]\s*(?:true|1|yes)\b", line):
            yield SecurityFinding(
                "MEDIUM",
                "debug_mode_enabled",
                relative_path,
                line_number,
                "A configuration value enables debug mode.",
                "Disable debug mode outside local development and keep detailed errors out of user-facing responses.",
            )
        if re.search(
            r"(?i)(?:allow_origins|cors_origins|allowed_origins)\s*[:=].{0,100}['\"]\*['\"]",
            line,
        ):
            yield SecurityFinding(
                "MEDIUM",
                "wildcard_cors_origin",
                relative_path,
                line_number,
                "A CORS configuration appears to allow every origin.",
                "Use an explicit allowlist of trusted origins.",
            )
        if relative_path.endswith(("requirements.txt", "requirements-dev.txt")):
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "-")) and not re.search(
                r"(?:===|==|~=|>=|<=|>|<)", stripped
            ):
                yield SecurityFinding(
                    "LOW",
                    "unpinned_python_dependency",
                    relative_path,
                    line_number,
                    "A Python dependency has no version constraint.",
                    "Pin and periodically review dependency versions; use hashes for reproducible high-assurance installs.",
                )
            if re.search(r"(?i)(?:git\+https?://|https?://\S+\.whl)", stripped):
                yield SecurityFinding(
                    "MEDIUM",
                    "remote_dependency_reference",
                    relative_path,
                    line_number,
                    "A dependency is installed from a direct remote URL.",
                    "Prefer a reviewed release from a trusted package index and pin its integrity.",
                )
        if Path(relative_path).name == "package.json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            for section in ("dependencies", "devDependencies", "optionalDependencies"):
                dependencies = data.get(section, {})
                if isinstance(dependencies, dict):
                    for version in dependencies.values():
                        if version == "*":
                            yield SecurityFinding(
                                "LOW",
                                "wildcard_npm_dependency",
                                relative_path,
                                line_number,
                                f"A wildcard version occurs in {section}.",
                                "Use a deliberate version range or lockfile and review dependency updates.",
                            )
                            break
            break


def _candidate_file(path: Path) -> bool:
    name = path.name.casefold()
    return (
        path.suffix.casefold() in _SOURCE_SUFFIXES
        or name == "dockerfile"
        or name.startswith(".env")
        or name in {
            "requirements",
            "requirements-dev",
            "requirements.txt",
            "requirements-dev.txt",
        }
    )


def audit_local_project(
    path: str = ".",
    *,
    authorized: bool = False,
) -> dict[str, object]:
    """Read-only, bounded static review of a user-authorized local project."""
    if authorized is not True:
        raise PermissionError(
            "Confirm that you own this local project or have explicit permission to assess it."
        )
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Local project directory not found: {root}")

    findings: list[SecurityFinding] = []
    scanned_files = 0
    skipped_large_files = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in _SKIP_DIRS
        )
        for name in sorted(filenames):
            candidate = Path(current) / name
            if candidate.is_symlink() or not _candidate_file(candidate):
                continue
            if scanned_files >= _MAX_FILES:
                break
            try:
                if candidate.stat().st_size > _MAX_FILE_BYTES:
                    skipped_large_files += 1
                    continue
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            scanned_files += 1
            relative = candidate.relative_to(root).as_posix()
            findings.extend(_secret_findings(text, relative))
            if candidate.suffix.casefold() == ".py":
                findings.extend(_python_findings(text, relative))
            findings.extend(_configuration_findings(text, relative))
            if len(findings) >= _MAX_FINDINGS:
                findings = findings[:_MAX_FINDINGS]
                break
        if scanned_files >= _MAX_FILES or len(findings) >= _MAX_FINDINGS:
            break

    counts = {
        severity: sum(item.severity == severity for item in findings)
        for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    }
    return {
        "scope": str(root),
        "mode": "read-only static checks; no network access, code execution, installation, or file changes",
        "scanned_files": scanned_files,
        "skipped_large_files": skipped_large_files,
        "findings": [finding.as_dict() for finding in findings],
        "severity_counts": counts,
        "limitations": [
            "Heuristic checks can miss vulnerabilities and can produce false positives.",
            "No dependency advisories or remote vulnerability databases were queried.",
            "Binary documents and files over 512 KB are not analyzed.",
        ],
    }


_LESSONS = {
    "secrets": (
        "Secrets in source control are compromised even after deletion. Revoke and rotate "
        "the credential, remove copies from active code and history, and load replacements "
        "from a secret manager or environment. Never paste raw credentials into reports."
    ),
    "dependencies": (
        "Review dependency provenance, lock files, release history, and advisories before "
        "updating. Prefer maintained releases and reproducible constraints; do not install "
        "untrusted packages just because a scanner recommends them."
    ),
    "least privilege": (
        "Give each process and token only the permissions required for its task. Keep "
        "destructive operations behind explicit approval, scope filesystem access narrowly, "
        "and use short-lived credentials where possible."
    ),
    "input validation": (
        "Validate data at trust boundaries using explicit types, ranges, and allowlists. "
        "Prefer parameterized APIs over building executable commands or queries from strings."
    ),
    "debug mode": (
        "Development diagnostics can reveal paths, configuration, and stack traces. Disable "
        "debug output in deployed environments and return sanitized errors to users."
    ),
    "tls": (
        "TLS certificate verification authenticates the remote endpoint. Do not disable "
        "verification to work around certificate errors; install or configure the correct "
        "trusted CA instead."
    ),
}


def explain_security_topic(topic: str) -> str:
    query = topic.strip().casefold()
    if not query:
        return "Choose a topic such as secrets, dependencies, least privilege, TLS, or input validation."
    matches = [
        (key, explanation)
        for key, explanation in _LESSONS.items()
        if key in query or query in key
    ]
    if matches:
        return "\n\n".join(f"{key.title()}: {text}" for key, text in matches)
    return (
        "I can explain secrets, dependency hygiene, least privilege, input validation, "
        "debug-mode risks, and TLS verification. Ask about one of those topics."
    )


def local_lab_guide(target: str) -> str:
    normalized = target.strip().casefold()
    if normalized not in {"juice shop", "juice-shop", "webgoat", "web goat"}:
        raise ValueError("Choose the intentionally vulnerable Juice Shop or WebGoat lab.")
    application = "OWASP Juice Shop" if "juice" in normalized else "OWASP WebGoat"
    return (
        f"{application} is intentionally vulnerable and is for training only. Use its "
        "official documentation and run it only on a machine/network you control. Keep it "
        "inside an isolated local lab; bind service ports to 127.0.0.1, do not expose them "
        "to LAN or the internet, and stop the lab when finished. Do not point ULTRON's "
        "security review at the lab or any other target unless you explicitly own it or "
        "have permission; unauthorized testing is out of scope. ULTRON does not start "
        "the app, scan it, or execute exercises."
    )
