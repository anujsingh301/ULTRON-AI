from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .process import run_bounded_process
from .redaction import redact_secrets

_TIMEOUT_SECONDS = 180
_MAX_OUTPUT_BYTES = 8_000_000
_MAX_FINDINGS = 500
_TOOLS = {"bandit", "semgrep", "trivy"}


def _project_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Local project directory not found: {root}")
    return root


def _tool_path(tool: str) -> str:
    if tool not in _TOOLS:
        raise ValueError(f"Unsupported local scanner: {tool}")
    executable = shutil.which(tool)
    if executable is None:
        raise FileNotFoundError(
            f"{tool} is not installed or not available on PATH; ULTRON will not install tools automatically."
        )
    return executable


def _run_json(
    command: list[str],
    *,
    cwd: Path,
    accepted_exit_codes: set[int],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    process_env = os.environ.copy()
    process_env["NO_COLOR"] = "1"
    if env:
        process_env.update(env)
    try:
        return_code, stdout, stderr = run_bounded_process(
            command,
            cwd=cwd,
            timeout=_TIMEOUT_SECONDS,
            max_output=_MAX_OUTPUT_BYTES,
            env=process_env,
        )
    except TimeoutError as exc:
        raise TimeoutError("Security scanner exceeded the 180 second time limit.") from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not start the security scanner ({type(exc).__name__})."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "Security scanner output exceeded the 8 MB safety limit."
        ) from exc
    if return_code not in accepted_exit_codes:
        detail = redact_secrets(stderr.strip())[:500]
        suffix = f" Details: {detail}" if detail else ""
        raise RuntimeError(
            f"Security scanner failed with exit code {return_code}.{suffix}"
        )
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        detail = redact_secrets(stderr.strip())[:500]
        suffix = f" Details: {detail}" if detail else ""
        raise RuntimeError(
            f"Security scanner returned invalid JSON.{suffix}"
        ) from exc
    if not isinstance(report, dict):
        raise RuntimeError("Security scanner returned an unexpected report shape.")
    return report


def _relative_file(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            return path.name
    return Path(value).as_posix()


def _bandit_findings(report: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    results = report.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Bandit returned an invalid findings list.")
    findings = []
    for result in results[:_MAX_FINDINGS]:
        if not isinstance(result, dict):
            continue
        findings.append(
            {
                "severity": str(result.get("issue_severity", "UNKNOWN")).upper(),
                "rule": str(result.get("test_id", "bandit.unknown")),
                "file": _relative_file(result.get("filename"), root),
                "line": result.get("line_number", 0),
                "evidence": redact_secrets(str(result.get("issue_text", ""))),
                "remediation": "Review the flagged code path and apply the tool's documented mitigation.",
            }
        )
    return findings


def _semgrep_findings(report: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    results = report.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Semgrep returned an invalid findings list.")
    findings = []
    for result in results[:_MAX_FINDINGS]:
        if not isinstance(result, dict):
            continue
        extra = result.get("extra")
        extra = extra if isinstance(extra, dict) else {}
        start = result.get("start")
        start = start if isinstance(start, dict) else {}
        findings.append(
            {
                "severity": str(extra.get("severity", "UNKNOWN")).upper(),
                "rule": str(result.get("check_id", "semgrep.unknown")),
                "file": _relative_file(result.get("path"), root),
                "line": start.get("line", 0),
                "evidence": redact_secrets(str(extra.get("message", ""))),
                "remediation": "Review the local rule's guidance and validate the finding in context.",
            }
        )
    return findings


def _trivy_findings(report: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    results = report.get("Results", [])
    if not isinstance(results, list):
        raise RuntimeError("Trivy returned an invalid results list.")
    findings: list[dict[str, Any]] = []
    for target in results:
        if not isinstance(target, dict):
            continue
        file_name = _relative_file(target.get("Target"), root)
        for section in ("Vulnerabilities", "Misconfigurations", "Secrets", "Licenses"):
            records = target.get(section, [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                rule = record.get("VulnerabilityID") or record.get("ID") or record.get("RuleID")
                title = record.get("Title") or record.get("Message") or record.get("PkgName") or section
                recommendation = record.get("Resolution") or record.get("FixedVersion")
                line = record.get("StartLine", 0)
                findings.append(
                    {
                        "severity": str(record.get("Severity", "UNKNOWN")).upper(),
                        "rule": str(rule or f"trivy.{section.casefold()}"),
                        "file": file_name,
                        "line": line if isinstance(line, int) else 0,
                        "evidence": redact_secrets(str(title)),
                        "remediation": redact_secrets(str(recommendation))
                        if recommendation
                        else "Review the Trivy advisory and choose an appropriate remediation.",
                    }
                )
                if len(findings) >= _MAX_FINDINGS:
                    return findings
    return findings


def run_local_security_tool(
    tool: str,
    path: str = ".",
    *,
    authorized: bool = False,
) -> str:
    """Run an optional local static scanner without installing or contacting services."""
    if authorized is not True:
        raise PermissionError(
            "Confirm that you own this local project or have explicit permission to assess it."
        )
    root = _project_root(path)
    executable = _tool_path(tool)

    if tool == "bandit":
        command = [executable, "-r", str(root), "-f", "json"]
        raw = _run_json(command, cwd=root, accepted_exit_codes={0, 1})
        findings = _bandit_findings(raw, root)
    elif tool == "semgrep":
        rules = Path(__file__).with_name("security_rules.yml")
        command = [
            executable,
            "scan",
            "--json",
            "--metrics=off",
            "--config",
            str(rules),
            str(root),
        ]
        raw = _run_json(command, cwd=root, accepted_exit_codes={0})
        findings = _semgrep_findings(raw, root)
    else:
        with tempfile.TemporaryDirectory(prefix="ultron-trivy-") as cache:
            command = [
                executable,
                "fs",
                "--format",
                "json",
                "--scanners",
                "vuln,misconfig,secret",
                "--skip-db-update",
                "--skip-check-update",
                "--offline-scan",
                "--no-progress",
                "--cache-dir",
                cache,
                str(root),
            ]
            raw = _run_json(command, cwd=root, accepted_exit_codes={0})
        findings = _trivy_findings(raw, root)

    return json.dumps(
        {
            "tool": tool,
            "target": str(root),
            "mode": "local read-only scan; no installation or network access",
            "findings": findings,
            "finding_count": len(findings),
            "limitations": _limitations(tool),
        },
        indent=2,
        ensure_ascii=False,
    )


def _limitations(tool: str) -> list[str]:
    limitations = [
        "Findings are heuristic and require human review; absence of findings is not proof of security.",
        "ULTRON does not install/update tools, modify the target, or execute target project code.",
        "Reports omit source snippets and secret match values; combined scanner output is stopped at 8 MB.",
    ]
    if tool == "bandit":
        limitations.append("Bandit analyzes Python AST patterns; it is not a general-purpose or dependency vulnerability scanner.")
    elif tool == "semgrep":
        limitations.append("Semgrep uses only ULTRON's bundled local Python rules; registry rules, metrics, and local project builds are not used.")
    else:
        limitations.append(
            "Trivy is offline-only: vulnerability/misconfiguration results require usable local databases and may be unavailable or stale."
        )
    return limitations
