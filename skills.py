import os
import re
import shlex
import tempfile
from datetime import datetime
from pathlib import Path

from .skill_system import (
    Skill,
    SkillDependencyError,
    SkillDiscoveryError,
    SkillError,
    SkillExecutionError,
    SkillRegistry,
)
from .paths import resolve_workspace_path
from .process import run_bounded_process
from .redaction import redact_secrets


def _workspace_path(path: str, workspace: str) -> Path:
    return resolve_workspace_path(path, workspace)


def list_files(path: str = ".", *, workspace: str = ".") -> str:
    root = _workspace_path(path, workspace)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    items = sorted(p.name for p in root.iterdir())
    return "\n".join(items) if items else "(empty directory)"


def read_file(path: str, *, workspace: str = ".") -> str:
    file_path = _workspace_path(path, workspace)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return file_path.read_text(encoding="utf-8")


def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    *,
    workspace: str = ".",
) -> str:
    file_path = _workspace_path(path, workspace)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and not overwrite:
        raise SkillExecutionError(
            f"File already exists: {file_path}. Request overwrite explicitly to replace it."
        )
    if file_path.exists() and not file_path.is_file():
        raise SkillExecutionError(f"Not a regular file: {file_path}")

    if not overwrite:
        with file_path.open("x", encoding="utf-8") as handle:
            handle.write(content)
        return f"Wrote new file {file_path}"

    backup_root = _workspace_path(".ultron/backups", workspace)
    backup_root.mkdir(parents=True, exist_ok=True)
    relative = file_path.relative_to(Path(workspace).expanduser().resolve())
    backup_name = (
        f"{relative.name}.{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.bak"
    )
    backup_path = backup_root / relative.parent / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        import shutil

        shutil.copy2(file_path, backup_path)
    else:
        backup_path = None

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        os.replace(temporary_path, file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    backup_note = f" Backup saved to {backup_path}." if backup_path else ""
    return f"Replaced {file_path}.{backup_note}"


def _approved_command(command: str, workspace: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError("Command quoting is invalid.") from exc
    if not parts:
        raise ValueError("Command cannot be empty.")

    if parts == ["pwd"]:
        return ["pwd"]
    if parts[0] == "ls":
        args = parts[1:]
        option = ""
        if args and args[0] in {"-l", "-a", "-la", "-al"}:
            option, args = args[0], args[1:]
        if len(args) > 1:
            raise ValueError("Only one workspace path may be listed.")
        path = _workspace_path(args[0] if args else ".", workspace)
        return ["ls", *([option] if option else []), str(path)]
    if parts in (["git", "status"], ["git", "status", "--short"]):
        return parts
    if parts in (["git", "diff", "--check"], ["git", "diff", "--stat"]):
        return parts
    raise ValueError(
        "Command blocked by ULTRON policy. Allowed commands are pwd, ls "
        "inside the selected workspace, git status [--short], and "
        "git diff --check/--stat. Use dedicated confirmed skills for other work."
    )


def _run_bounded(
    argv: list[str],
    workspace: str,
    *,
    timeout: float = 10,
    max_output: int = 4000,
) -> tuple[int, str]:
    return_code, stdout, stderr = run_bounded_process(
        argv,
        cwd=workspace,
        timeout=timeout,
        max_output=max_output,
    )
    output = "\n".join(part for part in (stdout, stderr) if part)
    return return_code, output


def run_command(command: str, *, workspace: str = ".") -> str:
    root = Path(workspace).expanduser().resolve()
    try:
        argv = _approved_command(command, str(root))
    except ValueError as exc:
        raise SkillExecutionError(str(exc)) from exc
    try:
        return_code, output = _run_bounded(argv, str(root))
    except TimeoutError as exc:
        raise SkillExecutionError("Command timed out after 10 seconds.") from exc
    except ValueError as exc:
        raise SkillExecutionError(str(exc)) from exc
    output = redact_secrets(output.strip())
    if return_code:
        raise SkillExecutionError(
            f"Command exited with status {return_code}.\n{output[:4000]}"
        )
    return output or "Command completed without output."


def search_text(
    pattern: str,
    root: str = ".",
    *,
    workspace: str = ".",
) -> str:
    root_path = _workspace_path(root, workspace)
    if not root_path.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if not pattern:
        raise ValueError("Search pattern cannot be empty.")

    matches: list[str] = []
    for file_path in root_path.rglob("*"):
        if (
            file_path.is_file()
            and file_path.resolve().is_relative_to(root_path)
            and file_path.suffix.lower() in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            }
        ):
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if pattern.casefold() in content.casefold():
                matches.append(str(file_path))
    return "\n".join(matches) if matches else "No matches found."


def _make_builtin_skills() -> list[Skill]:
    return [
        Skill(
            name="list_files",
            description="List files and folders inside a directory.",
            handler=list_files,
            input_schema={"path": str},
            output_description="Newline-separated file and folder names.",
            context_inputs=("workspace",),
        ),
        Skill(
            name="read_file",
            description="Read a UTF-8 text file.",
            handler=read_file,
            input_schema={"path": str},
            required_inputs=("path",),
            output_description="Text contents of the file.",
            context_inputs=("workspace",),
        ),
        Skill(
            name="write_file",
            description="Write UTF-8 text to a file.",
            handler=write_file,
            input_schema={"path": str, "content": str, "overwrite": bool},
            required_inputs=("path", "content"),
            output_description="Confirmation containing the written path.",
            requires_confirmation=True,
            context_inputs=("workspace",),
        ),
        Skill(
            name="run_command",
            description=(
                "Run a narrowly allowlisted read-only command inside the selected "
                "workspace; arbitrary shell commands are blocked."
            ),
            handler=run_command,
            input_schema={"command": str},
            required_inputs=("command",),
            output_description="Standard output or standard error.",
            requires_confirmation=True,
            context_inputs=("workspace",),
        ),
        Skill(
            name="search_text",
            description="Search supported text files for a literal substring.",
            handler=search_text,
            input_schema={"pattern": str, "root": str},
            required_inputs=("pattern",),
            output_description="Newline-separated paths containing the text.",
            context_inputs=("workspace",),
        ),
    ]


class UltronSkillRegistry(SkillRegistry):
    """Typed registry plus the original ULTRON commands and their handlers."""

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._context["workspace"] = str(
            Path(workspace or Path.cwd()).expanduser().resolve()
        )
        for skill in _make_builtin_skills():
            self.register(skill)
        if skills_dir is not None:
            self.discover(skills_dir)

    def add(self, name: str, description: str) -> str:
        """Compatibility shim for the original description-only registry."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
            raise ValueError("Skill names must be lowercase snake_case.")
        existing = self.get(name)
        if existing is not None:
            self.register(
                Skill(
                    name=name,
                    description=description,
                    handler=existing.handler,
                    input_schema=existing.input_schema,
                    required_inputs=existing.required_inputs,
                    output_description=existing.output_description,
                    dependencies=existing.dependencies,
                    requires_confirmation=existing.requires_confirmation,
                    context_inputs=existing.context_inputs,
                ),
                replace=True,
            )
        else:
            self.register(
                Skill(
                    name=name,
                    description=description,
                    handler=lambda: f"Skill '{name}' is registered but has no handler.",
                )
            )
        return f"Skill '{name}' registered."


__all__ = [
    "Skill",
    "SkillDependencyError",
    "SkillDiscoveryError",
    "SkillError",
    "SkillExecutionError",
    "SkillRegistry",
    "UltronSkillRegistry",
    "list_files",
    "read_file",
    "run_command",
    "search_text",
    "write_file",
]
