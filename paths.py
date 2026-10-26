from __future__ import annotations

from pathlib import Path

from .skill_system import SkillExecutionError

_APPROVED_USER_FOLDERS = (
    "Desktop",
    "Downloads",
    "Documents",
    "Pictures",
    "Music",
    "Movies",
)


def resolve_workspace_path(
    value: str | Path,
    workspace: str | Path,
    *,
    allow_user_folders: bool = False,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    approved_roots = [root]
    if allow_user_folders:
        approved_roots.extend(
            (Path.home() / folder).resolve() for folder in _APPROVED_USER_FOLDERS
        )
    if not any(resolved.is_relative_to(approved) for approved in approved_roots):
        raise SkillExecutionError("Path is outside the selected ULTRON workspace.")
    return resolved
