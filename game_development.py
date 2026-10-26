from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ultron.memory import UltronMemory
from ultron.redaction import redact_secrets

from .skill_system import Skill

_ENGINES = {"godot": "Godot", "ursina": "Ursina", "panda3d": "Panda3D"}


def _project_category(project_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project_name.casefold()).strip("-")
    if not slug:
        raise ValueError("Project name must contain letters or numbers.")
    suffix = hashlib.sha256(project_name.casefold().encode("utf-8")).hexdigest()[:8]
    return f"game-{slug[:26]}-{suffix}"


def create_game_plan(
    project_name: str,
    brief: str,
    engine: str = "godot",
) -> str:
    normalized_engine = engine.strip().casefold()
    if normalized_engine not in _ENGINES:
        raise ValueError("Choose Godot, Ursina, or Panda3D.")
    if not project_name.strip() or len(project_name) > 100:
        raise ValueError("Project name must contain 1 to 100 characters.")
    if not brief.strip() or len(brief) > 3000:
        raise ValueError("Game brief must contain 1 to 3,000 characters.")
    display_engine = _ENGINES[normalized_engine]
    return f"""# {project_name.strip()} — Game Development Plan

**Preferred engine:** {display_engine}
**Project brief:** {brief.strip()}

## Design outline

- **Player fantasy:** Clarify what the player should feel capable of doing.
- **Core loop:** Define an action, its immediate feedback/reward, and what motivates the next action.
- **Controls and accessibility:** List keyboard/controller inputs, remapping needs, and accessibility options.
- **Win/fail conditions:** State measurable completion and failure conditions.
- **Visual/audio direction:** Describe a small cohesive style and the minimum readable audio cues.

## First playable prototype

1. Create a minimal project and one test scene.
2. Implement one controllable player action and visible feedback.
3. Add one obstacle or interaction that exercises the core loop.
4. Add a restart/exit path and one basic success condition.
5. Test the loop with placeholder visuals before building content.

## Suggested systems

- Input and player state
- Scene/level flow
- Gameplay rules and feedback
- UI for objective, status, and pause
- Data/configuration separated from scene logic where practical
- Automated checks for deterministic gameplay rules

## Milestones

1. **Pre-production:** agree on a one-sentence pitch, core loop, target platform, and scope.
2. **Prototype:** make the first playable loop with placeholder assets.
3. **Vertical slice:** polish one representative level, UI flow, sound, and save behavior.
4. **Content:** add levels/assets incrementally while tracking performance.
5. **Release checks:** test clean startup, input, resolution, save recovery, and export on target hardware.

## Risks and decisions to record

- Confirm target platforms, resolution, input devices, and minimum hardware.
- Keep the first milestone small enough to test in minutes, not hours.
- Track third-party assets and their licenses before shipping.
- Use source control before edits; review generated code and diffs before applying.

## Validation

No engine was launched and no project was created or tested by generating this plan.
Use ULTRON's confirmation-gated `write_file`, code assistance, and supported project
checks after reviewing the proposed scope and code.
"""


def save_game_project_note(
    project_name: str,
    note: str,
    *,
    workspace: str | Path | None = None,
) -> str:
    category = _project_category(project_name)
    content = note.strip()
    if not content or len(content) > 3500:
        raise ValueError("Project note must contain 1 to 3,500 characters.")
    entry = UltronMemory(str(workspace or Path.cwd())).add_memory(
        f"Project: {project_name.strip()}\n{content}",
        category=category,
    )
    return f"Saved note {entry['id']} for game project '{project_name.strip()}'."


def search_game_project_notes(
    project_name: str,
    query: str,
    *,
    workspace: str | Path | None = None,
) -> str:
    category = _project_category(project_name)
    matches = UltronMemory(str(workspace or Path.cwd())).search_memories(
        query,
        category=category,
    )
    if not matches:
        return f"No matching notes saved for game project '{project_name.strip()}'."
    return "\n".join(
        f"- [{item['id']}] {redact_secrets(item['content'])}" for item in matches
    )


def update_game_project_note(
    project_name: str,
    memory_id: str,
    note: str,
    *,
    workspace: str | Path | None = None,
) -> str:
    category = _project_category(project_name)
    content = note.strip()
    if not content or len(content) > 3500:
        raise ValueError("Project note must contain 1 to 3,500 characters.")
    entry = UltronMemory(str(workspace or Path.cwd())).update_memory(
        memory_id,
        f"Project: {project_name.strip()}\n{content}",
        category=category,
    )
    return f"Updated note {entry['id']} for game project '{project_name.strip()}'."


def create_plan_skill() -> Skill:
    return Skill(
        name="game_development_plan",
        description=(
            "Create an offline game design and staged prototype plan for Godot, "
            "Ursina, or Panda3D without creating files or launching an engine."
        ),
        handler=create_game_plan,
        input_schema={"project_name": str, "brief": str, "engine": str},
        required_inputs=("project_name", "brief"),
        output_description="A reviewable Markdown game-development plan.",
    )


def create_note_save_skill() -> Skill:
    return Skill(
        name="save_game_project_note",
        description="Save an explicit local progress, decision, or asset note for a named game project.",
        handler=save_game_project_note,
        input_schema={"project_name": str, "note": str},
        required_inputs=("project_name", "note"),
        output_description="Confirmation and identifier for a local project note.",
        requires_confirmation=True,
        context_inputs=("workspace",),
    )


def create_note_search_skill() -> Skill:
    return Skill(
        name="search_game_project_notes",
        description="Search local progress, decision, and asset notes for a named game project.",
        handler=search_game_project_notes,
        input_schema={"project_name": str, "query": str},
        required_inputs=("project_name", "query"),
        output_description="Matching local project notes.",
        context_inputs=("workspace",),
    )


def create_note_update_skill() -> Skill:
    return Skill(
        name="update_game_project_note",
        description="Update an existing local note only within the named game project's memory scope.",
        handler=update_game_project_note,
        input_schema={"project_name": str, "memory_id": str, "note": str},
        required_inputs=("project_name", "memory_id", "note"),
        output_description="Confirmation that a project-scoped note was updated.",
        requires_confirmation=True,
        context_inputs=("workspace",),
    )
