from __future__ import annotations

import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)
SkillHandler = Callable[..., Any]


class SkillError(Exception):
    """Base class for skill registration, validation, and execution errors."""


class SkillDependencyError(SkillError):
    pass


class SkillDiscoveryError(SkillError):
    pass


class SkillExecutionError(SkillError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    handler: SkillHandler
    input_schema: Mapping[str, type] = field(default_factory=dict)
    required_inputs: tuple[str, ...] = ()
    output_description: str = "Skill result"
    dependencies: tuple[str, ...] = ()
    requires_confirmation: bool = False
    context_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("Skill names must be lowercase snake_case.")
        missing_schema = set(self.required_inputs) - set(self.input_schema)
        if missing_schema:
            raise ValueError(
                f"Required inputs missing from schema: {', '.join(sorted(missing_schema))}"
            )


class SkillRegistry:
    """Registry for typed, discoverable, and independently implemented skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._context: dict[str, Any] = {}

    def register(self, skill: Skill, *, replace: bool = False) -> None:
        if skill.name in self._skills and not replace:
            raise SkillError(f"Skill '{skill.name}' is already registered.")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def discover(self, directory: str | Path) -> list[str]:
        """Load trusted Python modules which expose create_skill() -> Skill."""
        root = Path(directory)
        if not root.exists():
            return []
        if not root.is_dir():
            raise SkillDiscoveryError(f"Skill path is not a directory: {root}")

        discovered: list[str] = []
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"_ultron_skill_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise SkillDiscoveryError(f"Could not load skill module: {path.name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                factory = getattr(module, "create_skill", None)
                if not callable(factory):
                    raise SkillDiscoveryError(
                        f"{path.name} must define create_skill()."
                    )
                skill = factory()
                if not isinstance(skill, Skill):
                    raise SkillDiscoveryError(
                        f"{path.name} create_skill() must return a Skill."
                    )
                self.register(skill)
                discovered.append(skill.name)
            except Exception as exc:
                sys.modules.pop(module_name, None)
                if isinstance(exc, SkillDiscoveryError):
                    raise
                raise SkillDiscoveryError(
                    f"Could not register skill module {path.name} ({type(exc).__name__})."
                ) from exc
        return discovered

    def list(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def invoke(self, name: str, inputs: Mapping[str, Any] | None = None) -> Any:
        skill = self.get(name)
        if skill is None:
            raise SkillExecutionError(f"Unknown skill: {name}")

        values = dict(inputs or {})
        missing = [key for key in skill.required_inputs if key not in values]
        if missing:
            raise SkillExecutionError(
                f"Skill '{name}' requires: {', '.join(sorted(missing))}."
            )

        for key, value in values.items():
            expected_type = skill.input_schema.get(key)
            if expected_type is None:
                raise SkillExecutionError(
                    f"Skill '{name}' does not accept input '{key}'."
                )
            if not isinstance(value, expected_type):
                raise SkillExecutionError(
                    f"Input '{key}' for skill '{name}' must be "
                    f"{expected_type.__name__}."
                )

        for key in skill.context_inputs:
            if key not in self._context:
                raise SkillExecutionError(
                    f"Skill '{name}' requires unavailable context '{key}'."
                )
            values[key] = self._context[key]

        for dependency in skill.dependencies:
            try:
                importlib.util.find_spec(dependency)
            except (ImportError, ModuleNotFoundError, ValueError):
                raise SkillDependencyError(
                    f"Skill '{name}' requires unavailable dependency '{dependency}'."
                ) from None
            if importlib.util.find_spec(dependency) is None:
                raise SkillDependencyError(
                    f"Skill '{name}' requires unavailable dependency '{dependency}'."
                )

        try:
            return skill.handler(**values)
        except SkillError:
            raise
        except Exception as exc:
            logger.warning("Skill %s failed (%s).", name, type(exc).__name__)
            raise SkillExecutionError(
                f"Skill '{name}' failed ({type(exc).__name__})."
            ) from exc
