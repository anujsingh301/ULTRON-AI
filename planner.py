from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .nlu import NLUResult
from .skill_system import SkillRegistry


@dataclass(frozen=True)
class PlanStep:
    skill: str
    inputs: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    intent: str
    steps: tuple[PlanStep, ...]
    clarification_question: str | None = None


class PlanningError(ValueError):
    pass


class AgentPlanner:
    def create_plan(self, understanding: NLUResult, registry: SkillRegistry) -> ExecutionPlan:
        if understanding.intent == "clarify":
            return ExecutionPlan(
                intent="clarify",
                steps=(),
                clarification_question=understanding.clarification_question,
            )
        if (
            understanding.intent in {"execute", "create_task"}
            and understanding.confidence < 0.55
        ):
            return ExecutionPlan(
                intent="clarify",
                steps=(),
                clarification_question="I'm not confident I understood the action. Could you rephrase it?",
            )

        plan_steps: list[PlanStep] = []
        for item in understanding.steps:
            skill_name = item.get("skill")
            inputs = item.get("inputs")
            if not isinstance(skill_name, str) or not isinstance(inputs, Mapping):
                raise PlanningError("Each plan step must name a skill and input object.")
            skill = registry.get(skill_name)
            if skill is None:
                return ExecutionPlan(
                    intent="clarify",
                    steps=(),
                    clarification_question=(
                        f"I don't have the '{skill_name}' skill registered. "
                        "Would you like to choose a different action?"
                    ),
                )
            missing = [name for name in skill.required_inputs if name not in inputs]
            if missing:
                return ExecutionPlan(
                    intent="clarify",
                    steps=(),
                    clarification_question=(
                        f"What value should I use for {', '.join(sorted(missing))}?"
                    ),
                )
            invalid = [
                name
                for name, value in inputs.items()
                if name not in skill.input_schema
                or not isinstance(value, skill.input_schema[name])
            ]
            if invalid:
                return ExecutionPlan(
                    intent="clarify",
                    steps=(),
                    clarification_question=(
                        f"I need valid values for {', '.join(sorted(invalid))}."
                    ),
                )
            plan_steps.append(PlanStep(skill=skill.name, inputs=dict(inputs)))
        return ExecutionPlan(understanding.intent, tuple(plan_steps))
