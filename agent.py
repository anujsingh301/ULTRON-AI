from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from .llm_router import ProviderError, configured_generator
from .memory import UltronMemory
from .nlu import NLUParser, NLUResult
from .planner import AgentPlanner, ExecutionPlan, PlanStep
from .redaction import redact_secrets
from .skill_system import SkillError
from .skills import UltronSkillRegistry

logger = logging.getLogger(__name__)

_UNTRUSTED_DATA_SKILLS = {
    "skill_draft",
}


class UltronAgent:
    def __init__(
        self,
        base_dir: str | None = None,
        *,
        skills_dir: str | None = None,
        nlu_parser: NLUParser | None = None,
        text_generator: Callable[..., str] | None = None,
    ):
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir
            else Path.cwd().resolve()
        )
        self.memory = UltronMemory(base_dir=str(self.base_dir))
        self.skills = UltronSkillRegistry(
            skills_dir=skills_dir or str(Path(__file__).parent.parent / "skills"),
            workspace=self.base_dir,
        )
        self.nlu = nlu_parser or NLUParser()
        self.planner = AgentPlanner()
        self.text_generator = (
            text_generator if text_generator is not None else configured_generator()
        )

    def handle_task(self, request: str) -> str:
        text = request.strip()
        if not text:
            return "What would you like me to do?"

        self.memory.append_conversation("user", text)
        if self._has_pending_confirmation():
            response = self._handle_confirmation(text)
            self._remember_response(response)
            return response

        if self._has_pending_slot():
            response = self._handle_slot_followup(text)
            self._remember_response(response)
            return response
        if self._has_pending_clarification():
            self.memory.set_pending_plan(None)

        if text.lower() in {"skills", "list skills", "show skills"}:
            response = self._list_skills()
            self._remember_response(response)
            return response

        understanding = self.nlu.parse(
            text,
            context=self.memory.recent_conversation()[:-1],
            skills=[
                {
                    "name": skill.name,
                    "description": skill.description,
                    "inputs": {
                        name: expected_type.__name__
                        for name, expected_type in skill.input_schema.items()
                    },
                    "required_inputs": list(skill.required_inputs),
                }
                for skill in self.skills.list()
            ],
        )
        self._remember_context_updates(understanding)
        task_id = uuid.uuid4().hex[:12]
        self.memory.record_execution(
            task_id,
            "running",
            intent=understanding.intent,
        )

        try:
            response = self._orchestrate(understanding, task_id)
            final_status = (
                "waiting"
                if understanding.intent == "clarify"
                or self._has_pending_confirmation()
                or self._has_pending_slot()
                or self._has_pending_clarification()
                else "failed"
                if self.memory.execution_failed(task_id)
                else "completed"
            )
            self.memory.record_execution(
                task_id,
                final_status,
                intent=understanding.intent,
            )
        except Exception as exc:
            self.memory.record_execution(
                task_id,
                "failed",
                intent=understanding.intent,
                error_type=type(exc).__name__,
            )
            logger.error("Task %s failed (%s).", task_id, type(exc).__name__)
            response = f"Task failed ({type(exc).__name__}). You can retry or rephrase the request."

        self._remember_response(response)
        return response

    def start_new_chat(self) -> None:
        self.memory.reset_conversation()

    def _orchestrate(self, understanding: NLUResult, task_id: str) -> str:
        if understanding.intent == "clarify":
            pending = understanding.entities.get("pending_slot")
            if isinstance(pending, dict):
                self.memory.set_pending_plan({"kind": "slot", **pending})
            else:
                self.memory.set_pending_plan({"kind": "clarification"})
            question = understanding.clarification_question or "Could you clarify what you mean?"
            return question

        if understanding.intent == "status":
            tasks = self.memory.list_tasks()
            if not tasks:
                return "No tasks tracked yet."
            return "\n".join(f"- {item['status']}: {item['task']}" for item in tasks)

        if understanding.intent == "create_task":
            task_text = str(understanding.entities.get("task", "")).strip()
            if not task_text:
                self.memory.set_pending_plan(
                    {"kind": "slot", "skill": "create_task", "slot": "task"}
                )
                return "What task should I add?"
            if understanding.confidence < 0.55:
                self.memory.set_pending_plan(
                    {"kind": "slot", "skill": "create_task", "slot": "task"}
                )
                return "I'm not confident I understood that task. Could you rephrase it?"
            task = self.memory.add_task(task_text)
            self.memory.update_task(task, "pending")
            return f"Task added: {task_text}"

        if understanding.intent == "chat":
            if understanding.response:
                return understanding.response.strip()
            prompt = str(understanding.entities.get("prompt", "")).strip()
            if not prompt:
                return "Format: ask <your question>"
            if self.text_generator is None:
                return (
                    "AI chat is unavailable because no AI provider is configured. "
                    "Local commands remain available."
                )
            prior_turns = self.memory.recent_conversation()
            if prior_turns and prior_turns[-1]["content"].startswith("[context] "):
                prior_turns = prior_turns[:-1]
            if prior_turns and prior_turns[-1]["role"] == "user":
                prior_turns = prior_turns[:-1]
            if prior_turns:
                prompt = json.dumps(
                    {"latest_message": prompt, "conversation": prior_turns},
                    ensure_ascii=False,
                )
            try:
                return self.text_generator(
                    prompt,
                    "You are ULTRON, a helpful personal AI assistant. "
                    "Use conversation context to answer follow-ups. Do not claim to "
                    "perform actions unless a registered skill did so.",
                )
            except Exception as exc:
                logger.warning("AI response failed (%s).", type(exc).__name__)
                details = (
                    str(exc)
                    if isinstance(exc, ProviderError)
                    else type(exc).__name__
                )
                return (
                    f"AI provider request failed: {details}. "
                    "Try again later or check the configured provider keys."
                )

        plan = self.planner.create_plan(understanding, self.skills)
        if plan.intent == "clarify":
            self.memory.set_pending_plan({"kind": "clarification"})
            return plan.clarification_question or "Which action did you mean?"
        return self._execute_plan(plan, task_id)

    def _execute_plan(self, plan: ExecutionPlan, task_id: str) -> str:
        if not plan.steps:
            return "I couldn't find an action to perform. Try 'ask <question>' for a conversation."

        for step in plan.steps:
            if step.skill == "mcp_call_tool":
                encoded = json.dumps(step.inputs, ensure_ascii=False)
                if (
                    redact_secrets(encoded) != encoded
                    or self._contains_credential_field(step.inputs)
                ):
                    return (
                        "MCP call arguments appear to contain a credential. Do not "
                        "pass credentials as tool arguments; configure an environment "
                        "variable through the server's env_passthrough setting instead."
                    )
            if step.skill in _UNTRUSTED_DATA_SKILLS and self._contains_untrusted_secret(step.inputs):
                return (
                    "This skill does not accept or store credentials. Remove any "
                    "passwords, API keys, tokens, or authorization data and try again."
                )

        if any(self.skills.get(step.skill).requires_confirmation for step in plan.steps):
            pending = {
                "kind": "confirmation",
                "intent": plan.intent,
                "task_id": task_id,
                "steps": [
                    {"skill": step.skill, "inputs": dict(step.inputs)}
                    for step in plan.steps
                ],
            }
            self.memory.set_pending_plan(pending)
            summary = "\n".join(
                f"- {step.skill}: {redact_secrets(', '.join(f'{key}={value}' for key, value in step.inputs.items()))}"
                for step in plan.steps
            )
            if any(
                step.skill in {"security_audit", "security_tool_scan"}
                for step in plan.steps
            ):
                if any(step.skill == "security_tool_scan" for step in plan.steps):
                    return (
                        "This plan will run an installed static-analysis tool "
                        "against local project files only. It will not access "
                        "the network, install or update tools, execute project "
                        "code, or modify the target. Confirm only if you own or "
                        "are explicitly authorized to assess this project:\n"
                        f"{summary}\nReply 'confirm' to continue or 'cancel' to stop."
                    )
                return (
                    "This plan will read local source/config files for static "
                    "security checks only; it will not access the network, execute "
                    "code, install packages, or modify files. Confirm only if you "
                    "own or are explicitly authorized to assess this project:\n"
                    f"{summary}\nReply 'confirm' to continue or 'cancel' to stop."
                )
            return f"This plan includes a system-changing action:\n{summary}\nReply 'confirm' to continue or 'cancel' to stop."

        return self._run_steps(plan, task_id)

    @classmethod
    def _contains_credential_field(cls, value: Any) -> bool:
        sensitive_name = re.compile(
            r"(?:api[_-]?key|access[_-]?token|token|password|secret|credential|authorization)",
            re.IGNORECASE,
        )
        if isinstance(value, dict):
            return any(
                sensitive_name.search(str(key))
                or cls._contains_credential_field(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(cls._contains_credential_field(item) for item in value)
        return False

    @classmethod
    def _contains_untrusted_secret(cls, value: Any) -> bool:
        if isinstance(value, str):
            if redact_secrets(value) != value:
                return True
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    return False
                return cls._contains_credential_field(decoded) or cls._contains_untrusted_secret(decoded)
            return False
        if isinstance(value, dict):
            return any(
                cls._contains_credential_field({str(key): item})
                or cls._contains_untrusted_secret(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(cls._contains_untrusted_secret(item) for item in value)
        return False

    def _run_steps(self, plan: ExecutionPlan, task_id: str) -> str:
        results: list[str] = []
        raw_results: list[str] = []
        for index, step in enumerate(plan.steps, start=1):
            self.memory.record_execution(
                task_id,
                "running",
                intent=plan.intent,
                step=step.skill,
            )
            try:
                resolved_inputs = self._resolve_step_inputs(step.inputs, raw_results)
                result = self.skills.invoke(step.skill, resolved_inputs)
            except SkillError as exc:
                self.memory.record_execution(
                    task_id,
                    "failed",
                    intent=plan.intent,
                    step=step.skill,
                    error_type=type(exc).__name__,
                )
                return f"Step {index} ({step.skill}) failed: {exc}"
            except Exception as exc:
                self.memory.record_execution(
                    task_id,
                    "failed",
                    intent=plan.intent,
                    step=step.skill,
                    error_type=type(exc).__name__,
                )
                logger.error("Skill %s failed (%s).", step.skill, type(exc).__name__)
                return f"Step {index} ({step.skill}) failed ({type(exc).__name__})."
            result_text = str(result)
            raw_results.append(result_text[:4000])
            results.append(f"{index}. {step.skill}: {result_text}")

        self.memory.record_execution(task_id, "completed", intent=plan.intent)
        return "\n".join(results)

    @classmethod
    def _resolve_step_inputs(
        cls,
        inputs: Any,
        prior_results: list[str],
    ) -> Any:
        if isinstance(inputs, str):
            def replace_result(match: re.Match[str]) -> str:
                position = int(match.group(1)) - 1
                if position < 0 or position >= len(prior_results):
                    raise ValueError("A plan referenced a step result that does not exist.")
                return prior_results[position]

            return re.sub(r"\{\{result_(\d+)\}\}", replace_result, inputs)
        if isinstance(inputs, list):
            return [cls._resolve_step_inputs(value, prior_results) for value in inputs]
        if isinstance(inputs, dict):
            return {
                key: cls._resolve_step_inputs(value, prior_results)
                for key, value in inputs.items()
            }
        return inputs

    def _handle_confirmation(self, text: str) -> str:
        pending = self.memory.get_pending_plan()
        reply = text.strip().casefold()
        if not pending:
            return "There is no action waiting for confirmation."
        if reply in {"cancel", "no", "no, cancel", "cancel task"}:
            self.memory.set_pending_plan(None)
            return "Action cancelled."
        if pending.get("kind") == "confirmation_second":
            if reply != "approve":
                return "This action needs the exact reply 'approve', or reply 'cancel'."
            self.memory.set_pending_plan(None)
            return self._run_confirmed_steps(pending)
        if reply not in {"confirm", "yes", "yes, confirm", "confirm task"}:
            return "Reply 'confirm' to continue or 'cancel' to stop."
        if pending.get("kind") != "confirmation":
            self.memory.set_pending_plan(None)
            return "There is no action waiting for confirmation."
        if self._requires_additional_approval(pending.get("steps", [])):
            pending["kind"] = "confirmation_second"
            self.memory.set_pending_plan(pending)
            mcp_call = any(
                step.get("skill") == "mcp_call_tool"
                for step in pending.get("steps", [])
            )
            if mcp_call:
                return (
                    "This MCP tool may access external services or change data. "
                    "A separate approval is required for this exact call only. "
                    "Reply 'approve' to proceed or 'cancel' to stop."
                )
            executes_code = any(
                step.get("skill") == "run_project_check"
                for step in pending.get("steps", [])
            )
            if executes_code:
                return (
                    "This project check can execute project code. A separate "
                    "approval is required for this run only. Reply 'approve' to "
                    "proceed or 'cancel' to stop."
                )
            return (
                "This operation can replace or move data. A separate approval is "
                "required for this action only. Reply 'approve' to proceed or "
                "'cancel' to stop."
            )
        self.memory.set_pending_plan(None)
        return self._run_confirmed_steps(pending)

    @staticmethod
    def _requires_additional_approval(steps: list[dict[str, Any]]) -> bool:
        for step in steps:
            skill = step.get("skill")
            inputs = step.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if skill == "write_file" and inputs.get("overwrite") is True:
                return True
            if skill == "run_project_check":
                return True
            if skill == "mcp_call_tool":
                return True
            action = inputs.get("action")
            if (
                skill == "manage_file"
                and isinstance(action, str)
                and action.casefold() in {"move", "trash"}
            ):
                return True
        return False

    def _run_confirmed_steps(self, pending: dict[str, Any]) -> str:
        understanding = NLUResult(
            intent=str(pending.get("intent", "execute")),
            steps=tuple(pending.get("steps", [])),
            confidence=1.0,
        )
        try:
            plan = self.planner.create_plan(understanding, self.skills)
            if plan.intent == "clarify":
                return plan.clarification_question or "That action is no longer available."
            return self._run_steps(
                plan,
                str(pending.get("task_id", uuid.uuid4().hex[:12])),
            )
        except Exception as exc:
            logger.error("Confirmed task failed (%s).", type(exc).__name__)
            return f"Confirmed task failed ({type(exc).__name__})."

    def _has_pending_confirmation(self) -> bool:
        pending = self.memory.get_pending_plan()
        return bool(
            pending
            and pending.get("kind") in {"confirmation", "confirmation_second"}
        )

    def _has_pending_slot(self) -> bool:
        pending = self.memory.get_pending_plan()
        return bool(pending and pending.get("kind") == "slot")

    def _has_pending_clarification(self) -> bool:
        pending = self.memory.get_pending_plan()
        return bool(pending and pending.get("kind") == "clarification")

    def _handle_slot_followup(self, text: str) -> str:
        pending = self.memory.get_pending_plan() or {}
        skill = str(pending.get("skill", ""))
        slot = str(pending.get("slot", ""))
        self.memory.set_pending_plan(None)
        if skill == "read_file" and slot == "path":
            result = self.skills.invoke(skill, {"path": text})
            return str(result)
        if skill == "run_command" and slot == "command":
            return self._execute_plan(
                ExecutionPlan(
                    intent="execute",
                    steps=(
                        PlanStep(skill=skill, inputs={"command": text}),
                    ),
                ),
                uuid.uuid4().hex[:12],
            )
        if skill == "search_text" and slot == "pattern":
            return str(self.skills.invoke(skill, {"pattern": text, "root": "."}))
        if skill == "write_file" and slot == "path_and_content":
            parts = text.split("::", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1]:
                self.memory.set_pending_plan(pending)
                return "Please provide the path and content as: <path> :: <content>"
            return self._execute_plan(
                ExecutionPlan(
                    intent="execute",
                    steps=(
                        PlanStep(
                            skill=skill,
                            inputs={"path": parts[0].strip(), "content": parts[1]},
                        ),
                    ),
                ),
                uuid.uuid4().hex[:12],
            )
        if skill == "create_task" and slot == "task":
            task = self.memory.add_task(text)
            self.memory.update_task(task, "pending")
            return f"Task added: {text}"
        return "I couldn't connect that reply to the pending question. Please start again."

    def _remember_context_updates(self, understanding: NLUResult) -> None:
        updates = understanding.context_updates
        if updates:
            self.memory.append_conversation(
                "assistant",
                f"[context] {json.dumps(updates, ensure_ascii=False)}",
            )

    def _remember_response(self, response: str) -> None:
        self.memory.append_conversation("assistant", str(response))

    def _list_skills(self) -> str:
        skills = self.skills.list()
        return "\n".join(
            f"- {skill.name}: {skill.description}" for skill in skills
        ) or "No skills registered."

    def _parse(self, request: str) -> dict[str, str]:
        """Backward-compatible parser facade for the original command API."""
        result = self.nlu.parse(
            request,
            context=[],
            skills=[
                {"name": skill.name, "description": skill.description}
                for skill in self.skills.list()
            ],
        )
        if result.intent == "execute" and result.steps:
            step = result.steps[0]
            return {"skill": str(step["skill"]), **{k: str(v) for k, v in step["inputs"].items()}}
        return {"skill": result.intent, "arg": request}
