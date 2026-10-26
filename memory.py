import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .redaction import redact_secrets


class UltronMemory:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = (
            Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
        )
        self.storage_dir = self.base_dir / ".ultron"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.storage_dir / "tasks.json"
        self.skills_path = self.storage_dir / "skills.json"
        self.conversation_path = self.storage_dir / "conversation.json"
        self.executions_path = self.storage_dir / "executions.json"
        self.memories_path = self.storage_dir / "memories.json"
        self.pending_path = self.storage_dir / "pending_plan.json"
        self._lock = threading.RLock()

        if not self.tasks_path.exists():
            self.tasks_path.write_text("[]", encoding="utf-8")
        if not self.skills_path.exists():
            self.skills_path.write_text("{}", encoding="utf-8")
        if not self.conversation_path.exists():
            self.conversation_path.write_text("[]", encoding="utf-8")
        if not self.executions_path.exists():
            self.executions_path.write_text("[]", encoding="utf-8")
        if not self.memories_path.exists():
            self.memories_path.write_text("[]", encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def add_task(self, task: str) -> dict[str, str]:
        with self._lock:
            tasks = self._read_json(self.tasks_path)
            entry = {"task": task, "status": "pending"}
            tasks.append(entry)
            self._write_json(self.tasks_path, tasks)
        return entry

    def list_tasks(self) -> list[dict[str, str]]:
        with self._lock:
            return self._read_json(self.tasks_path)

    def update_task(self, task: dict[str, str], status: str) -> None:
        if status not in {"pending", "running", "completed", "failed", "cancelled"}:
            raise ValueError(f"Unsupported task status: {status}")
        with self._lock:
            tasks = self._read_json(self.tasks_path)
            for index in range(len(tasks) - 1, -1, -1):
                if tasks[index] is task or tasks[index] == task:
                    tasks[index]["status"] = status
                    self._write_json(self.tasks_path, tasks)
                    return
            raise ValueError("Task is no longer present in memory.")

    def register_skill(self, name: str, description: str) -> dict[str, str]:
        with self._lock:
            skills = self._read_json(self.skills_path)
            skills[name] = description
            self._write_json(self.skills_path, skills)
        return {"name": name, "description": description}

    def list_skills(self) -> dict[str, str]:
        with self._lock:
            return self._read_json(self.skills_path)

    def add_memory(self, text: str, *, category: str = "general") -> dict[str, str]:
        content = self._redact_secrets(text).strip()
        normalized_category = category.strip().lower()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        if len(content) > 4000:
            raise ValueError("Memory content cannot exceed 4,000 characters.")
        if not normalized_category or len(normalized_category) > 40:
            raise ValueError("Memory category must contain 1 to 40 characters.")

        entry = {
            "id": uuid.uuid4().hex[:12],
            "category": normalized_category,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            memories = self._read_json(self.memories_path)
            memories.append(entry)
            self._write_json(self.memories_path, memories[-2000:])
        return dict(entry)

    def search_memories(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        terms = set(re.findall(r"[\w-]+", query.casefold()))
        if not terms:
            return []
        if not 1 <= limit <= 50:
            raise ValueError("Memory search limit must be between 1 and 50.")
        normalized_category = category.strip().lower() if category else None
        with self._lock:
            memories = self._read_json(self.memories_path)

        ranked: list[tuple[int, int, dict[str, str]]] = []
        for index, entry in enumerate(memories):
            if normalized_category and entry.get("category") != normalized_category:
                continue
            content = entry.get("content", "")
            searchable = f"{entry.get('category', '')} {content}".casefold()
            matches = sum(1 for term in terms if term in searchable)
            if matches:
                ranked.append((matches, index, entry))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        safe_results: list[dict[str, str]] = []
        for _, _, entry in ranked[:limit]:
            safe_entry = dict(entry)
            safe_entry["content"] = self._redact_secrets(safe_entry.get("content", ""))
            safe_results.append(safe_entry)
        return safe_results

    def delete_memory(self, memory_id: str) -> None:
        with self._lock:
            memories = self._read_json(self.memories_path)
            remaining = [item for item in memories if item.get("id") != memory_id]
            if len(remaining) == len(memories):
                raise ValueError("No saved memory has that ID.")
            self._write_json(self.memories_path, remaining)

    def update_memory(
        self,
        memory_id: str,
        text: str,
        *,
        category: str | None = None,
    ) -> dict[str, str]:
        content = self._redact_secrets(text).strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        if len(content) > 4000:
            raise ValueError("Memory content cannot exceed 4,000 characters.")
        normalized_category = category.strip().lower() if category else None
        if category is not None and (
            not normalized_category or len(normalized_category) > 40
        ):
            raise ValueError("Memory category must contain 1 to 40 characters.")
        with self._lock:
            memories = self._read_json(self.memories_path)
            for entry in memories:
                if entry.get("id") != memory_id:
                    continue
                if normalized_category and entry.get("category") != normalized_category:
                    raise ValueError("No saved memory has that ID in this project.")
                entry["content"] = content
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_json(self.memories_path, memories)
                return dict(entry)
        raise ValueError("No saved memory has that ID.")

    def recent_conversation(self, limit: int = 12) -> list[dict[str, str]]:
        with self._lock:
            turns = self._read_json(self.conversation_path)
        return turns[-limit:] if limit > 0 else []

    def append_conversation(self, role: str, content: str, limit: int = 12) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be user or assistant.")
        safe_content = self._redact_secrets(content)[:4000]
        with self._lock:
            turns = self._read_json(self.conversation_path)
            turns.append({"role": role, "content": safe_content})
            self._write_json(self.conversation_path, turns[-max(1, limit) :])

    def reset_conversation(self) -> None:
        with self._lock:
            self._write_json(self.conversation_path, [])
            self.pending_path.unlink(missing_ok=True)

    def record_execution(
        self,
        task_id: str,
        status: str,
        *,
        intent: str,
        step: str = "",
        error_type: str = "",
    ) -> None:
        if status not in {"running", "waiting", "completed", "failed", "cancelled"}:
            raise ValueError(f"Unsupported execution status: {status}")
        record = {
            "task_id": task_id,
            "status": status,
            "intent": intent,
            "step": step,
            "error_type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            records = self._read_json(self.executions_path)
            records.append(record)
            self._write_json(self.executions_path, records[-500:])

    def execution_failed(self, task_id: str) -> bool:
        with self._lock:
            records = self._read_json(self.executions_path)
        return any(
            record.get("task_id") == task_id and record.get("status") == "failed"
            for record in records
        )

    def set_pending_plan(self, plan: dict[str, Any] | None) -> None:
        with self._lock:
            if plan is None:
                self.pending_path.unlink(missing_ok=True)
                return
            self._write_json(self.pending_path, plan)

    def get_pending_plan(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.pending_path.exists():
                return None
            payload = self._read_json(self.pending_path)
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _redact_secrets(value: str) -> str:
        return redact_secrets(value)
