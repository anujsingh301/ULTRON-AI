from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ultron.agent import UltronAgent
from ultron.nlu import parse_local_command
from ultron.skill_builder import draft_skill
from ultron.skill_system import SkillExecutionError


VALID_DRAFT = """from ultron.skill_system import Skill

def handler(text: str) -> str:
    return text.upper()

def create_skill():
    return Skill(
        name="greeting",
        description="Uppercase text.",
        handler=handler,
        input_schema={"text": str},
        required_inputs=("text",),
    )
"""


class SkillBuilderTests(unittest.TestCase):
    def test_local_command_is_deterministic_and_confirmation_gated(self) -> None:
        result = parse_local_command(
            "skill draft greeting :: Make a greeting formatter"
        )
        self.assertEqual(result.intent, "execute")
        self.assertEqual(result.steps[0]["skill"], "skill_draft")
        with tempfile.TemporaryDirectory() as directory:
            agent = UltronAgent(base_dir=directory)
            self.assertTrue(agent.skills.get("skill_draft").requires_confirmation)
            prompt = agent.handle_task(
                "skill draft greeting :: Make a greeting formatter"
            )
            self.assertIn("Reply 'confirm'", prompt)
            self.assertFalse(
                list((Path(directory) / ".ultron").glob("skill_drafts/*.py"))
            )

    def test_confirmed_agent_action_writes_but_does_not_load_skill_draft(self) -> None:
        generator = Mock(return_value=VALID_DRAFT)
        with tempfile.TemporaryDirectory() as directory:
            agent = UltronAgent(base_dir=directory)
            with patch("ultron.skill_builder.configured_generator", return_value=generator):
                prompt = agent.handle_task(
                    "skill draft greeting :: Make a greeting formatter"
                )
                self.assertIn("Reply 'confirm'", prompt)
                generator.assert_not_called()
                response = agent.handle_task("confirm")
            self.assertIn("not imported, activated, or executed", response)
            self.assertIsNone(agent.skills.get("greeting"))
            self.assertTrue(
                (Path(directory) / ".ultron" / "skill_drafts" / "greeting.py").is_file()
            )
        generator.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("ultron.skill_builder.configured_generator", return_value=None)
    def test_no_provider_keeps_local_app_and_reports_skill_draft_unavailable(
        self, _configured
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SkillExecutionError, "configured AI provider key"):
                draft_skill("greeting", "Make a greeting formatter", workspace=directory)

    def test_generated_code_is_written_only_as_inert_review_draft(self) -> None:
        generator = Mock(return_value=VALID_DRAFT)
        with tempfile.TemporaryDirectory() as directory:
            with patch("ultron.skill_builder.configured_generator", return_value=generator):
                result = draft_skill(
                    "greeting",
                    "Make a greeting formatter",
                    workspace=directory,
                )
            draft_path = Path(directory) / ".ultron" / "skill_drafts" / "greeting.py"
            self.assertTrue(draft_path.is_file())
            self.assertIn("not imported, activated, or executed", result)
            self.assertEqual(
                draft_path.read_text(encoding="utf-8"), VALID_DRAFT.rstrip() + "\n"
            )
            self.assertEqual(draft_path.stat().st_mode & 0o777, 0o600)
        generator.assert_called_once()

    def test_invalid_generated_code_is_rejected_and_not_saved(self) -> None:
        generator = Mock(return_value="print('bad')\n")
        with tempfile.TemporaryDirectory() as directory:
            with patch("ultron.skill_builder.configured_generator", return_value=generator):
                with self.assertRaisesRegex(SkillExecutionError, "create_skill"):
                    draft_skill("bad_skill", "Do a thing", workspace=directory)
            self.assertFalse((Path(directory) / ".ultron" / "skill_drafts").exists())

    def test_dangerous_imports_calls_secret_values_and_duplicates_are_rejected(self) -> None:
        invalid_drafts = [
            "import subprocess\n" + VALID_DRAFT,
            "import os\n" + VALID_DRAFT.replace(
                "def handler(text: str) -> str:\n    return text.upper()",
                "def handler(text: str) -> str:\n    return eval(text)",
            ),
            VALID_DRAFT.replace("return text.upper()", "return 'ghp_" + "a" * 36 + "'"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, generated in enumerate(invalid_drafts):
                with self.subTest(index=index):
                    with patch(
                        "ultron.skill_builder.configured_generator",
                        return_value=lambda *_args, generated=generated: generated,
                    ):
                        with self.assertRaises(SkillExecutionError):
                            draft_skill(
                                f"draft_{index}",
                                "Do something safe",
                                workspace=directory,
                            )
            with patch("ultron.skill_builder.configured_generator", return_value=lambda *_: VALID_DRAFT):
                draft_skill("duplicate", "A draft", workspace=directory)
                with self.assertRaisesRegex(SkillExecutionError, "already exists"):
                    draft_skill("duplicate", "Another draft", workspace=directory)


if __name__ == "__main__":
    unittest.main()
