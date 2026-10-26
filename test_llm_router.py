from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from ultron.agent import UltronAgent
from ultron.llm_router import (
    PROVIDERS,
    ProviderError,
    chat,
    configured_generator,
    configured_provider_names,
    reset_cooldowns,
)


class LLMRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_cooldowns()

    def tearDown(self) -> None:
        reset_cooldowns()

    @patch.dict(os.environ, {}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_no_configured_provider_fails_without_network(self, urlopen: Mock) -> None:
        self.assertIsNone(configured_generator())
        with self.assertRaisesRegex(ProviderError, "No AI provider"):
            chat([{"role": "user", "content": "hi"}])
        urlopen.assert_not_called()

    def test_configured_provider_names_follow_router_priority(self) -> None:
        self.assertEqual(
            configured_provider_names(
                {
                    "MISTRAL_API_KEY": "m-key",
                    "GROQ_API_KEY": "g-key",
                    "OPENROUTER_API_KEY": "o-key",
                }
            ),
            ["groq", "mistral", "openrouter"],
        )
        self.assertEqual(PROVIDERS[-1].default_model, "recursive/flint-v1.0:free")

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-token"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_sends_openai_compatible_request_with_configured_model(
        self, urlopen: Mock
    ) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":" hello "}}]}'
        )
        urlopen.return_value.__enter__.return_value = response

        result = chat(
            [{"role": "user", "content": "hi"}],
            environ={
                "GROQ_API_KEY": "test-groq-token",
                "GROQ_MODEL": "test/model",
            },
        )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-groq-token"
        )
        body = json.loads(request.data)
        self.assertEqual(body["model"], "test/model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "hello")

    def test_falls_back_and_cools_down_failing_provider(self) -> None:
        calls: list[str] = []

        def requester(provider, _token, _model, _messages):
            calls.append(provider.name)
            if provider.name == "groq":
                raise ProviderError("rate limited (HTTP 429)")
            return "Gemini response"

        env = {
            "GROQ_API_KEY": "groq-secret",
            "GEMINI_API_KEY": "gemini-secret",
        }
        self.assertEqual(
            chat(
                [{"role": "user", "content": "hello"}],
                environ=env,
                now=100,
                request_func=requester,
            ),
            "Gemini response",
        )
        self.assertEqual(calls, ["groq", "gemini"])

        calls.clear()
        self.assertEqual(
            chat(
                [{"role": "user", "content": "hello"}],
                environ=env,
                now=101,
                request_func=requester,
            ),
            "Gemini response",
        )
        self.assertEqual(calls, ["gemini"])
        calls.clear()
        self.assertEqual(
            chat(
                [{"role": "user", "content": "hello"}],
                environ=env,
                now=161,
                request_func=requester,
            ),
            "Gemini response",
        )
        self.assertEqual(calls, ["groq", "gemini"])

    def test_all_provider_failures_are_sanitized(self) -> None:
        def requester(provider, _token, _model, _messages):
            raise ProviderError("authentication rejected (HTTP 401)")

        with self.assertRaises(ProviderError) as raised:
            chat(
                [{"role": "user", "content": "hello"}],
                environ={
                    "GROQ_API_KEY": "secret-groq-key",
                    "GEMINI_API_KEY": "secret-gemini-key",
                },
                now=500,
                request_func=requester,
                cooldown_seconds=0,
            )
        error = str(raised.exception)
        self.assertIn("groq: authentication rejected", error)
        self.assertIn("gemini: authentication rejected", error)
        self.assertNotIn("secret-groq-key", error)
        self.assertNotIn("secret-gemini-key", error)

    @patch.dict(os.environ, {"GROQ_API_KEY": "secret-key"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_http_failure_reports_status_without_provider_body_or_key(
        self, urlopen: Mock
    ) -> None:
        urlopen.side_effect = HTTPError(
            PROVIDERS[0].base_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"secret-key invalid token"),
        )
        with self.assertRaises(ProviderError) as raised:
            chat([{"role": "user", "content": "hello"}])
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("HTTP 401", str(raised.exception))

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_http_429_and_timeout_are_sanitized(self, urlopen: Mock) -> None:
        urlopen.side_effect = HTTPError(
            PROVIDERS[0].base_url,
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"private response body"),
        )
        with self.assertRaisesRegex(ProviderError, "rate limited"):
            chat([{"role": "user", "content": "hello"}])

        reset_cooldowns()
        urlopen.side_effect = TimeoutError("test-key timed out")
        with self.assertRaisesRegex(
            ProviderError, "connection failed or timed out"
        ) as raised:
            chat([{"role": "user", "content": "hello"}])
        self.assertNotIn("test-key", str(raised.exception))

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_malformed_provider_responses_are_not_exposed(
        self, urlopen: Mock
    ) -> None:
        response = io.BytesIO(b'{"error":"private provider body"}')
        urlopen.return_value.__enter__.return_value = response
        with self.assertRaisesRegex(ProviderError, "unexpected response") as raised:
            chat([{"role": "user", "content": "hello"}])
        self.assertNotIn("private provider body", str(raised.exception))

    def test_invalid_messages_are_rejected_before_request(self) -> None:
        requester = Mock()
        with self.assertRaisesRegex(ProviderError, "valid role/content"):
            chat(
                [{"role": "system", "content": "a", "extra": "bad"}],
                environ={"GROQ_API_KEY": "secret"},
                request_func=requester,
            )
        requester.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-token"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_agent_initialization_and_local_command_do_not_call_provider(
        self, urlopen: Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = UltronAgent(base_dir=directory)
            result = agent.handle_task("status")
        self.assertIn("No tasks tracked", result)
        urlopen.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-token"}, clear=True)
    @patch("ultron.llm_router.urlopen")
    def test_explicit_chat_uses_optional_router(self, urlopen: Mock) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"Hello from Groq."}}]}'
        )
        urlopen.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            agent = UltronAgent(base_dir=directory)
            result = agent.handle_task("ask Say hello")
        self.assertEqual(result, "Hello from Groq.")
        urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
