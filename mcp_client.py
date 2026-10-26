from __future__ import annotations

import json
import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .paths import resolve_workspace_path
from .redaction import redact_secrets
from .skill_system import SkillExecutionError

_MAX_CONFIG_BYTES = 64_000
_MAX_MESSAGE_BYTES = 100_000
_MAX_TOOLS = 200
_PROTOCOL_VERSION = "2025-03-26"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class MCPError(SkillExecutionError):
    pass


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    env_passthrough: tuple[str, ...]
    required_env: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    workspace_path_arguments: Mapping[str, tuple[str, ...]]
    timeout_seconds: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCPServerConfig:
        supported_keys = {
            "name",
            "command",
            "args",
            "env_passthrough",
            "required_env",
            "allowed_tools",
            "workspace_path_arguments",
            "timeout_seconds",
        }
        unknown = set(value) - supported_keys
        if unknown:
            raise MCPError(
                "MCP server configuration contains unsupported field(s): "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        name = value.get("name")
        command = value.get("command")
        args = value.get("args", [])
        env_passthrough = value.get("env_passthrough", [])
        required_env = value.get("required_env", [])
        allowed_tools = value.get("allowed_tools", [])
        path_arguments = value.get("workspace_path_arguments", {})
        timeout = value.get("timeout_seconds", 15)
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise MCPError("MCP server name must use lowercase letters, digits, '_' or '-'.")
        if not isinstance(command, str) or not command or "\x00" in command:
            raise MCPError(f"MCP server '{name}' needs a valid command.")
        for label, entries in (
            ("args", args),
            ("env_passthrough", env_passthrough),
            ("required_env", required_env),
            ("allowed_tools", allowed_tools),
        ):
            if not isinstance(entries, list) or any(
                not isinstance(item, str) or "\x00" in item for item in entries
            ):
                raise MCPError(f"MCP server '{name}' has invalid {label}.")
        if any(not _ENV_NAME.fullmatch(key) for key in env_passthrough + required_env):
            raise MCPError(f"MCP server '{name}' has an invalid environment variable name.")
        if not allowed_tools or any(not item for item in allowed_tools):
            raise MCPError(f"MCP server '{name}' must explicitly allow at least one tool.")
        if any(len(item) > 200 for item in allowed_tools):
            raise MCPError(f"MCP server '{name}' has an overlong tool name.")
        if not isinstance(path_arguments, dict):
            raise MCPError(f"MCP server '{name}' has invalid workspace path rules.")
        normalized_path_arguments: dict[str, tuple[str, ...]] = {}
        for tool_name, arg_names in path_arguments.items():
            if (
                not isinstance(tool_name, str)
                or tool_name not in allowed_tools
                or not isinstance(arg_names, list)
                or any(not isinstance(arg_name, str) or not arg_name for arg_name in arg_names)
            ):
                raise MCPError(f"MCP server '{name}' has invalid workspace path rules.")
            normalized_path_arguments[tool_name] = tuple(arg_names)
        if len(set(allowed_tools)) != len(allowed_tools):
            raise MCPError(f"MCP server '{name}' has duplicate allowed tools.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise MCPError(f"MCP server '{name}' has an invalid timeout.")
        if not 1 <= timeout <= 90:
            raise MCPError(f"MCP server '{name}' timeout must be between 1 and 90 seconds.")
        return cls(
            name=name,
            command=command,
            args=tuple(args),
            env_passthrough=tuple(dict.fromkeys(env_passthrough)),
            required_env=tuple(dict.fromkeys(required_env)),
            allowed_tools=tuple(allowed_tools),
            workspace_path_arguments=normalized_path_arguments,
            timeout_seconds=float(timeout),
        )


class MCPClient:
    """Explicit, bounded MCP stdio client for locally configured servers."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.environ = os.environ if environ is None else environ
        self.config_path = self.workspace / ".ultron" / "mcp_servers.json"

    def server_names(self) -> list[str]:
        return [server.name for server in self._load_servers()]

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        config = self._find_server(server_name)
        with _StdioSession(config, self.workspace, self.environ) as session:
            session.initialize()
            return self._list_tools(session, config)

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> str:
        if not isinstance(arguments, Mapping):
            raise MCPError("MCP tool arguments must be a JSON object.")
        config = self._find_server(server_name)
        with _StdioSession(config, self.workspace, self.environ) as session:
            session.initialize()
            tools = self._list_tools(session, config)
            selected = next((tool for tool in tools if tool["name"] == tool_name), None)
            if selected is None:
                raise MCPError(
                    f"Tool '{tool_name}' is not allowed for MCP server '{server_name}'."
                )
            safe_arguments = _enforce_workspace_paths(
                dict(arguments),
                config.workspace_path_arguments.get(tool_name, ()),
                self.workspace,
            )
            _validate_arguments(safe_arguments, selected.get("inputSchema"))
            try:
                encoded_arguments = json.dumps(
                    safe_arguments, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                raise MCPError("MCP arguments must contain only valid JSON values.") from None
            if len(encoded_arguments) > _MAX_MESSAGE_BYTES:
                raise MCPError("MCP arguments exceed the 100 KB message limit.")
            result = session.request(
                "tools/call",
                {"name": tool_name, "arguments": safe_arguments},
            )
        if not isinstance(result, dict):
            raise MCPError("MCP server returned an invalid tool result.")
        if result.get("isError") is True:
            raise MCPError("MCP tool reported an execution error.")
        rendered = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return redact_secrets(rendered)[:_MAX_MESSAGE_BYTES]

    @classmethod
    def _list_tools(
        cls, session: _StdioSession, config: MCPServerConfig
    ) -> list[dict[str, Any]]:
        all_tools: list[Any] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            page = session.request("tools/list", params)
            if not isinstance(page, dict) or not isinstance(page.get("tools"), list):
                raise MCPError(
                    f"MCP server '{config.name}' returned an invalid tools list."
                )
            all_tools.extend(page["tools"])
            if len(all_tools) > _MAX_TOOLS:
                raise MCPError(f"MCP server '{config.name}' returned too many tools.")
            cursor = page.get("nextCursor")
            if cursor is None:
                break
            if not isinstance(cursor, str) or not cursor or len(cursor) > 2048:
                raise MCPError(f"MCP server '{config.name}' returned an invalid cursor.")
        return cls._filter_tools(config, {"tools": all_tools})

    def _load_servers(self) -> list[MCPServerConfig]:
        try:
            resolved_config = self.config_path.resolve(strict=True)
            if not resolved_config.is_relative_to(self.workspace):
                raise MCPError("MCP configuration must remain inside the workspace.")
            raw = resolved_config.read_bytes()
        except FileNotFoundError:
            raise MCPError(
                "No MCP configuration found. Create .ultron/mcp_servers.json "
                "from mcp_servers.example.json and configure a server explicitly."
            ) from None
        except OSError:
            raise MCPError("Could not read the MCP configuration file.") from None
        if len(raw) > _MAX_CONFIG_BYTES:
            raise MCPError("MCP configuration exceeds the 64 KB limit.")
        try:
            data = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            raise MCPError("MCP configuration is not valid UTF-8 JSON.") from None
        if not isinstance(data, dict) or not isinstance(data.get("servers"), list):
            raise MCPError("MCP configuration must contain a 'servers' array.")
        if len(data["servers"]) > 50:
            raise MCPError("MCP configuration may contain at most 50 servers.")
        servers: list[MCPServerConfig] = []
        names: set[str] = set()
        for entry in data["servers"]:
            if not isinstance(entry, dict):
                raise MCPError("Each MCP server configuration must be an object.")
            config = MCPServerConfig.from_mapping(entry)
            if config.name in names:
                raise MCPError(f"Duplicate MCP server name '{config.name}'.")
            names.add(config.name)
            servers.append(config)
        return servers

    def _find_server(self, name: str) -> MCPServerConfig:
        for server in self._load_servers():
            if server.name == name:
                return server
        raise MCPError(f"MCP server '{name}' is not configured.")

    @staticmethod
    def _filter_tools(
        config: MCPServerConfig, response: Any
    ) -> list[dict[str, Any]]:
        if not isinstance(response, dict) or not isinstance(response.get("tools"), list):
            raise MCPError(f"MCP server '{config.name}' returned an invalid tools list.")
        if len(response["tools"]) > _MAX_TOOLS:
            raise MCPError(f"MCP server '{config.name}' returned too many tools.")
        available: dict[str, dict[str, Any]] = {}
        allowed = set(config.allowed_tools)
        for tool in response["tools"]:
            if not isinstance(tool, dict):
                raise MCPError(f"MCP server '{config.name}' returned an invalid tool.")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise MCPError(f"MCP server '{config.name}' returned a tool without a name.")
            if name not in allowed:
                continue
            if name in available:
                raise MCPError(f"MCP server '{config.name}' returned duplicate tool names.")
            schema = tool.get("inputSchema", {"type": "object", "properties": {}})
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise MCPError(f"MCP tool '{name}' has an unsupported input schema.")
            _validate_schema(schema)
            available[name] = {
                "name": name,
                "description": str(tool.get("description", ""))[:2000],
                "inputSchema": schema,
            }
        return [available[name] for name in config.allowed_tools if name in available]


class _StdioSession:
    def __init__(
        self,
        config: MCPServerConfig,
        workspace: Path,
        environ: Mapping[str, str],
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.environ = environ
        self.process: subprocess.Popen[bytes] | None = None
        self.selector = selectors.DefaultSelector()
        self.buffer = bytearray()
        self.next_id = 1

    def __enter__(self) -> _StdioSession:
        missing = [key for key in self.config.required_env if not self.environ.get(key)]
        if missing:
            raise MCPError(
                f"MCP server '{self.config.name}' requires environment variable(s): "
                + ", ".join(missing)
            )
        env = {
            key: value
            for key, value in self.environ.items()
            if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}
            or key in self.config.env_passthrough
        }
        try:
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                cwd=str(self.workspace),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
            )
        except OSError:
            raise MCPError(
                f"Could not start configured MCP server '{self.config.name}'. "
                "Check its command and installation."
            ) from None
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise MCPError(f"Could not open MCP stdio for '{self.config.name}'.")
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ULTRON", "version": "1.0"},
            },
        )
        if (
            not isinstance(result, dict)
            or result.get("protocolVersion") != _PROTOCOL_VERSION
        ):
            raise MCPError(f"MCP server '{self.config.name}' failed initialization.")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def request(self, method: str, params: dict[str, Any]) -> Any:
        if self.process is None or self.process.stdin is None:
            raise MCPError("MCP process is not running.")
        request_id = self.next_id
        self.next_id += 1
        try:
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError, UnicodeError):
            raise MCPError("MCP request contains invalid JSON values.") from None
        self._send_bytes(payload)
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            message = self._read_message(deadline)
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise MCPError(f"MCP server '{self.config.name}' returned invalid JSON-RPC.")
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                code = error.get("code") if isinstance(error, dict) else "unknown"
                detail = error.get("message", "") if isinstance(error, dict) else ""
                safe_detail = re.sub(
                    r"[\x00-\x1f\x7f]", " ", redact_secrets(str(detail))
                )[:240]
                suffix = f": {safe_detail}" if safe_detail else ""
                raise MCPError(
                    f"MCP server '{self.config.name}' returned error {code}{suffix}."
                )
            if "result" not in message:
                raise MCPError(f"MCP server '{self.config.name}' omitted its result.")
            return message["result"]

    def _send(self, value: dict[str, Any]) -> None:
        self._send_bytes(
            json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
        )

    def _send_bytes(self, payload: bytes) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError("MCP process is not running.")
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise MCPError("MCP request exceeds the 100 KB message limit.")
        descriptor = self.process.stdin.fileno()
        deadline = time.monotonic() + self.config.timeout_seconds
        offset = 0
        os.set_blocking(descriptor, False)
        with selectors.DefaultSelector() as writer:
            writer.register(descriptor, selectors.EVENT_WRITE)
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(
                        f"MCP server '{self.config.name}' timed out while receiving a request."
                    )
                if not writer.select(remaining):
                    continue
                try:
                    written = os.write(descriptor, payload[offset:])
                except BlockingIOError:
                    continue
                except (BrokenPipeError, OSError):
                    raise MCPError(
                        f"MCP server '{self.config.name}' closed its input."
                    ) from None
                if written <= 0:
                    raise MCPError(
                        f"MCP server '{self.config.name}' closed its input."
                    )
                offset += written

    def _read_message(self, deadline: float) -> Any:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                if len(line) > _MAX_MESSAGE_BYTES:
                    raise MCPError("MCP server response exceeds the 100 KB limit.")
                try:
                    return json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    raise MCPError(
                        f"MCP server '{self.config.name}' returned invalid JSON."
                    ) from None
            if len(self.buffer) > _MAX_MESSAGE_BYTES:
                raise MCPError("MCP server response exceeds the 100 KB limit.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(
                    f"MCP server '{self.config.name}' timed out after "
                    f"{self.config.timeout_seconds:g} seconds."
                )
            events = self.selector.select(remaining)
            if not events:
                continue
            assert self.process is not None and self.process.stdout is not None
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                raise MCPError(
                    f"MCP server '{self.config.name}' exited before replying."
                )
            self.buffer.extend(chunk)

    def close(self) -> None:
        self.selector.close()
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if process.stdout is not None:
            process.stdout.close()


def _validate_schema(schema: Any, depth: int = 0) -> None:
    if depth > 32 or not isinstance(schema, dict):
        raise MCPError("MCP tool input schema is invalid or too deeply nested.")
    if "$ref" in schema or "oneOf" in schema or "anyOf" in schema or "allOf" in schema:
        raise MCPError("MCP tool input schema uses unsupported composition.")
    kind = schema.get("type")
    if kind not in {"object", "array", "string", "number", "integer", "boolean", "null"}:
        raise MCPError("MCP tool input schema uses an unsupported type.")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise MCPError("MCP tool input schema has an invalid enum.")
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        bound = schema.get(key)
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 0):
            raise MCPError("MCP tool input schema has an invalid size constraint.")
    for key in ("minimum", "maximum"):
        bound = schema.get(key)
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
            raise MCPError("MCP tool input schema has an invalid numeric constraint.")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list) or any(
            not isinstance(key, str) for key in properties
        ) or any(not isinstance(key, str) for key in required):
            raise MCPError("MCP tool input schema has invalid object properties.")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, (bool, dict)):
            raise MCPError("MCP tool input schema has invalid additionalProperties.")
        for child in properties.values():
            _validate_schema(child, depth + 1)
        if isinstance(additional, dict):
            _validate_schema(additional, depth + 1)
    elif kind == "array" and "items" in schema:
        _validate_schema(schema["items"], depth + 1)


def _enforce_workspace_paths(
    arguments: dict[str, Any],
    path_arguments: tuple[str, ...],
    workspace: Path,
) -> dict[str, Any]:
    for key in path_arguments:
        if key not in arguments:
            continue
        value = arguments[key]
        if not isinstance(value, str):
            raise MCPError(f"Workspace-scoped MCP argument '{key}' must be a path string.")
        try:
            arguments[key] = str(resolve_workspace_path(value, workspace))
        except SkillExecutionError:
            raise MCPError(
                f"MCP path argument '{key}' is outside the approved workspace."
            ) from None
    return arguments


def _validate_arguments(arguments: Mapping[str, Any], schema: Any) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise MCPError("MCP tool has an unsupported input schema.")
    _validate_value(arguments, schema, "arguments")


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    types: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if expected not in types:
        raise MCPError(f"MCP schema has unsupported type at {path}.")
    valid_type = isinstance(value, types[expected])
    if expected in {"number", "integer"} and isinstance(value, bool):
        valid_type = False
    if not valid_type:
        raise MCPError(f"MCP {path} must be {expected}.")
    if "enum" in schema:
        options = schema["enum"]
        if not isinstance(options, list) or value not in options:
            raise MCPError(f"MCP {path} is not an allowed value.")
    if expected == "object":
        if any(not isinstance(key, str) for key in value):
            raise MCPError(f"MCP {path} keys must be strings.")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise MCPError(f"MCP schema is invalid at {path}.")
        missing = [name for name in required if name not in value]
        if missing:
            raise MCPError(f"MCP {path} is missing required field(s): {', '.join(missing)}.")
        additional = schema.get("additionalProperties", True)
        if additional is False:
            extra = set(value) - set(properties)
            if extra:
                raise MCPError(f"MCP {path} has unsupported field(s): {', '.join(sorted(extra))}.")
        for name, item in value.items():
            if name in properties:
                child = properties[name]
                if not isinstance(child, dict):
                    raise MCPError(f"MCP schema is invalid for {path}.{name}.")
                _validate_value(item, child, f"{path}.{name}")
            elif isinstance(additional, dict):
                _validate_value(item, additional, f"{path}.{name}")
        if len(value) < schema.get("minProperties", 0):
            raise MCPError(f"MCP {path} has fewer properties than allowed.")
        if len(value) > schema.get("maxProperties", _MAX_TOOLS):
            raise MCPError(f"MCP {path} has too many properties.")
    elif expected == "array":
        if "items" in schema:
            item_schema = schema["items"]
            if not isinstance(item_schema, dict):
                raise MCPError(f"MCP schema is invalid for {path} items.")
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]")
        if len(value) < schema.get("minItems", 0):
            raise MCPError(f"MCP {path} has fewer items than allowed.")
        if len(value) > schema.get("maxItems", _MAX_TOOLS):
            raise MCPError(f"MCP {path} has too many items.")
    elif expected == "string":
        if len(value) < schema.get("minLength", 0):
            raise MCPError(f"MCP {path} is shorter than allowed.")
        if len(value) > schema.get("maxLength", _MAX_MESSAGE_BYTES):
            raise MCPError(f"MCP {path} exceeds the allowed length.")
    elif expected in {"number", "integer"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise MCPError(f"MCP {path} is below the allowed minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise MCPError(f"MCP {path} is above the allowed maximum.")
