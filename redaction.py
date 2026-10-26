import re


_SECRET_PATTERNS = (
    (
        r"""(?i)(["']?(?:OPENROUTER_API_KEY|[A-Z0-9_-]*(?:api[_ -]?key|secret|access[_ -]?token|token|password|credential))["']?\s*[:=]\s*["']?)[^"'\s,;}]+""",
        r"\1[REDACTED]",
    ),
    (r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]"),
    (r"\bsk-or-v1-[A-Za-z0-9]{20,}\b", "[REDACTED]"),
    (r"\bsk-[A-Za-z0-9_-]{24,}\b", "[REDACTED]"),
    (r"\bAIza[0-9A-Za-z_-]{30,}\b", "[REDACTED]"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", "[REDACTED]"),
    (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED]"),
    (r"\bhf_[A-Za-z0-9]{20,}\b", "[REDACTED]"),
)


def redact_secrets(value: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        value = re.sub(pattern, replacement, value)
    return value
