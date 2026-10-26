# ULTRON

ULTRON is a Python assistant foundation with a local agent, optional
OpenAI-compatible AI provider failover, a desktop chat UI, and a modular skill
registry. This public snapshot intentionally omits the hand-written skill
modules from the local working copy. It includes only the AI skill-draft
adapter, so the app starts with a deliberately small set of capabilities.

## Quick start

ULTRON's core has no third-party Python runtime dependencies. Use Python 3.10
or newer:

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

The offline demo and optional desktop app are:

```bash
python3 main.py --demo
python3 main.py --desktop
```

The desktop app requires a Python installation with Tk support.

## Optional Gemini API key

Local task tracking works without an API key. To enable AI chat and
model-based request understanding, create a key in Google AI Studio and export
it in the same terminal session that runs ULTRON:

```zsh
read -s "GEMINI_API_KEY?Paste your Gemini API key: "; export GEMINI_API_KEY; echo
```

The input is hidden. Never add a real key to source code, `.env.example`, a
commit, or chat. `.env.example` lists variable names only and is not loaded
automatically.

The optional standard-library router tries configured providers in this
priority order: Groq, Gemini, Mistral, then OpenRouter. Set one or more
environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
`OPENROUTER_API_KEY`. Model defaults and optional `*_MODEL` overrides are
defined in `ultron/llm_router.py`. Local task tracking does not contact a
provider. AI requests are made only when you submit an AI-powered request;
prompts are sent to the selected provider. The project tests do not verify
live provider access, account quotas, or model availability.

## Included commands

- `status` — show tracked tasks
- `new task <description>` — add a task to local memory
- `ask <question>` — request an AI answer (requires a configured provider)
- `skills` — list available skills
- `list files <path>` and `read file <path>` — inspect the selected workspace
- `search <text> in <path>` — search supported text files
- `write file <path> :: <content>` — create a file after confirmation
- `run command <command>` — run only the allowlisted read-only commands, after
  confirmation
- `skill draft <name> :: <requirements>` — request an inert skill draft
- `exit` — close the terminal app

Task and conversation state is stored locally under `.ultron/`, which is
ignored by Git.

## AI-generated skill drafts

Example:

```text
skill draft greeting :: Make a skill that formats a friendly greeting
```

ULTRON asks for confirmation before contacting an AI provider or writing the
draft. Generated Python is stored in `.ultron/skill_drafts/<name>.py`; it is
not imported, enabled, or executed. Review and test drafts before manually
installing them in the trusted `skills/` directory. The validator checks
syntax, the required `create_skill()` factory, selected unsafe imports/calls,
and credential-like values. It is not a sandbox or a complete security audit.

Only `skills/skill_draft.py` is included from the plugin directory. The base
agent's built-in file and workspace operations remain available; other
hand-written plugin modules from the local working copy are intentionally
excluded from this GitHub snapshot.

## Safety and limitations

- This is a starter project, not an autonomous or unrestricted agent.
- Skill modules are trusted Python code and are imported at startup; never
  install an unreviewed generated draft.
- No real API key, account password, brokerage credential, or other secret
  belongs in this repository.
- The provider router is optional; there is no live-provider test.
- The project does not claim that any provider/model is available or free.

## Tests

Run the included focused tests with:

```bash
python3 -m unittest discover -s tests -v
```
