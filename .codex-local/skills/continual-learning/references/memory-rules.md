# Memory Rules

## Keep

- Recurring user preferences about language, tone, brevity, or collaboration style.
- Stable workspace facts such as main language, tooling, directory conventions, or deployment constraints.
- Broad engineering rules the user states as durable defaults.

## Exclude

- Passwords, tokens, cookies, API keys, personal data, or approval codes.
- Temporary troubleshooting state, transient errors, or one-off task instructions.
- Conversation-specific details that will not help in future sessions.
- Speculative inferences that are not well supported by repeated evidence.

## Conflict Resolution

- Prefer newer bullets over older bullets when they clearly conflict.
- Prefer explicit user statements over inferred patterns.
- Prefer broader, more reusable wording over narrow wording.
- If a candidate bullet can only be understood with historical context, do not keep it.

## Good Bullets

- Default to Chinese unless English is clearly more accurate.
- Prefer simple, direct, maintainable implementations.
- This workspace primarily uses Python.

## Bad Bullets

- Fix the GPU driver issue from March 17.
- Use branch feature/memory-loop.
- Password is redacted.

## Codex Transcript Notes

- Prefer `~/.codex/sessions/` because it contains structured session JSONL files.
- Avoid `~/.codex/history.jsonl` unless the user explicitly asks for it; it is a thin input history and can expose raw sensitive fragments.
- Favor `event_msg` entries with `payload.type == "user_message"` when extracting user text from Codex session logs.
- Run the helper script by resolving paths from the skill directory, not the current project cwd.
