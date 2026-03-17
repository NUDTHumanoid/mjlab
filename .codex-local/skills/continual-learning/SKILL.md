---
name: continual-learning
description: Use when the user wants to mine prior Codex sessions, preserve recurring corrections or preferences, or incrementally maintain AGENTS.md from conversation history.
---

# Continual Learning

## Overview

Keep AGENTS.md current by reviewing only new or changed Codex session logs, extracting durable memory, and writing back a minimal two-section memory file plus an incremental index.

## Quick Start

1. Read the existing `AGENTS.md` if it exists.
2. Resolve `scripts/update_memory.py` relative to this skill directory, not the current project cwd.
3. Run `scan` to discover changed session logs and extract user messages.
4. Review the scan output against `references/memory-rules.md`.
5. Decide which items are durable enough to keep.
6. Run `apply` to rewrite `AGENTS.md` and advance the index.

## Transcript Sources

- Prefer `~/.codex/sessions/` as the transcript root.
- Do not use `~/.codex/history.jsonl` by default. It is lossy and may contain raw sensitive fragments.
- Use a user-provided transcript directory if the user explicitly points to one.

## Workflow

### 1. Gather the delta

Run:

```bash
SKILL_DIR=/path/to/continual-learning
python "$SKILL_DIR/scripts/update_memory.py" scan \
  --transcript-root ~/.codex/sessions \
  --index-file .codex-local/state/continual-learning-index.json \
  --output /tmp/continual-learning-scan.json
```

This produces a JSON summary of new or modified session files and redacted user messages.

### 2. Extract candidates

Read the scan output and keep only items that satisfy all of these:

- actionable in future sessions
- stable across sessions
- repeated across sessions, or explicitly stated as a broad rule
- non-sensitive

If unsure about inclusion, read `references/memory-rules.md`.

### 3. Update AGENTS.md

Run `apply` with the bullets you want to keep:

```bash
SKILL_DIR=/path/to/continual-learning
python "$SKILL_DIR/scripts/update_memory.py" apply \
  --scan-file /tmp/continual-learning-scan.json \
  --agents-file AGENTS.md \
  --preference "Default to Chinese unless English is clearly more accurate." \
  --fact "This workspace primarily uses Python."
```

`apply` uses the index path recorded in the scan file. Do not point it at a different index.

If an older bullet should be replaced, drop it and add the revised wording in the same command:

```bash
SKILL_DIR=/path/to/continual-learning
python "$SKILL_DIR/scripts/update_memory.py" apply \
  --scan-file /tmp/continual-learning-scan.json \
  --drop-preference "Prefer direct answers." \
  --preference "Prefer direct, actionable answers with explicit assumptions."
```

## Output Contract

`AGENTS.md` must contain only:

- `## Learned User Preferences`
- `## Learned Workspace Facts`

Under each section, use plain bullet points only. Do not add evidence tags, confidence scores, rationale, or workflow notes.

## Safety Rules

- Never store secrets, tokens, passwords, personal data, or one-time approval codes.
- Never store branch names, commit hashes, temporary failures, or one-off task instructions.
- If the new information is too weak or too specific, do not add it.
- If nothing durable is found, still advance the index after review so the same sessions are not reprocessed forever.

## Resources

- `references/memory-rules.md`: inclusion bar, exclusions, good and bad bullet examples, and conflict resolution rules.
- `scripts/update_memory.py`: deterministic helper for delta discovery, AGENTS rewrite, and index maintenance.
