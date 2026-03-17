#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_TRANSCRIPT_ROOT = Path.home() / ".codex" / "sessions"
DEFAULT_AGENTS_FILE = Path("AGENTS.md")
DEFAULT_INDEX_FILE = Path(".codex-local/state/continual-learning-index.json")

PREFERENCES_HEADER = "## Learned User Preferences"
FACTS_HEADER = "## Learned Workspace Facts"

SENSITIVE_PATTERNS = [
  re.compile(pattern, re.IGNORECASE)
  for pattern in [
    r"\b(password|passwd|pwd|secret|token|api[_ -]?key|access[_ -]?token|refresh[_ -]?token)\b",
    r"(密码|口令|密钥|验证码|令牌)",
    r"https?://\S*[?&](token|key|secret|auth|code|password)=",
  ]
]
TOKEN_LIKE_PATTERNS = [
  re.compile(pattern)
  for pattern in [
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"sk-[A-Za-z0-9]{16,}",
    r"AIza[0-9A-Za-z_-]{20,}",
  ]
]


@dataclass
class TranscriptSummary:
  path: str
  mtime_ms: int
  user_messages: list[str]


def iso_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def load_index(index_file: Path) -> dict:
  if not index_file.exists():
    return {"version": 1, "transcripts": {}}
  return json.loads(index_file.read_text())


def save_index(index_file: Path, index: dict) -> None:
  index_file.parent.mkdir(parents=True, exist_ok=True)
  index_file.write_text(json.dumps(index, indent=2, ensure_ascii=True) + "\n")


def redact_message(message: str) -> str:
  stripped = message.strip()
  if not stripped:
    return ""
  if any(pattern.search(stripped) for pattern in SENSITIVE_PATTERNS):
    return "[REDACTED: sensitive-looking input]"
  if any(pattern.search(stripped) for pattern in TOKEN_LIKE_PATTERNS):
    return "[REDACTED: token-like input]"
  if (
    " " not in stripped
    and "/" not in stripped
    and len(stripped) >= 24
    and re.fullmatch(r"[A-Za-z0-9._-]{24,}", stripped)
  ):
    return "[REDACTED: token-like input]"
  if re.fullmatch(r"\d{1,8}", stripped):
    return "[REDACTED: short numeric input]"
  return stripped


def extract_user_messages(session_file: Path) -> list[str]:
  messages: list[str] = []
  for raw_line in session_file.read_text().splitlines():
    if not raw_line.strip():
      continue
    try:
      record = json.loads(raw_line)
    except json.JSONDecodeError:
      continue
    if record.get("type") != "event_msg":
      continue
    payload = record.get("payload", {})
    if payload.get("type") != "user_message":
      continue
    message = redact_message(str(payload.get("message", "")))
    if message:
      messages.append(message)
  return messages


def discover_changed_transcripts(
  transcript_root: Path,
  index: dict,
  limit_files: int | None = None,
) -> list[TranscriptSummary]:
  known = index.get("transcripts", {})
  changed: list[TranscriptSummary] = []
  session_files = sorted(transcript_root.rglob("*.jsonl"))
  for session_file in session_files:
    mtime_ms = int(session_file.stat().st_mtime * 1000)
    key = str(session_file)
    previous = known.get(key)
    if previous is not None and previous.get("mtimeMs") == mtime_ms:
      continue
    changed.append(
      TranscriptSummary(
        path=key,
        mtime_ms=mtime_ms,
        user_messages=extract_user_messages(session_file),
      )
    )
    if limit_files is not None and len(changed) >= limit_files:
      break
  return changed


def parse_agents_file(agents_file: Path) -> tuple[list[str], list[str]]:
  if not agents_file.exists():
    return [], []

  preferences: list[str] = []
  facts: list[str] = []
  current: list[str] | None = None

  for line in agents_file.read_text().splitlines():
    stripped = line.strip()
    if stripped == PREFERENCES_HEADER:
      current = preferences
      continue
    if stripped == FACTS_HEADER:
      current = facts
      continue
    if current is not None and stripped.startswith("- "):
      current.append(stripped[2:].strip())

  return preferences, facts


def normalize_bullet(bullet: str) -> str:
  return " ".join(bullet.split()).strip().lower()


def merge_bullets(
  existing: Iterable[str],
  additions: Iterable[str],
  drops: Iterable[str],
) -> list[str]:
  drop_set = {normalize_bullet(item) for item in drops}
  merged: list[str] = []
  seen: set[str] = set()
  for bullet in list(existing) + list(additions):
    normalized = normalize_bullet(bullet)
    if not normalized or normalized in drop_set or normalized in seen:
      continue
    seen.add(normalized)
    merged.append(" ".join(bullet.split()).strip())
  return merged


def write_agents_file(
  agents_file: Path,
  preferences: list[str],
  facts: list[str],
) -> None:
  lines = [PREFERENCES_HEADER]
  lines.extend(f"- {item}" for item in preferences)
  lines.append("")
  lines.append(FACTS_HEADER)
  lines.extend(f"- {item}" for item in facts)
  agents_file.parent.mkdir(parents=True, exist_ok=True)
  agents_file.write_text("\n".join(lines).rstrip() + "\n")


def cleanup_index_entries(index: dict, transcript_root: Path) -> None:
  existing_files = {str(path) for path in transcript_root.rglob("*.jsonl")}
  transcripts = index.get("transcripts", {})
  stale_keys = [
    key
    for key in transcripts
    if Path(key).is_relative_to(transcript_root) and key not in existing_files
  ]
  for key in stale_keys:
    transcripts.pop(key, None)


def command_scan(args: argparse.Namespace) -> int:
  transcript_root = Path(args.transcript_root).expanduser()
  index_file = Path(args.index_file)
  index = load_index(index_file)
  changed = discover_changed_transcripts(
    transcript_root=transcript_root,
    index=index,
    limit_files=args.limit_files,
  )
  payload = {
    "version": 1,
    "generatedAt": iso_now(),
    "transcriptRoot": str(transcript_root),
    "indexFile": str(index_file),
    "changedTranscripts": [
      {
        "path": item.path,
        "mtimeMs": item.mtime_ms,
        "userMessages": item.user_messages,
      }
      for item in changed
    ],
  }
  output = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
  if args.output:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)
  else:
    print(output, end="")
  return 0


def command_apply(args: argparse.Namespace) -> int:
  scan_file = Path(args.scan_file)
  scan_payload = json.loads(scan_file.read_text())

  agents_file = Path(args.agents_file)
  scan_index_file = Path(scan_payload.get("indexFile", DEFAULT_INDEX_FILE))
  if args.index_file is not None and Path(args.index_file) != scan_index_file:
    print(
      f"[WARN] Ignoring mismatched --index-file {args.index_file}; "
      f"using scan file index {scan_index_file}"
    )
  index_file = scan_index_file
  transcript_root = Path(scan_payload["transcriptRoot"]).expanduser()
  existing_preferences, existing_facts = parse_agents_file(agents_file)

  preferences = merge_bullets(
    existing=existing_preferences,
    additions=args.preference or [],
    drops=args.drop_preference or [],
  )
  facts = merge_bullets(
    existing=existing_facts,
    additions=args.fact or [],
    drops=args.drop_fact or [],
  )

  write_agents_file(agents_file, preferences, facts)

  index = load_index(index_file)
  transcripts = index.setdefault("transcripts", {})
  processed_at = iso_now()
  for entry in scan_payload.get("changedTranscripts", []):
    transcripts[entry["path"]] = {
      "mtimeMs": entry["mtimeMs"],
      "lastProcessedAt": processed_at,
    }
  cleanup_index_entries(index, transcript_root)
  save_index(index_file, index)

  print(f"Wrote {agents_file}")
  print(f"Indexed {len(scan_payload.get('changedTranscripts', []))} transcript files")
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Incrementally scan Codex session logs and maintain AGENTS.md."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  scan_parser = subparsers.add_parser("scan", help="Find changed session logs")
  scan_parser.add_argument(
    "--transcript-root",
    default=str(DEFAULT_TRANSCRIPT_ROOT),
    help="Transcript root to scan",
  )
  scan_parser.add_argument(
    "--index-file",
    default=str(DEFAULT_INDEX_FILE),
    help="Incremental index JSON file",
  )
  scan_parser.add_argument(
    "--output",
    help="Optional JSON output path for the scan summary",
  )
  scan_parser.add_argument(
    "--limit-files",
    type=int,
    help="Optional cap on changed transcript files",
  )
  scan_parser.set_defaults(func=command_scan)

  apply_parser = subparsers.add_parser(
    "apply",
    help="Rewrite AGENTS.md and update the index from a scan file",
  )
  apply_parser.add_argument("--scan-file", required=True, help="Scan summary JSON file")
  apply_parser.add_argument(
    "--agents-file",
    default=str(DEFAULT_AGENTS_FILE),
    help="AGENTS.md path to rewrite",
  )
  apply_parser.add_argument(
    "--index-file",
    help="Optional override; must match the index file recorded in the scan file",
  )
  apply_parser.add_argument(
    "--preference",
    action="append",
    help="Preference bullet to add",
  )
  apply_parser.add_argument(
    "--fact",
    action="append",
    help="Workspace fact bullet to add",
  )
  apply_parser.add_argument(
    "--drop-preference",
    action="append",
    help="Existing preference bullet to remove",
  )
  apply_parser.add_argument(
    "--drop-fact",
    action="append",
    help="Existing fact bullet to remove",
  )
  apply_parser.set_defaults(func=command_apply)

  return parser


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
