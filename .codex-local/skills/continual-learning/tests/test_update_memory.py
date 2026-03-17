from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
  Path(__file__).resolve().parent.parent / "scripts" / "update_memory.py"
)
SPEC = importlib.util.spec_from_file_location("update_memory", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContinualLearningScriptTests(unittest.TestCase):
  def test_cleanup_index_only_touches_scanned_subtree(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      scanned_root = root / "sessions" / "2026" / "03" / "17"
      other_root = root / "sessions" / "2026" / "03" / "18"
      scanned_root.mkdir(parents=True)
      other_root.mkdir(parents=True)

      scanned_file = scanned_root / "a.jsonl"
      scanned_file.write_text("")
      other_file = other_root / "b.jsonl"
      other_file.write_text("")

      index = {
        "version": 1,
        "transcripts": {
          str(scanned_file): {"mtimeMs": 1, "lastProcessedAt": "t1"},
          str(other_file): {"mtimeMs": 2, "lastProcessedAt": "t2"},
        },
      }

      MODULE.cleanup_index_entries(index, scanned_root)

      self.assertIn(str(scanned_file), index["transcripts"])
      self.assertIn(str(other_file), index["transcripts"])

  def test_apply_uses_scan_index_file_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      transcript_root = root / "sessions"
      transcript_root.mkdir()
      transcript_file = transcript_root / "example.jsonl"
      transcript_file.write_text("")

      scan_file = root / "scan.json"
      expected_index = root / "chosen-index.json"
      default_index = root / "different-index.json"
      agents_file = root / "AGENTS.md"

      scan_file.write_text(
        json.dumps(
          {
            "version": 1,
            "generatedAt": "2026-03-17T00:00:00+00:00",
            "transcriptRoot": str(transcript_root),
            "indexFile": str(expected_index),
            "changedTranscripts": [
              {
                "path": str(transcript_file),
                "mtimeMs": 123,
                "userMessages": ["Prefer Chinese."],
              }
            ],
          }
        )
      )

      parser = MODULE.build_parser()
      args = parser.parse_args(
        [
          "apply",
          "--scan-file",
          str(scan_file),
          "--agents-file",
          str(agents_file),
          "--index-file",
          str(default_index),
          "--preference",
          "Default to Chinese.",
        ]
      )
      MODULE.command_apply(args)

      self.assertTrue(expected_index.exists())
      self.assertFalse(default_index.exists())

  def test_redact_message_handles_token_like_inputs(self) -> None:
    self.assertEqual(
      MODULE.redact_message("ghp_1234567890abcdefghijklmnopqrstuvwxyz"),
      "[REDACTED: token-like input]",
    )
    self.assertEqual(
      MODULE.redact_message("https://example.com/callback?token=abc123"),
      "[REDACTED: sensitive-looking input]",
    )
    self.assertEqual(
      MODULE.redact_message("A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"),
      "[REDACTED: token-like input]",
    )


if __name__ == "__main__":
  unittest.main()
