from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from mjlab.flashsac_verification import (
  CHECKPOINT_MILESTONES,
  FAIL_FAST_GATES,
  TRACKING_BASELINE_SHA,
  authoritative_ppo_gold_baseline,
  build_lanev_comparison_snapshot,
  build_lanev_protocol_bundle,
  judge_acceptance,
  ppo_control_protocol,
  render_lanev_acceptance_notes,
  validate_comparison_entry,
)


def _valid_comparison_entry() -> dict[str, object]:
  return {
    "lane_id": "laneA",
    "hypothesis": "Upstream parity should fix the replay/update mismatch.",
    "backend": "flashsac",
    "baseline_sha": TRACKING_BASELINE_SHA,
    "candidate_sha": "abc1234",
    "worktree_name": "wt/flashsac-parity",
    "changed_files": ["src/mjlab/flashsac/agent.py"],
    "train_command": "uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac",
    "evaluate_command": "uv run evaluate-tracking --backend flashsac",
    "play_command": "uv run play Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac",
    "config_hash": "cfg-hash",
    "eval_hash": "eval-hash",
    "play_hash": "play-hash",
    "seeds": [7, 11],
    "artifact_bundle_path": "/tmp/flashsac-lane-a",
    "fail_fast_status": {gate: "pass" for gate in FAIL_FAST_GATES},
    "checkpoint_status": {
      milestone: "pass" if milestone in {"1M", "5M", "10M"} else "pending"
      for milestone in CHECKPOINT_MILESTONES
    },
    "visually_obvious_tracking": True,
    "ppo_comparable": True,
    "change_classification": "generic_backend_fix",
    "reproducible": True,
    "recommendation": "promote",
    "notes": "Ready for review.",
  }


def _sample_local_ppo_eval() -> dict[str, object]:
  return {
    "success_rate": 0.25,
    "mpkpe": 0.0982,
    "r_mpkpe": 0.0527,
    "joint_vel_error": 4.2959,
    "ee_pos_error": 0.1228,
    "ee_ori_error": 0.2172,
  }


def _sample_flashsac_eval() -> dict[str, object]:
  return {
    "success_rate": 0.0,
    "mpkpe": 0.1827,
    "r_mpkpe": 0.0986,
    "joint_vel_error": 6.6340,
    "ee_pos_error": 0.2528,
    "ee_ori_error": 0.5170,
  }


def test_protocol_bundle_uses_absolute_source_of_truth_paths() -> None:
  bundle = build_lanev_protocol_bundle()
  source_of_truth = cast(dict[str, str], bundle["source_of_truth"])
  gold_baseline = cast(
    dict[str, object], bundle["authoritative_ppo_gold_baseline"]
  )

  assert bundle["baseline_sha"] == TRACKING_BASELINE_SHA
  for path in source_of_truth.values():
    resolved = Path(path)
    assert resolved.is_absolute()
    assert resolved.is_file()

  ppo_protocol = ppo_control_protocol()
  assert ppo_protocol["gold_baseline"] == gold_baseline
  required_fields = cast(tuple[str, ...], ppo_protocol["required_manifest_fields"])
  assert ppo_protocol["source_of_truth"] == source_of_truth
  assert "summary_table_row_id" in required_fields
  json.dumps(bundle, ensure_ascii=False)


def test_authoritative_ppo_gold_baseline_points_to_existing_files() -> None:
  gold_baseline = authoritative_ppo_gold_baseline()
  run_dir = Path(cast(str, gold_baseline["run_dir"]))
  checkpoint_file = Path(cast(str, gold_baseline["preferred_checkpoint_file"]))
  motion_file = Path(cast(str, gold_baseline["motion_file"]))
  params = cast(dict[str, str], gold_baseline["params"])
  artifacts = cast(dict[str, object], gold_baseline["artifacts"])
  local_commands = cast(
    dict[str, str], gold_baseline["local_reproduction_commands"]
  )

  assert run_dir.is_absolute()
  assert run_dir.is_dir()
  assert checkpoint_file.is_file()
  assert motion_file.is_file()
  assert Path(params["agent_yaml"]).is_file()
  assert Path(params["env_yaml"]).is_file()
  assert artifacts["checkpoint_sha256"] == (
    "29bf464dfb8bf28bde707054a58aff6e363af9e97f23eafd90a58bc13ae2940c"
  )
  assert artifacts["motion_file_sha256"] == (
    "89d469b7ac8b1c0cb75425ad3ef43f2e6336fecd809d5981b41794e3c9204818"
  )
  assert "--checkpoint-file" in local_commands["evaluate"]
  assert "--motion-file" in local_commands["evaluate"]
  assert "shared-path bug signal" in cast(str, gold_baseline["comparison_guidance"])


def test_validate_comparison_entry_requires_nested_gate_coverage() -> None:
  entry = _valid_comparison_entry()
  fail_fast_status = cast(dict[str, str], entry["fail_fast_status"])
  checkpoint_status = cast(dict[str, str], entry["checkpoint_status"])
  del fail_fast_status["50k"]
  del checkpoint_status["10M"]
  entry["recommendation"] = "ship-it"

  issues = validate_comparison_entry(entry)

  assert "fail_fast_status.50k" in issues
  assert "checkpoint_status.10M" in issues
  assert "recommendation=invalid" in issues


def test_judge_acceptance_passes_generic_backend_fix() -> None:
  decision = judge_acceptance(
    {
      "fixed_eval_play_protocol": True,
      "stable_visually_obvious_tracking": True,
      "ppo_comparable": True,
      "evaluated_seeds": 2,
      "evaluation_windows": 3,
      "reproducible": True,
      "change_classification": "generic_backend_fix",
      "evidence_bundle_complete": True,
    }
  )

  assert decision.passed is True
  assert decision.failures == ()


def test_judge_acceptance_requires_justified_contract_alignment() -> None:
  missing_justification = judge_acceptance(
    {
      "fixed_eval_play_protocol": True,
      "stable_visually_obvious_tracking": True,
      "ppo_comparable": True,
      "evaluated_seeds": 2,
      "evaluation_windows": 3,
      "reproducible": True,
      "change_classification": "generic_contract_alignment",
      "evidence_bundle_complete": True,
    }
  )
  with_justification = judge_acceptance(
    {
      "fixed_eval_play_protocol": True,
      "stable_visually_obvious_tracking": True,
      "ppo_comparable": True,
      "evaluated_seeds": 2,
      "evaluation_windows": 3,
      "reproducible": True,
      "change_classification": "generic_contract_alignment",
      "evidence_bundle_complete": True,
      "justification": "Aligns terminated/truncated semantics with the upstream contract.",
    }
  )

  assert missing_justification.passed is False
  assert (
    "Generic contract alignment requires an explicit justification."
    in missing_justification.failures
  )
  assert with_justification.passed is True


def test_judge_acceptance_rejects_cosmetic_tracking_relaxations() -> None:
  decision = judge_acceptance(
    {
      "fixed_eval_play_protocol": True,
      "stable_visually_obvious_tracking": True,
      "ppo_comparable": True,
      "evaluated_seeds": 2,
      "evaluation_windows": 3,
      "reproducible": True,
      "change_classification": "task_specific_relaxation",
      "evidence_bundle_complete": True,
      "cosmetic_tracking_relaxation": True,
    }
  )

  assert decision.passed is False
  assert any("cosmetic tracking relaxation" in failure for failure in decision.failures)
  assert any(
    "Task-specific relaxations" in failure for failure in decision.failures
  )


def test_lanev_comparison_snapshot_separates_floor_from_debug_signal() -> None:
  snapshot = build_lanev_comparison_snapshot(
    local_ppo_eval=_sample_local_ppo_eval(),
    flashsac_eval=_sample_flashsac_eval(),
    shared_path_differential={
      "shared_path_root_cause": (
        "evaluate.py and play.py load env/agent config from the current task_id "
        "instead of the saved run params in the checkpoint directory."
      ),
      "implication_for_flashsac": (
        "FlashSAC comparisons can be invalidated by the same config-loading/task-id "
        "mismatch before any algorithm-specific conclusion is warranted."
      ),
    },
  )

  rows = cast(list[dict[str, object]], snapshot["rows"])
  authoritative = rows[0]
  local_reproduced = rows[1]
  flashsac = rows[2]
  delta = cast(dict[str, float], snapshot["local_minus_flashsac_delta"])
  notes = cast(list[str], snapshot["notes"])

  assert authoritative["lane_id"] == "authoritative_native_ppo"
  assert authoritative["role"] == "authoritative_floor"
  assert local_reproduced["lane_id"] == "local_reproduced_ppo"
  assert local_reproduced["role"] == "debug_signal_only"
  assert flashsac["lane_id"] == "flashsac_reference_smoke"
  assert delta["success_rate"] == 0.25
  assert any("shared-path root cause" in note for note in notes)
  assert any("FlashSAC comparison implication" in note for note in notes)


def test_render_lanev_acceptance_notes_includes_three_way_split() -> None:
  rendered = render_lanev_acceptance_notes(
    build_lanev_comparison_snapshot(
      local_ppo_eval=_sample_local_ppo_eval(),
      flashsac_eval=_sample_flashsac_eval(),
      shared_path_differential={
        "shared_path_root_cause": (
          "evaluate.py and play.py load env/agent config from the current task_id "
          "instead of the saved run params in the checkpoint directory."
        ),
      },
    )
  )

  assert "# LaneV comparison and acceptance notes" in rendered
  assert "| authoritative_native_ppo | authoritative_floor | ppo |" in rendered
  assert "| local_reproduced_ppo | debug_signal_only | ppo | 0.2500 |" in rendered
  assert "| flashsac_reference_smoke | candidate_reference | flashsac | 0.0000 |" in rendered
  assert "Current best shared-path root cause" in rendered
