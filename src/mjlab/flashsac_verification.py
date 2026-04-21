from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TRACKING_BASELINE_SHA = "2fc0c19dfc4b87187d6372bf97965b3d40bda6d0"
AUTHORITATIVE_PPO_GOLD_RUN_DIR = Path(
  "/home/ssy/ssy_files/mjlab/logs/rsl_rl/g1_tracking_handstand1/"
  "2026-04-14_12-19-21_handstand1_acrobatics_ft_40000"
)
AUTHORITATIVE_PPO_GOLD_CHECKPOINT = AUTHORITATIVE_PPO_GOLD_RUN_DIR / "model_31500.pt"
AUTHORITATIVE_PPO_GOLD_MOTION_FILE = Path(
  "/home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/"
  "pipeline/mjlab/motion.npz"
)
FAIL_FAST_GATES = ("pretrain", "50k", "100k", "250k")
CHECKPOINT_MILESTONES = ("1M", "5M", "10M", "25M", "50M")
STATUS_VALUES = ("not_run", "pending", "pass", "fail", "blocked")
CHANGE_CLASSIFICATIONS = (
  "generic_backend_fix",
  "generic_contract_alignment",
  "task_specific_relaxation",
  "unknown",
)
RECOMMENDATIONS = ("promote", "hold", "quarantine", "reject")
COMPARISON_METRIC_KEYS = (
  "success_rate",
  "mpkpe",
  "r_mpkpe",
  "joint_vel_error",
  "ee_pos_error",
  "ee_ori_error",
)
ChangeClassification = Literal[
  "generic_backend_fix",
  "generic_contract_alignment",
  "task_specific_relaxation",
  "unknown",
]


@dataclass(frozen=True)
class SourceOfTruthPaths:
  spec_path: str
  prd_path: str
  test_spec_path: str


@dataclass(frozen=True)
class SchemaField:
  name: str
  type: str
  description: str
  required: bool = True
  required_keys: tuple[str, ...] = ()
  allowed_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChecklistItem:
  item_id: str
  layer: Literal["baseline", "static", "smoke", "checkpoint", "acceptance"]
  description: str
  evidence: str
  required: bool = True


@dataclass(frozen=True)
class AcceptanceEvidence:
  fixed_eval_play_protocol: bool
  stable_visually_obvious_tracking: bool
  ppo_comparable: bool
  evaluated_seeds: int
  evaluation_windows: int
  reproducible: bool
  change_classification: ChangeClassification
  evidence_bundle_complete: bool = True
  cosmetic_tracking_relaxation: bool = False
  transient_only: bool = False
  justification: str | None = None

  @classmethod
  def from_mapping(cls, data: Mapping[str, object]) -> "AcceptanceEvidence":
    classification = _require_str(data, "change_classification")
    if classification not in CHANGE_CLASSIFICATIONS:
      raise ValueError(
        "change_classification must be one of "
        + ", ".join(CHANGE_CLASSIFICATIONS)
      )

    return cls(
      fixed_eval_play_protocol=_require_bool(data, "fixed_eval_play_protocol"),
      stable_visually_obvious_tracking=_require_bool(
        data, "stable_visually_obvious_tracking"
      ),
      ppo_comparable=_require_bool(data, "ppo_comparable"),
      evaluated_seeds=_require_int(data, "evaluated_seeds"),
      evaluation_windows=_require_int(data, "evaluation_windows"),
      reproducible=_require_bool(data, "reproducible"),
      change_classification=classification,
      evidence_bundle_complete=_optional_bool(data, "evidence_bundle_complete", True),
      cosmetic_tracking_relaxation=_optional_bool(
        data, "cosmetic_tracking_relaxation", False
      ),
      transient_only=_optional_bool(data, "transient_only", False),
      justification=_optional_str(data, "justification"),
    )


@dataclass(frozen=True)
class AcceptanceDecision:
  passed: bool
  failures: tuple[str, ...]
  warnings: tuple[str, ...]


def _repo_root(repo_root: Path | None = None) -> Path:
  start = (
    repo_root.expanduser().resolve()
    if repo_root is not None
    else Path(__file__).resolve().parents[2]
  )
  required_paths = (
    ".omx/specs/deep-interview-flashsac-rootcause-parallel-refactor.md",
    ".omx/plans/prd-flashsac-tracking-rootcause-team.md",
    ".omx/plans/test-spec-flashsac-tracking-rootcause-team.md",
  )
  for candidate in (start, *start.parents):
    if all((candidate / relative).is_file() for relative in required_paths):
      return candidate
  return start


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _require_bool(data: Mapping[str, object], key: str) -> bool:
  value = data.get(key)
  if not isinstance(value, bool):
    raise TypeError(f"{key} must be a bool")
  return value


def _require_int(data: Mapping[str, object], key: str) -> int:
  value = data.get(key)
  if not isinstance(value, int) or isinstance(value, bool):
    raise TypeError(f"{key} must be an int")
  return value


def _require_str(data: Mapping[str, object], key: str) -> str:
  value = data.get(key)
  if not isinstance(value, str) or not value:
    raise TypeError(f"{key} must be a non-empty str")
  return value


def _optional_bool(
  data: Mapping[str, object], key: str, default: bool
) -> bool:
  value = data.get(key, default)
  if not isinstance(value, bool):
    raise TypeError(f"{key} must be a bool")
  return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
  value = data.get(key)
  if value is None:
    return None
  if not isinstance(value, str) or not value.strip():
    raise TypeError(f"{key} must be a non-empty str when provided")
  return value


def _optional_mapping(
  data: Mapping[str, object], key: str
) -> Mapping[str, object] | None:
  value = data.get(key)
  if value is None:
    return None
  if not isinstance(value, Mapping):
    raise TypeError(f"{key} must be an object when provided")
  return value


def _optional_number(data: Mapping[str, object], key: str) -> float | None:
  value = data.get(key)
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int | float):
    raise TypeError(f"{key} must be a number when provided")
  return float(value)


def source_of_truth_paths(repo_root: Path | None = None) -> SourceOfTruthPaths:
  root = _repo_root(repo_root)
  return SourceOfTruthPaths(
    spec_path=str(
      (
        root / ".omx/specs/deep-interview-flashsac-rootcause-parallel-refactor.md"
      ).resolve()
    ),
    prd_path=str((root / ".omx/plans/prd-flashsac-tracking-rootcause-team.md").resolve()),
    test_spec_path=str(
      (root / ".omx/plans/test-spec-flashsac-tracking-rootcause-team.md").resolve()
    ),
  )


def authoritative_ppo_gold_baseline() -> dict[str, object]:
  run_dir = AUTHORITATIVE_PPO_GOLD_RUN_DIR.resolve()
  checkpoint_file = AUTHORITATIVE_PPO_GOLD_CHECKPOINT.resolve()
  motion_file = AUTHORITATIVE_PPO_GOLD_MOTION_FILE.resolve()
  agent_yaml = (run_dir / "params/agent.yaml").resolve()
  env_yaml = (run_dir / "params/env.yaml").resolve()
  eval_output = (
    '$OMX_TEAM_LEADER_CWD/.omx/logs/lanev-ppo-authoritative-baseline-eval.json'
  )
  play_output = (
    '$OMX_TEAM_LEADER_CWD/.omx/artifacts/flashsac_tracking/'
    "laneV-ppo-control/seed-42/play/lanev-ppo-control-step-0.mp4"
  )
  return {
    "status": "authoritative_native_ppo_gold_baseline",
    "source": "user-supplied successful native PPO run",
    "backend": "ppo",
    "task_id": "Mjlab-Tracking-Flat-Unitree-G1",
    "run_dir": str(run_dir),
    "preferred_checkpoint_file": str(checkpoint_file),
    "motion_file": str(motion_file),
    "params": {
      "agent_yaml": str(agent_yaml),
      "env_yaml": str(env_yaml),
    },
    "artifacts": {
      "checkpoint_sha256": _sha256_file(checkpoint_file),
      "checkpoint_bytes": checkpoint_file.stat().st_size,
      "motion_file_sha256": _sha256_file(motion_file),
      "motion_file_bytes": motion_file.stat().st_size,
      "agent_yaml_sha256": _sha256_file(agent_yaml),
      "env_yaml_sha256": _sha256_file(env_yaml),
    },
    "local_reproduction_commands": {
      "evaluate": (
        'cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking '
        "Mjlab-Tracking-Flat-Unitree-G1 --checkpoint-file "
        f"{checkpoint_file} --motion-file {motion_file} --num-envs 4 --device cpu "
        f"--output-file {eval_output}"
      ),
      "play": (
        'cd "$OMX_TEAM_LEADER_CWD" && uv run python <inline-rslrl-video-recorder> '
        f"--checkpoint-file {checkpoint_file} --motion-file {motion_file} --num-envs 1 "
        f'--device cpu --video-length 200 --output-video {play_output}'
      ),
    },
    "comparison_guidance": (
      "Use the native run directory/checkpoint as the PPO floor. Treat local "
      "checkpoint+motion eval/play as a shared-path bug signal until it reproduces "
      "the native success."
    ),
  }


def ppo_control_protocol(repo_root: Path | None = None) -> dict[str, object]:
  gold_baseline = authoritative_ppo_gold_baseline()
  return {
    "owner": "LaneV independent verifier",
    "backend": "ppo",
    "task_id": "Mjlab-Tracking-Flat-Unitree-G1",
    "baseline_sha": TRACKING_BASELINE_SHA,
    "source_of_truth": asdict(source_of_truth_paths(repo_root)),
    "gold_baseline": gold_baseline,
    "minimum_acceptance": {
      "min_seeds": 2,
      "min_evaluation_windows": 3,
      "fixed_eval_play_protocol": True,
      "visually_obvious_tracking": True,
      "ppo_sets_the_floor": True,
      "authoritative_native_ppo_gold_baseline": True,
    },
    "required_manifest_fields": (
      "train_command",
      "evaluate_command",
      "play_command",
      "config_hash",
      "eval_hash",
      "play_hash",
      "seed_list",
      "artifact_bundle_path",
      "evaluation_json_paths",
      "video_artifact_paths",
      "summary_table_row_id",
    ),
    "comparison_policy": {
      "authoritative_floor": "gold_baseline",
      "local_reproduction_status": "debug-signal-only-until-parity-restored",
    },
    "shared_protocol_requirements": (
      "Use the same evaluation JSON schema, video naming, and comparison-table format as FlashSAC lanes.",
      "Keep seed policy identical to the comparison table entry.",
      "Record config/eval/play hashes before the first PPO control run.",
      "Publish a PPO evidence bundle that can be replayed from the same worktree without hidden overrides.",
      "Do not treat degraded local reproduced PPO eval/play as evidence that PPO itself failed; keep it separated as shared-path debugging telemetry.",
    ),
  }


def _comparison_schema_fields() -> tuple[SchemaField, ...]:
  return (
    SchemaField("lane_id", "string", "Lane identifier such as lane0, laneA, or ppo_control."),
    SchemaField("hypothesis", "string", "Why this lane should change behavior."),
    SchemaField("backend", "string", "flashsac or ppo."),
    SchemaField("baseline_sha", "string", "Pinned clean baseline commit SHA."),
    SchemaField("candidate_sha", "string", "Lane commit SHA for the evidence bundle."),
    SchemaField("worktree_name", "string", "Human-readable worktree / branch label."),
    SchemaField("changed_files", "array[string]", "Files changed in the lane."),
    SchemaField("train_command", "string", "Exact train command."),
    SchemaField("evaluate_command", "string", "Exact evaluate command."),
    SchemaField("play_command", "string", "Exact play / video command."),
    SchemaField("config_hash", "string", "Hash of the effective training config."),
    SchemaField("eval_hash", "string", "Hash of the evaluation script/config."),
    SchemaField("play_hash", "string", "Hash of the play script/config."),
    SchemaField("seeds", "array[int]", "Seed list used by the lane."),
    SchemaField(
      "artifact_bundle_path",
      "string",
      "Absolute or repo-root-relative path to the evidence bundle.",
    ),
    SchemaField(
      "fail_fast_status",
      "object",
      "Per-gate status map for pretrain/50k/100k/250k.",
      required_keys=FAIL_FAST_GATES,
      allowed_values=STATUS_VALUES,
    ),
    SchemaField(
      "checkpoint_status",
      "object",
      "Per-checkpoint status map for 1M/5M/10M/25M/50M.",
      required_keys=CHECKPOINT_MILESTONES,
      allowed_values=STATUS_VALUES,
    ),
    SchemaField(
      "visually_obvious_tracking",
      "bool",
      "Whether play/video evidence shows stable tracking.",
    ),
    SchemaField(
      "ppo_comparable",
      "bool",
      "Whether this lane is at least PPO-comparable under the shared protocol.",
    ),
    SchemaField(
      "change_classification",
      "string",
      "How to classify the retained changes.",
      allowed_values=CHANGE_CLASSIFICATIONS,
    ),
    SchemaField(
      "reproducible",
      "bool",
      "Whether the evidence bundle is sufficient to replay the claim.",
    ),
    SchemaField(
      "recommendation",
      "string",
      "Final LaneV recommendation for integration.",
      allowed_values=RECOMMENDATIONS,
    ),
    SchemaField(
      "notes",
      "string",
      "Optional verifier notes or blockers.",
      required=False,
    ),
  )


def comparison_table_schema() -> dict[str, object]:
  return {
    "schema_name": "flashsac_tracking_lane_comparison_row",
    "required_fields": [asdict(field) for field in _comparison_schema_fields()],
    "status_values": STATUS_VALUES,
    "change_classifications": CHANGE_CLASSIFICATIONS,
    "recommendations": RECOMMENDATIONS,
  }


def verification_checklist(
  repo_root: Path | None = None,
) -> list[dict[str, object]]:
  plan_artifacts = asdict(source_of_truth_paths(repo_root))
  items = (
    ChecklistItem(
      item_id="baseline-sha-pinned",
      layer="baseline",
      description="Baseline SHA is pinned to the clean 2fc0c19 team baseline.",
      evidence=TRACKING_BASELINE_SHA,
    ),
    ChecklistItem(
      item_id="source-of-truth-artifacts",
      layer="baseline",
      description="Spec, PRD, and test spec are recorded with absolute paths for this lane.",
      evidence=json.dumps(plan_artifacts, ensure_ascii=False),
    ),
    ChecklistItem(
      item_id="commands-and-hashes-recorded",
      layer="baseline",
      description="Train/eval/play commands plus config/eval/play hashes are present.",
      evidence="comparison_table_row.{train_command,evaluate_command,play_command,config_hash,eval_hash,play_hash}",
    ),
    ChecklistItem(
      item_id="contract-tests-run",
      layer="static",
      description="Targeted static / contract tests cover replay, adapter, and eval semantics.",
      evidence="tests/test_flashsac_backend.py plus any lane-specific regression tests",
    ),
    ChecklistItem(
      item_id="smoke-gates-complete",
      layer="smoke",
      description="Pretrain, 50k, 100k, and 250k fail-fast gates are all recorded.",
      evidence="comparison_table_row.fail_fast_status",
    ),
    ChecklistItem(
      item_id="smoke-metrics-sane",
      layer="smoke",
      description="Replay fill, Q/entropy/alpha, action spread, and done ratios are collected.",
      evidence="lane evidence bundle summary and logs",
    ),
    ChecklistItem(
      item_id="checkpoint-evidence-attached",
      layer="checkpoint",
      description="Accepted checkpoints include evaluation JSON plus play/video artifacts.",
      evidence="comparison_table_row.checkpoint_status + artifact bundle paths",
    ),
    ChecklistItem(
      item_id="ppo-control-parity",
      layer="checkpoint",
      description="PPO control uses the same task, eval/play protocol, and summary table format.",
      evidence="authoritative_ppo_gold_baseline + ppo_control_protocol + ppo evidence bundle",
    ),
    ChecklistItem(
      item_id="stable-tracking-demonstrated",
      layer="acceptance",
      description="Candidate shows stable, visually obvious tracking over accepted windows.",
      evidence="comparison_table_row.visually_obvious_tracking",
    ),
    ChecklistItem(
      item_id="ppo-comparable-threshold",
      layer="acceptance",
      description="Candidate is PPO-comparable or better under the fixed protocol.",
      evidence="comparison_table_row.ppo_comparable",
    ),
    ChecklistItem(
      item_id="genericity-retained",
      layer="acceptance",
      description="Retained changes remain generic backend fixes or justified generic contract alignment.",
      evidence="comparison_table_row.change_classification",
    ),
    ChecklistItem(
      item_id="reproducibility-bundle",
      layer="acceptance",
      description="Evidence bundle is sufficient to reproduce the final recommendation.",
      evidence="comparison_table_row.reproducible",
    ),
  )
  return [asdict(item) for item in items]


def validate_comparison_entry(entry: Mapping[str, object]) -> tuple[str, ...]:
  issues: list[str] = []
  for field in _comparison_schema_fields():
    name = field.name
    required = field.required
    if required and name not in entry:
      issues.append(name)
      continue
    if name not in entry:
      continue

    required_keys = field.required_keys
    allowed_values = field.allowed_values
    value = entry[name]
    if required_keys:
      if not isinstance(value, Mapping):
        issues.append(name)
        continue
      for key in required_keys:
        if key not in value:
          issues.append(f"{name}.{key}")
          continue
        if allowed_values and value[key] not in allowed_values:
          issues.append(f"{name}.{key}=invalid")
    elif allowed_values and value not in allowed_values:
      issues.append(f"{name}=invalid")
  return tuple(issues)


def acceptance_rubric() -> dict[str, object]:
  return {
    "pass_requirements": (
      "fixed evaluation/play protocol preserved",
      "stable visually obvious tracking demonstrated",
      "PPO-comparable or better under at least 2 seeds",
      "at least 3 accepted evaluation windows recorded",
      "retained changes are a generic backend fix or justified generic contract alignment",
      "reproducible evidence bundle exists",
    ),
    "automatic_fail_conditions": (
      "candidate depends on cosmetic tracking relaxation",
      "results are only transient or one-off non-zero success",
      "evidence bundle is incomplete or not reproducible",
      "change classification is task_specific_relaxation or unknown",
    ),
    "change_classification_policy": {
      "accepted": (
        "generic_backend_fix",
        "generic_contract_alignment",
      ),
      "requires_justification": ("generic_contract_alignment",),
      "rejected": (
        "task_specific_relaxation",
        "unknown",
      ),
    },
  }


def judge_acceptance(
  evidence: AcceptanceEvidence | Mapping[str, object],
) -> AcceptanceDecision:
  resolved = (
    evidence
    if isinstance(evidence, AcceptanceEvidence)
    else AcceptanceEvidence.from_mapping(evidence)
  )
  failures: list[str] = []
  warnings: list[str] = []

  if resolved.cosmetic_tracking_relaxation:
    failures.append(
      "Candidate depends on a cosmetic tracking relaxation instead of a generic fix."
    )
  if resolved.transient_only:
    failures.append("Results are transient and do not satisfy stable tracking.")
  if not resolved.evidence_bundle_complete:
    failures.append("Evidence bundle is incomplete.")
  if not resolved.reproducible:
    failures.append("Evidence bundle is not reproducible.")
  if not resolved.fixed_eval_play_protocol:
    failures.append("Fixed evaluation/play protocol was not preserved.")
  if not resolved.stable_visually_obvious_tracking:
    failures.append("Stable visually obvious tracking was not demonstrated.")
  if not resolved.ppo_comparable:
    failures.append("PPO-comparable performance was not demonstrated.")
  if resolved.evaluated_seeds < 2:
    failures.append("At least 2 seeds are required for acceptance.")
  if resolved.evaluation_windows < 3:
    failures.append("At least 3 accepted evaluation windows are required.")
  if resolved.change_classification == "task_specific_relaxation":
    failures.append("Task-specific relaxations cannot be retained as the mainline fix.")
  if resolved.change_classification == "unknown":
    failures.append("Change classification is unknown; the retained fix is not justified.")
  if (
    resolved.change_classification == "generic_contract_alignment"
    and not resolved.justification
  ):
    failures.append(
      "Generic contract alignment requires an explicit justification."
    )
  if resolved.change_classification == "generic_backend_fix" and resolved.justification:
    warnings.append(
      "Generic backend fixes usually do not need extra justification; keep it concise."
    )

  return AcceptanceDecision(
    passed=not failures,
    failures=tuple(failures),
    warnings=tuple(warnings),
  )


def _metric_subset(metrics: Mapping[str, object]) -> dict[str, float]:
  subset: dict[str, float] = {}
  for key in COMPARISON_METRIC_KEYS:
    value = _optional_number(metrics, key)
    if value is not None:
      subset[key] = value
  return subset


def _metric_delta(
  left: Mapping[str, object] | None,
  right: Mapping[str, object] | None,
) -> dict[str, float]:
  if left is None or right is None:
    return {}
  left_metrics = _metric_subset(left)
  right_metrics = _metric_subset(right)
  return {
    key: left_metrics[key] - right_metrics[key]
    for key in COMPARISON_METRIC_KEYS
    if key in left_metrics and key in right_metrics
  }


def build_lanev_comparison_snapshot(
  local_ppo_eval: Mapping[str, object] | None = None,
  flashsac_eval: Mapping[str, object] | None = None,
  shared_path_differential: Mapping[str, object] | None = None,
) -> dict[str, object]:
  baseline = authoritative_ppo_gold_baseline()
  rows: list[dict[str, object]] = [
    {
      "lane_id": "authoritative_native_ppo",
      "role": "authoritative_floor",
      "backend": "ppo",
      "status": "source_of_truth",
      "run_dir": baseline["run_dir"],
      "checkpoint_file": baseline["preferred_checkpoint_file"],
      "motion_file": baseline["motion_file"],
      "notes": (
        "Canonical PPO floor from the user-supplied successful native run. "
        "Do not downgrade PPO capability based on local reproduced eval/play telemetry."
      ),
    }
  ]
  notes = [
    (
      "Authoritative native PPO success remains the only PPO capability floor. "
      "Local reproduced PPO eval/play must stay debug-only until parity is restored."
    )
  ]

  if local_ppo_eval is not None:
    rows.append(
      {
        "lane_id": "local_reproduced_ppo",
        "role": "debug_signal_only",
        "backend": "ppo",
        "status": "shared_path_debug_signal",
        "metrics": _metric_subset(local_ppo_eval),
        "notes": (
          "Checkpoint+motion replay under shared local runtime/eval/inference "
          "plumbing. Useful for reproduction debugging, not for redefining PPO capability."
        ),
      }
    )
    notes.append(
      (
        "Current local reproduced PPO metrics are debugging telemetry for the "
        "shared path, not a replacement for the authoritative PPO baseline."
      )
    )

  if flashsac_eval is not None:
    rows.append(
      {
        "lane_id": "flashsac_reference_smoke",
        "role": "candidate_reference",
        "backend": "flashsac",
        "status": "current_flashsac_reference",
        "metrics": _metric_subset(flashsac_eval),
        "notes": (
          "Current FlashSAC smoke/reference artifact. Compare against the authoritative "
          "PPO floor only after shared-path reproduction issues are separated."
        ),
      }
    )

  root_cause = None
  implication = None
  if shared_path_differential is not None:
    root_cause = _optional_str(shared_path_differential, "shared_path_root_cause")
    implication = _optional_str(shared_path_differential, "implication_for_flashsac")
  if root_cause:
    notes.append(f"Current best shared-path root cause: {root_cause}")
  if implication:
    notes.append(f"FlashSAC comparison implication: {implication}")

  return {
    "baseline_sha": TRACKING_BASELINE_SHA,
    "metric_keys": list(COMPARISON_METRIC_KEYS),
    "rows": rows,
    "local_minus_flashsac_delta": _metric_delta(local_ppo_eval, flashsac_eval),
    "shared_path_root_cause": root_cause,
    "implication_for_flashsac": implication,
    "notes": notes,
  }


def _format_metric(value: object) -> str:
  if value is None:
    return "n/a"
  if isinstance(value, bool):
    return str(value).lower()
  if isinstance(value, int | float):
    return f"{float(value):.4f}"
  return str(value)


def render_lanev_acceptance_notes(snapshot: Mapping[str, object]) -> str:
  baseline_sha = _require_str(snapshot, "baseline_sha")
  rows_value = snapshot.get("rows")
  if not isinstance(rows_value, Sequence) or isinstance(rows_value, str):
    raise TypeError("rows must be a sequence")
  notes_value = snapshot.get("notes")
  if not isinstance(notes_value, Sequence) or isinstance(notes_value, str):
    raise TypeError("notes must be a sequence")

  lines = [
    "# LaneV comparison and acceptance notes",
    "",
    f"- Baseline SHA: `{baseline_sha}`",
    "- Comparator policy: authoritative native PPO floor > debug-only local reproduced PPO telemetry > FlashSAC candidate artifacts.",
    "",
    "## Comparison table",
    "",
    "| Row | Role | Backend | success_rate | mpkpe | r_mpkpe | joint_vel_error | ee_pos_error | ee_ori_error | Notes |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  ]
  for row in rows_value:
    if not isinstance(row, Mapping):
      raise TypeError("rows entries must be objects")
    metrics = _optional_mapping(row, "metrics") or {}
    lines.append(
      "| "
      + " | ".join(
        [
          _format_metric(row.get("lane_id")),
          _format_metric(row.get("role")),
          _format_metric(row.get("backend")),
          _format_metric(metrics.get("success_rate")),
          _format_metric(metrics.get("mpkpe")),
          _format_metric(metrics.get("r_mpkpe")),
          _format_metric(metrics.get("joint_vel_error")),
          _format_metric(metrics.get("ee_pos_error")),
          _format_metric(metrics.get("ee_ori_error")),
          _format_metric(row.get("notes")),
        ]
      )
      + " |"
    )

  delta = _optional_mapping(snapshot, "local_minus_flashsac_delta") or {}
  if delta:
    lines.extend(
      [
        "",
        "## Local reproduced PPO minus FlashSAC reference delta",
        "",
        "| Metric | Delta |",
        "| --- | --- |",
      ]
    )
    for key in COMPARISON_METRIC_KEYS:
      if key in delta:
        lines.append(f"| {key} | {_format_metric(delta[key])} |")

  lines.extend(["", "## Acceptance notes", ""])
  for note in notes_value:
    lines.append(f"- {_format_metric(note)}")

  return "\n".join(lines) + "\n"


def build_lanev_protocol_bundle(
  repo_root: Path | None = None,
) -> dict[str, object]:
  return {
    "baseline_sha": TRACKING_BASELINE_SHA,
    "source_of_truth": asdict(source_of_truth_paths(repo_root)),
    "authoritative_ppo_gold_baseline": authoritative_ppo_gold_baseline(),
    "ppo_control_protocol": ppo_control_protocol(repo_root),
    "comparison_table_schema": comparison_table_schema(),
    "verification_checklist": verification_checklist(repo_root),
    "acceptance_rubric": acceptance_rubric(),
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Emit the LaneV FlashSAC verification protocol bundle derived from the "
      "team source-of-truth plan/spec artifacts."
    )
  )
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=None,
    help="Override the repository root used to resolve the source-of-truth paths.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional JSON output path. Prints to stdout when omitted.",
  )
  parser.add_argument(
    "--local-ppo-eval",
    type=Path,
    default=None,
    help="Optional JSON metrics from the local reproduced PPO eval path.",
  )
  parser.add_argument(
    "--flashsac-eval",
    type=Path,
    default=None,
    help="Optional JSON metrics from the current FlashSAC reference artifact.",
  )
  parser.add_argument(
    "--shared-path-differential",
    type=Path,
    default=None,
    help="Optional JSON summary describing the current shared-path differential diagnosis.",
  )
  parser.add_argument(
    "--comparison-output",
    type=Path,
    default=None,
    help="Optional JSON output path for the LaneV comparison snapshot.",
  )
  parser.add_argument(
    "--acceptance-output",
    type=Path,
    default=None,
    help="Optional Markdown output path for rendered LaneV acceptance notes.",
  )
  args = parser.parse_args(list(argv) if argv is not None else None)

  payload = build_lanev_protocol_bundle(args.repo_root)
  rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
  if args.output is None:
    print(rendered, end="")
  else:
    args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")

  def _load_json(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
      return None
    loaded = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
      raise TypeError(f"{path} must contain a JSON object")
    return loaded

  if args.comparison_output is not None or args.acceptance_output is not None:
    snapshot = build_lanev_comparison_snapshot(
      local_ppo_eval=_load_json(args.local_ppo_eval),
      flashsac_eval=_load_json(args.flashsac_eval),
      shared_path_differential=_load_json(args.shared_path_differential),
    )
    if args.comparison_output is not None:
      args.comparison_output.expanduser().resolve().write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
      )
    if args.acceptance_output is not None:
      args.acceptance_output.expanduser().resolve().write_text(
        render_lanev_acceptance_notes(snapshot),
        encoding="utf-8",
      )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
