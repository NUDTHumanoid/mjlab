from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from mjlab.scripts.flashsac_lane0_manifest import (
  DEFAULT_BASELINE_SHA,
  ManifestConfig,
  generate_lane0_manifests,
)

ROOT = Path(
  os.environ.get("OMX_TEAM_LEADER_CWD", Path(__file__).resolve().parent.parent)
).resolve()


def test_generate_lane0_manifests_writes_expected_artifacts(tmp_path: Path) -> None:
  outputs = generate_lane0_manifests(
    ManifestConfig(
      leader_cwd=str(ROOT),
      output_dir=str(tmp_path),
    )
  )

  shared_manifest = json.loads(
    outputs["shared_manifest_path"].read_text(encoding="utf-8")
  )
  baseline_provenance = json.loads(
    outputs["baseline_provenance_path"].read_text(encoding="utf-8")
  )
  ppo_control = json.loads(
    outputs["ppo_control_manifest_path"].read_text(encoding="utf-8")
  )

  assert shared_manifest["baseline"]["required_sha"] == DEFAULT_BASELINE_SHA
  assert shared_manifest["baseline"]["pinned_sha"] == DEFAULT_BASELINE_SHA
  assert shared_manifest["baseline"]["repo_root"] == str(ROOT)
  assert shared_manifest["commands"]["lane0_refresh"].startswith(
    'cd "$OMX_TEAM_LEADER_CWD" && uv run flashsac-lane0-manifest'
  )
  assert shared_manifest["commands"]["by_lane"]["laneA-parity"][
    "smoke_train"
  ].startswith('cd "$OMX_TEAM_LEADER_CWD" && uv run train')
  assert (
    "--backend flashsac"
    in shared_manifest["commands"]["by_lane"]["laneA-parity"]["smoke_train"]
  )
  assert (
    "--wandb-run-path"
    in shared_manifest["commands"]["by_lane"]["laneV-ppo-control"]["smoke_eval"]
  )
  assert [gate["env_steps"] for gate in shared_manifest["fail_fast_gates"]] == [
    0,
    50_000,
    100_000,
    250_000,
  ]
  assert (
    Path(shared_manifest["artifact_schema"]["comparison_table_template"])
    == outputs["comparison_template_path"]
  )

  assert baseline_provenance["baseline"]["required_sha"] == DEFAULT_BASELINE_SHA
  assert len(baseline_provenance["pinned_hashes"]["flashsac_eval_contract_hash"]) == 64
  assert (
    shared_manifest["contract_files"]["flashsac_eval"]["git_ref"]
    == DEFAULT_BASELINE_SHA
  )

  assert ppo_control["lane"] == "laneV-ppo-control"
  assert ppo_control["backend"] == "ppo"
  assert "smoke_train" in ppo_control["commands"]

  with outputs["comparison_template_path"].open(
    "r", encoding="utf-8", newline=""
  ) as handle:
    header = next(csv.reader(handle))
  assert header[:5] == [
    "lane",
    "backend",
    "hypothesis",
    "baseline_sha",
    "head_sha",
  ]
