"""Tests for project console-script entry points."""

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def test_project_scripts_expose_retarget_pipeline_commands() -> None:
  """pyproject should expose the retarget pipeline scripts for uv run."""
  pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
  scripts = pyproject["project"]["scripts"]

  assert scripts["raw-human-npz-to-keypoints"] == (
    "mjlab.scripts.raw_human_npz_to_smpl_keypoints:main"
  )
  assert scripts["pyroki-npz-to-csv"] == "mjlab.scripts.pyroki_npz_to_csv:main"
  assert scripts["smpl-keypoints-to-g1-npz"] == (
    "mjlab.scripts.smpl_keypoints_to_g1_npz:main"
  )
  assert scripts["build-tracking-motion"] == "mjlab.scripts.build_tracking_motion:main"
  assert scripts["report-and-preview-motion"] == (
    "mjlab.scripts.report_and_preview_motion:cli"
  )
