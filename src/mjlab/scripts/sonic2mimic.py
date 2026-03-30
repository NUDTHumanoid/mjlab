#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Sequence, Tuple


POSITION_SCALE = 0.01
EULER_ORDER = "ZYX"


JOINT_COLUMNS = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]

REQUIRED_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    *JOINT_COLUMNS,
]


def quat_mul_xyzw(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return (x, y, z, w)


def quat_from_axis_angle_deg(axis: str, angle_deg: float) -> Tuple[float, float, float, float]:
    angle = math.radians(angle_deg)
    s = math.sin(angle * 0.5)
    c = math.cos(angle * 0.5)
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    if axis == "z":
        return (0.0, 0.0, s, c)
    raise ValueError(f"Unsupported axis: {axis}")


def normalize_quat_xyzw(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm quaternion.")
    return (x / norm, y / norm, z / norm, w / norm)


def euler_degrees_to_quat_xyzw(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> Tuple[float, float, float, float]:
    """Convert Euler angles in degrees to quaternion (x, y, z, w)."""
    quats = {
        "X": quat_from_axis_angle_deg("x", rx_deg),
        "Y": quat_from_axis_angle_deg("y", ry_deg),
        "Z": quat_from_axis_angle_deg("z", rz_deg),
    }

    q = (0.0, 0.0, 0.0, 1.0)
    for axis in EULER_ORDER:
        q = quat_mul_xyzw(q, quats[axis])
    return normalize_quat_xyzw(q)


def default_single_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_mimic{input_path.suffix}")


def collect_csv_files(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*.csv")
        if p.is_file() and not p.name.endswith("_mimic.csv")
    )


def ensure_required_columns(fieldnames: Sequence[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def resolve_batch_output(input_root: Path, output_root: Path, input_file: Path) -> Path:
    relative = input_file.relative_to(input_root)
    out_name = f"{input_file.stem}_mimic{input_file.suffix}"
    return output_root / relative.parent / out_name


def convert_row(row: dict, z_offset: float) -> List[float]:
    tx = float(row["root_translateX"]) * POSITION_SCALE
    ty = float(row["root_translateY"]) * POSITION_SCALE
    tz = float(row["root_translateZ"]) * POSITION_SCALE + z_offset

    rx = float(row["root_rotateX"])
    ry = float(row["root_rotateY"])
    rz = float(row["root_rotateZ"])

    qx, qy, qz, qw = euler_degrees_to_quat_xyzw(rx, ry, rz)

    joints_rad = [math.radians(float(row[col])) for col in JOINT_COLUMNS]

    return [tx, ty, tz, qx, qy, qz, qw, *joints_rad]


def convert_file(input_file: Path, output_file: Path, z_offset: float) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    converted_rows = 0
    with input_file.open("r", encoding="utf-8-sig", newline="") as f_in, \
         output_file.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to be empty or missing a header row.")
        ensure_required_columns(reader.fieldnames)

        writer = csv.writer(f_out)
        for row in reader:
            if not row:
                continue
            writer.writerow(convert_row(row, z_offset=z_offset))
            converted_rows += 1

    return converted_rows


def process_single(input_path: Path, output_path: Path | None, z_offset: float) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a .csv file: {input_path}")

    final_output = output_path if output_path is not None else default_single_output(input_path)
    row_count = convert_file(input_path, final_output, z_offset=z_offset)
    print(
        f"[OK] {input_path} -> {final_output} "
        f"({row_count} rows, euler_order={EULER_ORDER}, pos_scale={POSITION_SCALE}, z_offset={z_offset})"
    )


def process_batch(input_dir: Path, output_dir: Path | None, z_offset: float) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Batch input must be a directory: {input_dir}")

    final_output_dir = output_dir if output_dir is not None else input_dir
    files = collect_csv_files(input_dir)
    if not files:
        print(f"[INFO] No CSV files found in: {input_dir}")
        return

    total_rows = 0
    for src in files:
        dst = resolve_batch_output(input_dir, final_output_dir, src)
        row_count = convert_file(src, dst, z_offset=z_offset)
        total_rows += row_count
        print(f"[OK] {src} -> {dst} ({row_count} rows)")

    print(
        f"[DONE] Converted {len(files)} files, total rows: {total_rows}, "
        f"euler_order={EULER_ORDER}, pos_scale={POSITION_SCALE}, z_offset={z_offset}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SONIC/BONES-style CSV files into BeyondMimic-compatible CSV files. "
            "Defaults assume root rotations are Euler angles in degrees, joints are in degrees. "
            f"Root translations are scaled by a fixed factor of {POSITION_SCALE}. "
            f"Root rotations use a fixed Euler composition order of {EULER_ORDER}."
        )
    )
    parser.add_argument("--inputs", type=str, help="Single input CSV file path.")
    parser.add_argument(
        "--outputs",
        type=str,
        help="Single output CSV file path. If omitted, writes next to input with suffix _mimic.",
    )
    parser.add_argument("--inputm", type=str, help="Batch input directory.")
    parser.add_argument(
        "--outputm",
        type=str,
        help="Batch output directory. If omitted, writes into input directory with suffix _mimic.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Add a constant offset to root_translateZ after scaling. Default: 0.0.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    using_single = args.inputs is not None
    using_batch = args.inputm is not None

    if using_single == using_batch:
        parser.error("Use exactly one mode: (--inputs [--outputs]) or (--inputm [--outputm]).")

    if using_single:
        process_single(
            Path(args.inputs),
            Path(args.outputs) if args.outputs else None,
            z_offset=args.z_offset,
        )
    else:
        process_batch(
            Path(args.inputm),
            Path(args.outputm) if args.outputm else None,
            z_offset=args.z_offset,
        )


if __name__ == "__main__":
    main()
