"""
Validate converted motion data files.

Checks:
1. NPZ format compatibility with BVHLoader / robot_retarget_socp3.py
2. BVH file structural correctness
3. Data sanity (NaN, shape, range)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# NPZ validation (BVH-like format for SOCP3)
# ---------------------------------------------------------------------------
REQUIRED_NPZ_KEYS = {
    "positions", "rotations", "joint_names", "connections",
    "frame_time", "frame_count",
}

OPTIONAL_NPZ_KEYS = {"axis_up", "position_unit"}


def validate_bvhlike_npz(filepath: str) -> list[str]:
    """
    Validate a BVH-like NPZ file against the format expected by
    robot_retarget_socp3.py / BVHLoader.

    Returns a list of issues (empty = all good).
    """
    issues: list[str] = []
    path = Path(filepath)

    if not path.exists():
        return [f"File not found: {filepath}"]
    if path.suffix.lower() != ".npz":
        issues.append(f"Expected .npz extension, got {path.suffix}")

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        return [f"Cannot load NPZ: {e}"]

    keys = set(data.files)
    missing = REQUIRED_NPZ_KEYS - keys
    if missing:
        issues.append(f"Missing required keys: {missing}")
        return issues

    pos = data["positions"]
    rot = data["rotations"]
    names = data["joint_names"]
    conn = data["connections"]
    ft = float(data["frame_time"])
    fc = int(data["frame_count"])

    # Shape checks
    if pos.ndim != 3 or pos.shape[2] != 3:
        issues.append(f"positions shape should be (T, J, 3), got {pos.shape}")

    if rot.ndim != 3 or rot.shape[2] != 4:
        issues.append(f"rotations shape should be (T, J, 4), got {rot.shape}")

    if pos.shape[0] != rot.shape[0]:
        issues.append(f"Frame count mismatch: positions={pos.shape[0]}, rotations={rot.shape[0]}")

    if pos.shape[1] != rot.shape[1]:
        issues.append(f"Joint count mismatch: positions J={pos.shape[1]}, rotations J={rot.shape[1]}")

    T, J, _ = pos.shape

    if len(names) != J:
        issues.append(f"joint_names length ({len(names)}) != J ({J})")

    if fc != T:
        issues.append(f"frame_count ({fc}) != actual frames ({T})")

    if ft <= 0:
        issues.append(f"frame_time should be > 0, got {ft}")

    # NaN / Inf checks
    if np.any(np.isnan(pos)):
        issues.append("positions contains NaN")
    if np.any(np.isinf(pos)):
        issues.append("positions contains Inf")
    if np.any(np.isnan(rot)):
        issues.append("rotations contains NaN")
    if np.any(np.isinf(rot)):
        issues.append("rotations contains Inf")

    # Quaternion norm check (should be ~1)
    qnorms = np.linalg.norm(rot, axis=-1)
    if np.any(qnorms < 0.9) or np.any(qnorms > 1.1):
        bad_ratio = np.mean((qnorms < 0.9) | (qnorms > 1.1))
        issues.append(
            f"Quaternion norms out of range [0.9, 1.1]: "
            f"{bad_ratio*100:.1f}% of entries (min={qnorms.min():.3f}, max={qnorms.max():.3f})"
        )

    # Position magnitude check (should be in metres, typically < 5m)
    pos_max = np.abs(pos).max()
    if pos_max > 20:
        issues.append(
            f"Position values seem too large ({pos_max:.1f}). "
            f"Check if units are metres. BVHLoader auto-detects >20 as cm."
        )

    # Connections check
    if conn.ndim != 2 or conn.shape[1] != 2:
        issues.append(f"connections shape should be (E, 2), got {conn.shape}")
    else:
        if np.any(conn < 0) or np.any(conn >= J):
            issues.append(f"connections has out-of-range indices (J={J})")

    return issues


# ---------------------------------------------------------------------------
# BVH validation
# ---------------------------------------------------------------------------
def validate_bvh(filepath: str) -> list[str]:
    """
    Basic structural validation of a BVH file.

    Returns a list of issues (empty = all good).
    """
    issues: list[str] = []
    path = Path(filepath)

    if not path.exists():
        return [f"File not found: {filepath}"]

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return [f"Cannot read BVH: {e}"]

    lines = content.strip().split("\n")

    if not any(line.strip().upper().startswith("HIERARCHY") for line in lines[:5]):
        issues.append("Missing HIERARCHY section")

    motion_line = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("MOTION"):
            motion_line = i
            break

    if motion_line is None:
        issues.append("Missing MOTION section")
        return issues

    # Parse Frames and Frame Time
    frames_line = lines[motion_line + 1].strip() if motion_line + 1 < len(lines) else ""
    time_line = lines[motion_line + 2].strip() if motion_line + 2 < len(lines) else ""

    import re
    m_frames = re.search(r"Frames\s*:\s*(\d+)", frames_line, re.IGNORECASE)
    m_time = re.search(r"Frame\s*Time\s*:\s*([0-9.eE+-]+)", time_line, re.IGNORECASE)

    if not m_frames:
        issues.append(f"Cannot parse frame count from: {frames_line}")
    if not m_time:
        issues.append(f"Cannot parse frame time from: {time_line}")

    if m_frames and m_time:
        expected_frames = int(m_frames.group(1))
        data_lines = [l for l in lines[motion_line + 3:] if l.strip()]
        actual_frames = len(data_lines)
        if actual_frames != expected_frames:
            issues.append(
                f"Declared {expected_frames} frames but found {actual_frames} data lines"
            )

    # Count joints
    root_count = content.upper().count("ROOT ")
    joint_count = content.upper().count("JOINT ")
    total_joints = root_count + joint_count
    if total_joints == 0:
        issues.append("No ROOT or JOINT found in hierarchy")
    else:
        endsite_count = content.upper().count("END SITE")
        print(f"  BVH structure: {total_joints} joints, {endsite_count} end sites")

    return issues


# ---------------------------------------------------------------------------
# Processed SMPLX validation
# ---------------------------------------------------------------------------
def validate_processed_smplx(filepath: str) -> list[str]:
    """Validate processed SMPLX NPZ (global_joint_positions + height)."""
    issues: list[str] = []

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        return [f"Cannot load NPZ: {e}"]

    if "global_joint_positions" not in data.files:
        issues.append("Missing 'global_joint_positions' key")
        return issues

    gjp = data["global_joint_positions"]
    if gjp.ndim != 3 or gjp.shape[2] != 3:
        issues.append(f"global_joint_positions should be (T, J, 3), got {gjp.shape}")

    if np.any(np.isnan(gjp)):
        issues.append("global_joint_positions contains NaN")

    if "height" in data.files:
        h = float(data["height"])
        if h < 0.5 or h > 2.5:
            issues.append(f"height ({h:.3f}m) outside plausible range [0.5, 2.5]")
    else:
        issues.append("Missing 'height' key (optional but recommended)")

    return issues


# ---------------------------------------------------------------------------
# Auto-detect and validate
# ---------------------------------------------------------------------------
def validate_file(filepath: str, verbose: bool = True) -> bool:
    """
    Auto-detect file type and validate.

    Returns True if valid, False otherwise.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if verbose:
        print(f"\nValidating: {filepath}")
        print("=" * 60)

    if suffix == ".bvh":
        issues = validate_bvh(filepath)
        fmt_name = "BVH"
    elif suffix == ".npz":
        try:
            data = np.load(filepath, allow_pickle=True)
            keys = set(data.files)
        except Exception as e:
            if verbose:
                print(f"  FAIL: Cannot load: {e}")
            return False

        if "positions" in keys and "rotations" in keys:
            issues = validate_bvhlike_npz(filepath)
            fmt_name = "BVH-like NPZ"
        elif "global_joint_positions" in keys:
            issues = validate_processed_smplx(filepath)
            fmt_name = "Processed SMPLX NPZ"
        elif "poses" in keys and "betas" in keys:
            if verbose:
                n_pose = data["poses"].shape[-1]
                print(f"  Format: Raw AMASS NPZ (poses dim={n_pose})")
                print(f"  Keys: {sorted(data.files)}")
                print(f"  Frames: {data['poses'].shape[0]}")
                print("  Status: This is a source file, not yet converted.")
            return True
        else:
            if verbose:
                print(f"  Unknown NPZ format. Keys: {sorted(data.files)}")
            return False
    else:
        if verbose:
            print(f"  Unsupported file type: {suffix}")
        return False

    if verbose:
        print(f"  Format: {fmt_name}")

    if not issues:
        if verbose:
            _print_summary(filepath, fmt_name)
            print("  Status: PASS - All checks passed!")
        return True
    else:
        if verbose:
            for issue in issues:
                print(f"  ISSUE: {issue}")
            print(f"  Status: FAIL - {len(issues)} issue(s) found")
        return False


def _print_summary(filepath: str, fmt_name: str):
    """Print a summary of the file contents."""
    if fmt_name == "BVH-like NPZ":
        data = np.load(filepath, allow_pickle=True)
        pos = data["positions"]
        T, J, _ = pos.shape
        print(f"  Frames: {T}, Joints: {J}")
        print(f"  Frame time: {float(data['frame_time']):.4f}s "
              f"({1/float(data['frame_time']):.1f} fps)")
        print(f"  Joint names: {list(data['joint_names'])}")
        print(f"  Position range: [{pos.min():.3f}, {pos.max():.3f}]")
        if "axis_up" in data.files:
            print(f"  Axis up: {data['axis_up']}")

    elif fmt_name == "Processed SMPLX NPZ":
        data = np.load(filepath, allow_pickle=True)
        gjp = data["global_joint_positions"]
        T, J, _ = gjp.shape
        print(f"  Frames: {T}, Joints: {J}")
        if "height" in data.files:
            print(f"  Height: {float(data['height']):.3f}m")

    elif fmt_name == "BVH":
        with open(filepath) as f:
            content = f.read()
        lines = content.strip().split("\n")
        data_lines = 0
        in_motion = False
        for l in lines:
            if l.strip().upper().startswith("MOTION"):
                in_motion = True
                continue
            if in_motion and l.strip() and not l.strip().upper().startswith("FRAME"):
                data_lines += 1
        print(f"  Motion frames: ~{data_lines}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate converted motion data files"
    )
    parser.add_argument("files", nargs="+", help="Files to validate (.npz or .bvh)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print PASS/FAIL")

    args = parser.parse_args()

    all_pass = True
    for filepath in args.files:
        ok = validate_file(filepath, verbose=not args.quiet)
        if not ok:
            all_pass = False
        if args.quiet:
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {filepath}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
