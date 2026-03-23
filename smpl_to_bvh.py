"""
Convert SMPL / SMPL-X / SMPL+H motion data to standard BVH format.

Supported inputs:
    - Raw AMASS .npz (poses, betas, trans, mocap_frame_rate …)
    - Processed SMPLX .npz (global_joint_positions)
    - BVH-like .npz (positions, rotations, joint_names, connections)

The BVH output can be loaded by any standard BVH viewer (Blender, MotionBuilder,
bvhacker, …) or further converted to BVH-like NPZ via bvh2npz.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from smpl_to_npz import (
    SMPL_22_TPOSE_OFFSETS,
    SMPLX_22_JOINT_NAMES,
    SMPLX_22_PARENTS,
    load_and_convert,
)


# BVH uses centimetre offsets (SMPL_22_TPOSE_OFFSETS is in metres, Y-up)
SMPLX_22_OFFSETS_CM = SMPL_22_TPOSE_OFFSETS * 100.0


def _global_to_local_rotations(
    global_quats: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    """
    Convert global quaternions to local (parent-relative) quaternions.

    Args:
        global_quats: (T, J, 4) quaternions in xyzw (scipy) convention
        parents: list of parent indices, -1 for root

    Returns:
        (T, J, 4) local quaternions (xyzw)
    """
    T, J, _ = global_quats.shape
    local = np.zeros_like(global_quats)

    for t in range(T):
        for j in range(J):
            p = parents[j]
            g = R.from_quat(global_quats[t, j])
            if p < 0:
                local[t, j] = g.as_quat()
            else:
                gp = R.from_quat(global_quats[t, p])
                loc = gp.inv() * g
                local[t, j] = loc.as_quat()

    return local


def _euler_from_quat(q_xyzw: np.ndarray, order: str = "ZYX") -> np.ndarray:
    """(4,) xyzw quaternion → (3,) Euler angles in degrees."""
    return R.from_quat(q_xyzw).as_euler(order, degrees=True)


def write_bvh(
    filepath: str,
    joint_names: list[str],
    parents: list[int],
    offsets_cm: np.ndarray,
    positions: np.ndarray,
    local_quats: np.ndarray,
    frame_time: float,
    euler_order: str = "ZYX",
):
    """
    Write a BVH file.

    Args:
        filepath    : output .bvh path
        joint_names : (J,) joint names
        parents     : parent indices (-1 = root)
        offsets_cm  : (J, 3) rest-pose offsets in centimetres
        positions   : (T, J, 3) global positions (only root translation used)
        local_quats : (T, J, 4) local quaternions (xyzw)
        frame_time  : seconds per frame
        euler_order : Euler angle convention for channels
    """
    T, J, _ = positions.shape

    children = [[] for _ in range(J)]
    for c, p in enumerate(parents):
        if p >= 0:
            children[p].append(c)

    lines: list[str] = ["HIERARCHY"]

    def _write_joint(j: int, depth: int, is_root: bool = False):
        indent = "\t" * depth
        tag = "ROOT" if is_root else "JOINT"
        lines.append(f"{indent}{tag} {joint_names[j]}")
        lines.append(f"{indent}{{")

        off = offsets_cm[j]
        lines.append(f"{indent}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")

        if is_root:
            ch_str = f"Xposition Yposition Zposition {euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation"
            lines.append(f"{indent}\tCHANNELS 6 {ch_str}")
        else:
            ch_str = f"{euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation"
            lines.append(f"{indent}\tCHANNELS 3 {ch_str}")

        if not children[j]:
            lines.append(f"{indent}\tEnd Site")
            lines.append(f"{indent}\t{{")
            lines.append(f"{indent}\t\tOFFSET 0.000000 5.000000 0.000000")
            lines.append(f"{indent}\t}}")
        else:
            for c in children[j]:
                _write_joint(c, depth + 1)

        lines.append(f"{indent}}}")

    root_idx = parents.index(-1)
    _write_joint(root_idx, 0, is_root=True)

    lines.append("MOTION")
    lines.append(f"Frames: {T}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    root_pos_cm = positions[:, root_idx, :] * 100.0

    for t in range(T):
        vals: list[float] = []
        for j in range(J):
            if j == root_idx:
                vals.extend(root_pos_cm[t].tolist())
            euler = _euler_from_quat(local_quats[t, j], euler_order)
            vals.extend(euler.tolist())
        lines.append(" ".join(f"{v:.6f}" for v in vals))

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def convert_to_bvh(
    input_path: str,
    output_path: str | None = None,
    model_path: str | None = None,
    model_type: str | None = None,
    target_fps: int = 30,
    use_bodymodel: bool = False,
) -> str:
    """
    High-level API: convert any supported source to BVH.

    Returns the output BVH path.
    """
    output_path = output_path or str(Path(input_path).with_suffix(".bvh"))

    npz_data = load_and_convert(
        input_path=input_path,
        model_path=model_path,
        model_type=model_type,
        target_fps=target_fps,
        num_joints=22,
        use_bodymodel=use_bodymodel,
        to_z_up=False,  # BVH is typically Y-up
    )

    positions = npz_data["positions"]
    rotations = npz_data["rotations"]
    joint_names = list(npz_data["joint_names"])
    frame_time = float(npz_data["frame_time"])

    J = len(joint_names)
    parents = SMPLX_22_PARENTS[:J]
    offsets = SMPLX_22_OFFSETS_CM[:J]

    local_quats = _global_to_local_rotations(rotations, parents)

    write_bvh(
        filepath=output_path,
        joint_names=joint_names,
        parents=parents,
        offsets_cm=offsets,
        positions=positions,
        local_quats=local_quats,
        frame_time=frame_time,
    )

    print(f"Saved BVH: {output_path}")
    print(f"  Joints: {J}, Frames: {positions.shape[0]}, FPS: {1/frame_time:.1f}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL/SMPLX motion data to BVH format"
    )
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .bvh path (default: <input>.bvh)")
    parser.add_argument("--model_path", default=None,
                        help="Path to SMPL model directory")
    parser.add_argument("--model_type", default=None,
                        choices=["smpl", "smplh", "smplx"],
                        help="Body model type (auto-detected if not specified)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--use_bodymodel", action="store_true",
                        help="Use official body model for FK")

    args = parser.parse_args()

    convert_to_bvh(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model_path,
        model_type=args.model_type,
        target_fps=args.fps,
        use_bodymodel=args.use_bodymodel,
    )


if __name__ == "__main__":
    main()
