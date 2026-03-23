"""
SMPL/SMPL+H/SMPL-X → BVH-like NPZ converter for retargeting.

Output keys: positions (T,J,3), rotations (T,J,4), joint_names, connections,
             frame_time, frame_count, axis_up
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Skeleton definitions
# ---------------------------------------------------------------------------
SMPLX_22_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

LAFAN_22_JOINT_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToe", "RightToe",
    "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
]

SMPLX_22_PARENTS = [
    -1, 0, 0, 0, 1, 2,
     3, 4, 5, 6, 7, 8,
     9, 9, 9, 12, 13, 14,
    16, 17, 18, 19,
]

SMPL_24_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

SMPL_24_PARENTS = [
    -1, 0, 0, 0, 1, 2,
     3, 4, 5, 6, 7, 8,
     9, 9, 9, 12, 13, 14,
    16, 17, 18, 19, 20, 21,
]

# Neutral-body (betas=0) T-pose offsets, Y-up, metres.
# Fallback when body model is unavailable.
SMPL_22_TPOSE_OFFSETS = np.array([
    [ 0.0000,  0.0000,  0.0000],
    [ 0.0591, -0.0823, -0.0102],
    [-0.0591, -0.0823, -0.0102],
    [ 0.0028,  0.2139, -0.0228],
    [ 0.0434, -0.4129, -0.0156],
    [-0.0434, -0.4129, -0.0156],
    [ 0.0030,  0.2164,  0.0200],
    [ 0.0167, -0.4112, -0.0077],
    [-0.0167, -0.4112, -0.0077],
    [-0.0069,  0.2813, -0.0152],
    [ 0.0124, -0.1104,  0.1478],
    [-0.0124, -0.1104,  0.1478],
    [ 0.0010,  0.2260, -0.0136],
    [ 0.0724,  0.1360, -0.0289],
    [-0.0724,  0.1360, -0.0289],
    [-0.0011,  0.1073,  0.0108],
    [ 0.1669,  0.0303, -0.0152],
    [-0.1669,  0.0303, -0.0152],
    [ 0.2570, -0.0268, -0.0134],
    [-0.2570, -0.0268, -0.0134],
    [ 0.2526,  0.0178,  0.0076],
    [-0.2526,  0.0178,  0.0076],
], dtype=np.float64)


def _build_connections(parents: list[int]) -> np.ndarray:
    edges = []
    for child, parent in enumerate(parents):
        if parent >= 0:
            edges.append([parent, child])
    return np.array(edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    shape = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    mats = R.from_rotvec(flat).as_matrix()
    return mats.reshape(*shape, 3, 3)


def fk_smpl(
    poses_aa: np.ndarray,
    trans: np.ndarray,
    parents: list[int],
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T, J, _ = poses_aa.shape
    local_rot = _axis_angle_to_matrix(poses_aa)

    positions = np.zeros((T, J, 3))
    global_rot = np.zeros((T, J, 3, 3))

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                global_rot[t, j] = local_rot[t, j]
                positions[t, j] = trans[t] + offsets[j]
            else:
                global_rot[t, j] = global_rot[t, p] @ local_rot[t, j]
                positions[t, j] = positions[t, p] + global_rot[t, p] @ offsets[j]

    quats = np.zeros((T, J, 4))
    for t in range(T):
        for j in range(J):
            quats[t, j] = R.from_matrix(global_rot[t, j]).as_quat()

    return positions, quats


# ---------------------------------------------------------------------------
# Body model FK (loads .npz weights directly, no torch needed)
# ---------------------------------------------------------------------------
def _resolve_body_model_path(model_root: str, model_type: str, gender: str) -> str:
    mt = model_type.lower()
    g = gender.lower()
    candidates = [
        os.path.join(model_root, mt, f"{mt.upper()}_{g.upper()}.npz"),
        os.path.join(model_root, mt, g, "model.npz"),
        os.path.join(model_root, mt, f"{mt.upper()}_{g.upper()}.pkl"),
        os.path.join(model_root, mt, f"{mt.upper()}_NEUTRAL.npz"),
        os.path.join(model_root, mt, "neutral", "model.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {model_type} body model for gender '{gender}' under {model_root}.\n"
        f"Tried: " + ", ".join(candidates)
    )


def fk_smplx_bodymodel(
    npz_path: str,
    model_path: str,
    model_type: str = "smplx",
    num_joints: int = 22,
    target_fps: int = 30,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(npz_path, allow_pickle=True)
    ori_fps = float(data.get("mocap_frame_rate", 120))
    if target_fps <= 0:
        target_fps = 30
    ds = max(1, int(ori_fps / target_fps))

    gender = str(data["gender"])
    if hasattr(gender, "item"):
        gender = gender.item()

    poses = data["poses"][::ds]
    trans_np = data["trans"][::ds]
    betas = data["betas"]

    T = poses.shape[0]
    n_pose_joints = poses.shape[1] // 3
    aa_all = poses.reshape(T, n_pose_joints, 3)

    J = min(num_joints, n_pose_joints, 22)
    aa = aa_all[:, :J, :]

    num_betas = int(data.get("num_betas", 16))
    if betas.ndim == 1:
        betas = betas[:num_betas]
    else:
        betas = betas[0, :num_betas]

    bm_fname = _resolve_body_model_path(model_path, model_type, gender)
    print(f"Body model: {bm_fname}  (gender={gender})")
    bm = np.load(bm_fname, allow_pickle=True)

    v_template = bm["v_template"]
    shapedirs = bm["shapedirs"]
    J_regressor = bm["J_regressor"]
    kintree = bm["kintree_table"]

    nb = min(betas.shape[0], shapedirs.shape[2])
    v_shaped = v_template + np.einsum("vcd,d->vc", shapedirs[:, :, :nb], betas[:nb])

    if hasattr(J_regressor, "toarray"):
        J_all = J_regressor.toarray() @ v_shaped
    else:
        J_all = J_regressor @ v_shaped

    parents_model = kintree[0].astype(int).tolist()
    parents_model[0] = -1
    parents = parents_model[:J]

    offsets = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        p = parents[j]
        offsets[j] = J_all[j] if p < 0 else J_all[j] - J_all[p]

    positions, quats = fk_smpl(aa, trans_np, parents, offsets)

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    bad = (norms < 1e-10) | ~np.isfinite(norms) | ~np.isfinite(quats).all(axis=-1, keepdims=True)
    quats = np.where(bad, np.array([0, 0, 0, 1], dtype=quats.dtype), quats)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12

    return positions, quats, 1.0 / target_fps


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _y_up_to_z_up(positions):
    out = positions.copy()
    out[..., 1], out[..., 2] = positions[..., 2].copy(), positions[..., 1].copy()
    return out

def _z_up_to_y_up(positions):
    return _y_up_to_z_up(positions)

def _detect_up_axis(positions):
    extents = positions[0].max(axis=0) - positions[0].min(axis=0)
    return int(np.argmax(extents))

def _normalize_to_z_up(positions):
    up = _detect_up_axis(positions)
    if up == 2: return positions
    if up == 1: return _y_up_to_z_up(positions)
    out = positions.copy()
    out[..., 0], out[..., 2] = positions[..., 2].copy(), positions[..., 0].copy()
    return out

def _normalize_to_y_up(positions):
    up = _detect_up_axis(positions)
    if up == 1: return positions
    if up == 2: return _z_up_to_y_up(positions)
    out = positions.copy()
    out[..., 0], out[..., 1] = positions[..., 1].copy(), positions[..., 0].copy()
    return out

def _normalize_height(positions, reference_cm=170.0):
    up_idx = _detect_up_axis(positions)
    extent_m = float(positions[0, :, up_idx].max() - positions[0, :, up_idx].min())
    if extent_m < 0.1:
        return positions
    scale = (reference_cm / 100.0) / extent_m
    return positions * scale

def _centre_root(positions, z_up=True):
    pos = positions.copy()
    if z_up:
        pos[:, :, :2] -= pos[0, 0, :2]
    else:
        pos[:, :, [0, 2]] -= pos[0, 0, [0, 2]]
    return pos


# ---------------------------------------------------------------------------
# G1 bone scaling + hip levelling
# ---------------------------------------------------------------------------
def _get_all_descendants(node, children_map):
    result = [node]
    stack = [node]
    while stack:
        n = stack.pop()
        for c in children_map[n]:
            result.append(c)
            stack.append(c)
    return result


# Legs → LAFAN targets; Arms → G1 robot targets (solver-frame metres)
_SCALING_CHAINS = [
    (["Hips", "LeftUpLeg"],                     0.080),
    (["Hips", "RightUpLeg"],                    0.080),
    (["LeftUpLeg",  "LeftLeg"],                 0.325),
    (["RightUpLeg", "RightLeg"],                0.325),
    (["LeftLeg",    "LeftFoot"],                0.317),
    (["RightLeg",   "RightFoot"],               0.317),
    (["Spine2", "LeftShoulder",  "LeftArm"],    0.273824),
    (["Spine2", "RightShoulder", "RightArm"],   0.273819),
    (["LeftArm",    "LeftForeArm"],             0.184500),
    (["RightArm",   "RightForeArm"],            0.184500),
    (["LeftForeArm","LeftHand"],                0.184281),
    (["RightForeArm","RightHand"],              0.184281),
]


def _rescale_bones(positions, parents, bvh_scale=0.747):
    T, J, _ = positions.shape
    if J < 22:
        return positions

    name2idx = {n: i for i, n in enumerate(LAFAN_22_JOINT_NAMES)}
    children = [[] for _ in range(J)]
    for j in range(J):
        if 0 <= parents[j] < J:
            children[parents[j]].append(j)

    chains = [([name2idx[n] for n in names], tgt) for names, tgt in _SCALING_CHAINS]

    pos0 = positions[0]
    ratios = []
    for idxs, target_m in chains:
        target_here = target_m / bvh_scale
        cur = sum(np.linalg.norm(pos0[idxs[k+1]] - pos0[idxs[k]])
                  for k in range(len(idxs) - 1))
        ratios.append(target_here / cur if cur > 1e-8 else 1.0)

    result = positions.copy()
    for t in range(T):
        for (idxs, _), ratio in zip(chains, ratios):
            for k in range(len(idxs) - 1):
                p_idx, c_idx = idxs[k], idxs[k + 1]
                vec = result[t, c_idx] - result[t, p_idx]
                disp = vec * (ratio - 1.0)
                for d in _get_all_descendants(c_idx, children):
                    result[t, d] += disp
    return result


def _level_hips(positions, parents):
    T, J, _ = positions.shape
    if J < 22:
        return positions

    idx_hips = LAFAN_22_JOINT_NAMES.index("Hips")
    idx_lul  = LAFAN_22_JOINT_NAMES.index("LeftUpLeg")
    idx_rul  = LAFAN_22_JOINT_NAMES.index("RightUpLeg")
    up = _detect_up_axis(positions)

    children = [[] for _ in range(J)]
    for j in range(J):
        if 0 <= parents[j] < J:
            children[parents[j]].append(j)

    left_desc  = _get_all_descendants(idx_lul, children)
    right_desc = _get_all_descendants(idx_rul, children)

    result = positions.copy()
    for t in range(T):
        hips_up = result[t, idx_hips, up]
        for j in left_desc:
            result[t, j, up] += hips_up - result[t, idx_lul, up]
        for j in right_desc:
            result[t, j, up] += hips_up - result[t, idx_rul, up]
    return result


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def detect_format(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    if "global_joint_positions" in keys:
        return "processed_smplx"
    if "poses" in keys and ("betas" in keys or "trans" in keys):
        n_pose = data["poses"].shape[-1]
        if n_pose >= 165: return "amass_smplx"
        if n_pose >= 156: return "amass_smplh"
        return "amass_smpl"
    if "positions" in keys and "rotations" in keys:
        return "bvh_npz"
    raise ValueError(f"Cannot detect format of {npz_path}. Keys: {sorted(keys)}")


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def load_and_convert(
    input_path: str,
    model_path: str | None = None,
    model_type: str | None = None,
    target_fps: int = 30,
    num_joints: int = 22,
    use_bodymodel: bool = False,
    to_z_up: bool = True,
    use_lafan_names: bool = False,
    for_bvh_loader: bool = False,
    scale_for_g1: bool = False,
    bvh_scale: float = 0.747,
) -> dict:
    if for_bvh_loader:
        use_lafan_names = True
        to_z_up = False

    fmt = detect_format(input_path)

    if model_type is None:
        model_type = {"amass_smplx": "smplx", "amass_smplh": "smplh",
                      "amass_smpl": "smpl"}.get(fmt, "smplx")

    if fmt == "processed_smplx":
        return _load_processed_smplx(input_path, target_fps, to_z_up)

    if fmt == "bvh_npz":
        return dict(np.load(input_path, allow_pickle=True))

    if use_bodymodel:
        if model_path is None:
            raise ValueError("--model_path required for body-model FK")
        positions, quats, frame_time = fk_smplx_bodymodel(
            input_path, model_path, model_type, num_joints, target_fps
        )
    else:
        positions, quats, frame_time = _fk_from_amass(
            input_path, fmt, target_fps, num_joints
        )

    if for_bvh_loader and scale_for_g1:
        positions = _normalize_to_y_up(positions)

        # Post-FK X↔Z swap to match LAFAN convention in BVHLoader
        positions = positions[:, :, [2, 1, 0]]

        J = min(positions.shape[1], 22)
        positions = positions[:, :J, :]
        T = positions.shape[0]

        up_extent = float(positions[0, :, 1].max() - positions[0, :, 1].min())
        if up_extent > 0.1:
            positions *= 1.70 / up_extent

        p_list = SMPLX_22_PARENTS[:J]
        positions = _level_hips(positions, p_list)
        positions = _rescale_bones(positions, p_list, bvh_scale=bvh_scale)

        # Toe → directly below ankle (matches G1 ankle_roll geometry)
        _ANKLE_OFFSET_M = 0.017558
        offset_here = _ANKLE_OFFSET_M / bvh_scale
        for fn_foot, fn_toe in [("LeftFoot", "LeftToe"), ("RightFoot", "RightToe")]:
            fi = LAFAN_22_JOINT_NAMES.index(fn_foot)
            ti = LAFAN_22_JOINT_NAMES.index(fn_toe)
            positions[:, ti, :] = positions[:, fi, :]
            positions[:, ti, 1] -= offset_here

        positions *= 100.0
        positions[:, :, 1] -= positions[0, :, 1].min()   # ground align
        positions[:, :, [0, 2]] -= positions[0, 0, [0, 2]]

        joint_names = np.array(LAFAN_22_JOINT_NAMES[:J])
        quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, J, 1))

        # LAFAN rest-pose foot quaternions (flat-foot orientation target)
        _LAFAN_FOOT = {
            "LeftToe":  np.array([-0.002340, +0.615327, -0.017666, -0.788070]),
            "RightToe": np.array([-0.038694, -0.867183, +0.032181, +0.495439]),
        }
        for fn, fq in _LAFAN_FOOT.items():
            quats[:, LAFAN_22_JOINT_NAMES.index(fn)] = fq

        parents = SMPLX_22_PARENTS[:J]
        connections = _build_connections(parents)
        axis_up = "y"

    elif for_bvh_loader:
        positions = _normalize_to_y_up(positions)
        positions = _normalize_height(positions, reference_cm=170.0)
        positions *= 100.0
        positions = _centre_root(positions, z_up=False)

        J = min(positions.shape[1], 22)
        positions = positions[:, :J, :]
        T = positions.shape[0]

        joint_names = np.array(LAFAN_22_JOINT_NAMES[:J])
        quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, J, 1))
        parents = SMPLX_22_PARENTS[:J]
        connections = _build_connections(parents)
        axis_up = "y"

    else:
        if to_z_up:
            positions = _normalize_to_z_up(positions)
        positions = _centre_root(positions, z_up=to_z_up)

        J = min(positions.shape[1], 22)
        positions = positions[:, :J, :]
        quats = quats[:, :J, :]

        if use_lafan_names and J <= 22:
            joint_names = np.array(LAFAN_22_JOINT_NAMES[:J])
        elif J <= 22:
            joint_names = np.array(SMPLX_22_JOINT_NAMES[:J])
        else:
            joint_names = np.array(SMPL_24_JOINT_NAMES[:J])

        parents = SMPLX_22_PARENTS[:J] if J <= 22 else SMPL_24_PARENTS[:J]
        connections = _build_connections(parents)
        axis_up = "z" if to_z_up else "y"

    return {
        "positions": positions.astype(np.float32),
        "rotations": quats.astype(np.float32),
        "joint_names": joint_names,
        "connections": connections,
        "frame_time": frame_time,
        "frame_count": positions.shape[0],
        "axis_up": axis_up,
    }


def load_and_convert_for_opt(
    input_path: str,
    model_path: str | None = None,
    model_type: str | None = None,
    target_fps: int = 30,
    num_joints: int = 22,
    use_bodymodel: bool = False,
) -> dict:
    bvh_data = load_and_convert(
        input_path=input_path, model_path=model_path, model_type=model_type,
        target_fps=target_fps, num_joints=num_joints,
        use_bodymodel=use_bodymodel, to_z_up=True,
    )
    positions = bvh_data["positions"]
    height = float(positions[0, :, 2].max() - positions[0, :, 2].min())
    if height < 0.5:
        height = 1.7
    return {"global_joint_positions": positions, "height": height}


def _load_processed_smplx(npz_path, target_fps, to_z_up):
    data = np.load(npz_path, allow_pickle=True)
    positions = data["global_joint_positions"]
    T, J, _ = positions.shape
    if to_z_up:
        positions = _normalize_to_z_up(positions)
    positions = _centre_root(positions, z_up=to_z_up)
    quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, J, 1))
    joint_names = np.array(SMPLX_22_JOINT_NAMES[:J])
    parents = SMPLX_22_PARENTS[:J]
    connections = _build_connections(parents)
    return {
        "positions": positions.astype(np.float32),
        "rotations": quats.astype(np.float32),
        "joint_names": joint_names,
        "connections": connections,
        "frame_time": 1.0 / target_fps,
        "frame_count": T,
        "axis_up": "z" if to_z_up else "y",
    }


def _fk_from_amass(npz_path, fmt, target_fps, num_joints):
    if target_fps <= 0:
        target_fps = 30
    data = np.load(npz_path, allow_pickle=True)
    ori_fps = float(data.get("mocap_frame_rate", 120))
    ds = max(1, int(ori_fps / target_fps))

    poses = data["poses"][::ds]
    trans = data["trans"][::ds]

    T = poses.shape[0]
    n_total = poses.shape[1] // 3
    aa = poses.reshape(T, n_total, 3)

    J = min(num_joints, n_total, 22)
    aa = aa[:, :J, :]
    parents = SMPLX_22_PARENTS[:J]
    offsets = SMPL_22_TPOSE_OFFSETS[:J].copy()

    offsets = offsets[:, [2, 1, 0]]
    aa = -aa[:, :, [2, 1, 0]]
    trans = trans[:, [2, 1, 0]]

    positions, quats = fk_smpl(aa, trans, parents, offsets)

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    bad = (norms < 1e-10) | ~np.isfinite(norms) | ~np.isfinite(quats).all(axis=-1, keepdims=True)
    quats = np.where(bad, np.array([0, 0, 0, 1], dtype=quats.dtype), quats)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12

    return positions, quats, 1.0 / target_fps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SMPL → BVH-like NPZ converter")
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_type", default=None, choices=["smpl", "smplh", "smplx"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--use_bodymodel", action="store_true")
    parser.add_argument("--y_up", action="store_true")
    parser.add_argument("--for_bvh_loader", action="store_true")
    parser.add_argument("--scale_for_g1", action="store_true")
    parser.add_argument("--bvh_scale", type=float, default=0.747)

    args = parser.parse_args()
    output = args.output or str(Path(args.input).with_suffix("")) + "_bvhlike.npz"

    result = load_and_convert(
        input_path=args.input, model_path=args.model_path, model_type=args.model_type,
        target_fps=args.fps, num_joints=args.num_joints,
        use_bodymodel=args.use_bodymodel, to_z_up=not args.y_up,
        for_bvh_loader=args.for_bvh_loader,
        scale_for_g1=args.scale_for_g1, bvh_scale=args.bvh_scale,
    )

    np.savez(output, **result)
    print(f"\nSaved: {output}")
    print(f"  positions : {result['positions'].shape}")
    print(f"  rotations : {result['rotations'].shape}")
    print(f"  joints    : {list(result['joint_names'])}")
    print(f"  frames    : {result['frame_count']}")


if __name__ == "__main__":
    main()
