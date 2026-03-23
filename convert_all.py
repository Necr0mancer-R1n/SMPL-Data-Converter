#!/usr/bin/env python3
"""批量转换 AMASS 数据 → Drake / MuJoCo 格式."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "input"
OUTPUT_ROOT = SCRIPT_DIR / "output"


def find_amass_files(directory: Path) -> list[Path]:
    from smpl_to_npz import detect_format
    results = []
    for f in sorted(directory.rglob("*.npz")):
        try:
            fmt = detect_format(str(f))
            if fmt.startswith("amass_") or fmt == "processed_smplx":
                results.append(f)
        except Exception:
            continue
    return results


def list_input_folders() -> list[Path]:
    if not INPUT_ROOT.exists():
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return []
    folders = []
    for item in sorted(INPUT_ROOT.iterdir()):
        if item.is_dir() and len(list(item.rglob("*.npz"))) > 0:
            folders.append(item)
    return folders


def prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = " ←" if i == default else ""
        print(f"  [{i}] {opt}{marker}")
    while True:
        raw = input(f"\n请输入编号 (默认 {default}): ").strip()
        if raw == "":
            return default
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  无效输入，请输入 0~{len(options)-1}")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(prompt + suffix).strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "是")


def run_conversion(
    input_dir: Path,
    output_dir: Path,
    use_bodymodel: bool,
    model_path: str | None,
    model_type: str | None,
    fps: int,
    also_bvh: bool,
):
    from smpl_to_npz import load_and_convert, load_and_convert_for_opt

    drake_dir = output_dir / "Drake"
    mujoco_dir = output_dir / "MuJoCo"
    drake_dir.mkdir(parents=True, exist_ok=True)
    mujoco_dir.mkdir(parents=True, exist_ok=True)

    files = find_amass_files(input_dir)
    if not files:
        print(f"\n在 {input_dir} 下未找到 AMASS .npz 文件。")
        return

    print(f"\n{'=' * 60}")
    print(f"输入: {input_dir}")
    print(f"输出: Drake → {drake_dir}")
    print(f"      MuJoCo → {mujoco_dir}")
    print(f"帧率: {fps} fps  |  文件数: {len(files)}")
    print(f"{'=' * 60}\n")

    if not prompt_yes_no("确认开始转换？"):
        print("已取消。")
        return

    print()
    success, failed = 0, 0

    for i, npz_file in enumerate(files, 1):
        rel = npz_file.relative_to(input_dir)
        out_stem = str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")

        print(f"[{i}/{len(files)}] {rel}")
        try:
            common_args = dict(
                input_path=str(npz_file),
                model_path=model_path,
                model_type=model_type,
                target_fps=fps,
                num_joints=22,
                use_bodymodel=use_bodymodel,
            )

            drake_result = load_and_convert(**common_args,
                                            for_bvh_loader=True,
                                            scale_for_g1=True)
            drake_out = drake_dir / f"{out_stem}.npz"
            np.savez(str(drake_out), **drake_result)
            T = drake_result["frame_count"]
            print(f"  -> Drake/{drake_out.name}  ({T} frames)")

            mujoco_result = load_and_convert_for_opt(**common_args)
            mujoco_out = mujoco_dir / f"{out_stem}.npz"
            np.savez(str(mujoco_out), **mujoco_result)
            print(f"  -> MuJoCo/{mujoco_out.name}  ({T} frames)")

            if also_bvh:
                from smpl_to_bvh import convert_to_bvh
                bvh_out = drake_dir / f"{out_stem}.bvh"
                convert_to_bvh(
                    input_path=str(npz_file),
                    output_path=str(bvh_out),
                    model_path=model_path,
                    model_type=model_type,
                    target_fps=fps,
                    use_bodymodel=use_bodymodel,
                )
                print(f"  -> Drake/{bvh_out.name}")

            success += 1
        except Exception as e:
            print(f"  x 失败: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"完成  成功: {success}, 失败: {failed}")
    print(f"输出: {output_dir}")


def main():
    print("=" * 60)
    print("  SMPL Data Converter")
    print("=" * 60)

    folders = list_input_folders()
    if not folders:
        print(f"\n在 {INPUT_ROOT}/ 下未找到数据。")
        print(f"请将 AMASS 数据放入 input/ 目录。")
        sys.exit(1)

    folder_labels = [f"{f.name}/  ({len(list(f.rglob('*.npz')))} 个文件)"
                     for f in folders]
    folder_labels.append("全部转换")

    choice = prompt_choice("选择数据集:", folder_labels)
    selected_folders = folders if choice == len(folders) else [folders[choice]]

    models_dir = SCRIPT_DIR / "models"
    available = [n for n in ["smplh", "smplx"] if (models_dir / n).exists()]

    if not available:
        print(f"\n未找到 body model ({models_dir}/)。请放入 smplh/ 或 smplx/ 后重试。")
        sys.exit(1)

    if len(available) == 1:
        chosen_model = available[0]
        print(f"\n使用 body model: {chosen_model}")
    else:
        idx = prompt_choice("选择 body model:", available, default=0)
        chosen_model = available[idx]

    also_bvh = prompt_yes_no("同时导出 BVH？", default=False)

    for folder in selected_folders:
        output_dir = OUTPUT_ROOT / folder.name
        print(f"\n>>> {folder.name}/")
        run_conversion(
            input_dir=folder,
            output_dir=output_dir,
            use_bodymodel=True,
            model_path=str(models_dir),
            model_type=chosen_model,
            fps=30,
            also_bvh=also_bvh,
        )

    print("\n全部完成！")


if __name__ == "__main__":
    main()
