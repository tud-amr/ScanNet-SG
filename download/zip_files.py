#!/usr/bin/env python3
"""
Package a processed dataset into publishable zip files.

Rules:
1) Zip full folders:
   - meta_data -> meta_data.zip
   - pkl       -> pkl.zip
2) For each split training/ and test/:
   - Create one zip for all core map files from all scenes:
       instance_cloud_background.ply
       instance_cloud_cleaned.ply
       instance_name_map.csv
       topology_map_cleaned.json  -> topology_map.json (inside zip only)
       matched_instance_correspondence_to_00.csv
       transformation.npy
       inv_transformation.txt
   - Create another zip for all files under refined_instance:
       *.png
       <number>_final_instance.json
3) Ignore anything inside any per_frame_points folder.
4) Keep original relative paths from dataset root in zip entries.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile


MAP_FILE_NAMES = {
    "instance_cloud_background.ply",
    "instance_cloud_cleaned.ply",
    "instance_name_map.csv",
    "topology_map_cleaned.json",
    "matched_instance_correspondence_to_00.csv",
    "transformation.npy",
    "inv_transformation.txt",
}

FINAL_INSTANCE_JSON_RE = re.compile(r"^\d+_final_instance\.json$")


def is_ignored_path(path: Path) -> bool:
    """Return True if path is in a per_frame_points subtree."""
    return "per_frame_points" in path.parts


def collect_all_files(root: Path) -> Iterable[Path]:
    """Yield all files recursively under root, skipping ignored subtree."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip descending into ignored subtrees entirely for faster traversal.
        dirnames[:] = [d for d in dirnames if d != "per_frame_points"]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if is_ignored_path(path):
                continue
            yield path


def print_progress(prefix: str, done: int, total: int) -> None:
    """Render a simple terminal progress bar."""
    if total <= 0:
        sys.stdout.write(f"\r{prefix} |{'-' * 30}| 100.0% (0/0)")
        sys.stdout.flush()
        return

    bar_width = 30
    ratio = done / total
    filled = int(ratio * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = ratio * 100.0
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:5.1f}% ({done}/{total})")
    sys.stdout.flush()


def write_zip_records(
    zip_path: Path,
    records: Sequence[Tuple[Path, str]],
    label: str,
) -> int:
    """Write a zip from (src_path, arcname) records with progress display."""
    total = len(records)
    print_progress(label, 0, total)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for idx, (src_path, arcname) in enumerate(records, start=1):
            zf.write(src_path, arcname)
            print_progress(label, idx, total)
    print()
    return total


def zip_full_folder(dataset_root: Path, folder_name: str, save_dir: Path) -> Path:
    """Zip a full first-level folder under dataset_root."""
    source_dir = dataset_root / folder_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Required folder not found: {source_dir}")

    zip_path = save_dir / f"{folder_name}.zip"
    records = [
        (file_path, file_path.relative_to(dataset_root).as_posix())
        for file_path in collect_all_files(source_dir)
    ]
    count = write_zip_records(zip_path, records, f"[ZIP] {folder_name}")
    print(f"[OK] {zip_path} ({count} files)")
    return zip_path


def collect_scene_map_files(scene_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Collect map files for one scene.

    Returns list of (src_path, zip_arcname_path).
    """
    entries: List[Tuple[Path, Path]] = []
    for file_path in collect_all_files(scene_dir):
        name = file_path.name
        if name not in MAP_FILE_NAMES:
            continue

        arcname = file_path
        if name == "topology_map_cleaned.json":
            arcname = file_path.with_name("topology_map.json")
        entries.append((file_path, arcname))
    return entries


def collect_scene_refined_files(scene_dir: Path) -> List[Path]:
    """Collect refined_instance png and <number>_final_instance.json files."""
    matched: List[Path] = []
    for file_path in collect_all_files(scene_dir):
        if "refined_instance" not in file_path.parts:
            continue

        if file_path.suffix.lower() == ".png":
            matched.append(file_path)
            continue

        if FINAL_INSTANCE_JSON_RE.match(file_path.name):
            matched.append(file_path)
    return matched


def collect_scene_files(scene_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """Collect map and refined files in one pass for better speed."""
    map_entries: List[Tuple[Path, Path]] = []
    refined_files: List[Path] = []

    for file_path in collect_all_files(scene_dir):
        name = file_path.name

        if name in MAP_FILE_NAMES:
            arcname = file_path
            if name == "topology_map_cleaned.json":
                arcname = file_path.with_name("topology_map.json")
            map_entries.append((file_path, arcname))

        if "refined_instance" in file_path.parts:
            if file_path.suffix.lower() == ".png" or FINAL_INSTANCE_JSON_RE.match(name):
                refined_files.append(file_path)

    return map_entries, refined_files


def write_zip_with_entries(
    zip_path: Path,
    dataset_root: Path,
    entries: Sequence[Tuple[Path, Path]],
    label: str,
) -> int:
    """Write zip from (src, arcname_abs_like) pairs, arcnames become root-relative."""
    records = [
        (src_path, arcname_path.relative_to(dataset_root).as_posix())
        for src_path, arcname_path in entries
    ]
    return write_zip_records(zip_path, records, label)


def write_zip_with_paths(
    zip_path: Path,
    dataset_root: Path,
    files: Sequence[Path],
    label: str,
) -> int:
    """Write zip from src paths, preserving relative path to dataset_root."""
    records = [
        (src_path, src_path.relative_to(dataset_root).as_posix()) for src_path in files
    ]
    return write_zip_records(zip_path, records, label)


def package_training_or_test(
    dataset_root: Path,
    split: str,
    save_dir: Path,
    *,
    include_maps: bool = True,
    include_refined_instance: bool = True,
) -> None:
    """Package one split into zips: maps and/or refined_instance."""
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Required folder not found: {split_dir}")

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not scene_dirs:
        print(f"[WARN] No scenes found under: {split_dir}")
        return

    all_map_entries: List[Tuple[Path, Path]] = []
    all_refined_files: List[Path] = []

    for scene_dir in scene_dirs:
        scene_map_entries, scene_refined_files = collect_scene_files(scene_dir)
        all_map_entries.extend(scene_map_entries)
        all_refined_files.extend(scene_refined_files)

    if include_maps:
        if all_map_entries:
            map_zip = save_dir / f"{split}_maps.zip"
            map_count = write_zip_with_entries(
                map_zip,
                dataset_root,
                all_map_entries,
                f"[ZIP] {split}_maps",
            )
            print(f"[OK] {map_zip} ({map_count} files)")
        else:
            print(f"[SKIP] {split}: no map files matched")

    if include_refined_instance:
        if all_refined_files:
            refined_zip = save_dir / f"{split}_refined_instance.zip"
            refined_count = write_zip_with_paths(
                refined_zip,
                dataset_root,
                all_refined_files,
                f"[ZIP] {split}_refined_instance",
            )
            print(f"[OK] {refined_zip} ({refined_count} files)")
        else:
            print(f"[SKIP] {split}: no refined_instance files matched")


def collect_per_frame_points_ply(split_dir: Path) -> List[Path]:
    """Collect all .ply files under any per_frame_points folder for a split."""
    matched: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(split_dir):
        base = Path(dirpath)
        in_per_frame_points = "per_frame_points" in base.parts
        if not in_per_frame_points:
            continue
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() == ".ply":
                matched.append(path)
    return matched


def package_per_frame_points_ply(dataset_root: Path, split: str, save_dir: Path) -> None:
    """Package all per_frame_points ply files in one split-level zip."""
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Required folder not found: {split_dir}")

    ply_files = collect_per_frame_points_ply(split_dir)
    if not ply_files:
        print(f"[SKIP] {split}: no per_frame_points ply files matched")
        return

    zip_path = save_dir / f"{split}_per_frame_points_ply.zip"
    count = write_zip_with_paths(
        zip_path,
        dataset_root,
        ply_files,
        f"[ZIP] {split}_per_frame_points_ply",
    )
    print(f"[OK] {zip_path} ({count} files)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publishable zip files from ScanNet-like processed dataset folder."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Dataset root path, e.g. "
            "/media/cc/Expansion/scannet/processed/ScanNet-SG-509"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to write zip files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "training", "test"],
        default="all",
        help="Choose which split to package. Default: all.",
    )
    parser.add_argument(
        "--include-per-frame-points-ply",
        action="store_true",
        help=(
            "Also package all .ply files under per_frame_points into "
            "<split>_per_frame_points_ply.zip."
        ),
    )
    parser.add_argument(
        "--only-maps",
        action="store_true",
        help="Package only <split>_maps.zip and skip all other archives.",
    )
    parser.add_argument(
        "--only-per-frame-points-ply",
        action="store_true",
        help=(
            "Package only per_frame_points .ply files and skip all other "
            "archives (meta_data/pkl/maps/refined_instance)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.source.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {dataset_root}")

    splits = ["training", "test"] if args.split == "all" else [args.split]

    if args.only_per_frame_points_ply:
        for split in splits:
            package_per_frame_points_ply(dataset_root, split, save_dir)
        print(f"[DONE] Per-frame-points ply zip files are saved in: {save_dir}")
        return

    if not args.only_maps:
        zip_full_folder(dataset_root, "meta_data", save_dir)
        zip_full_folder(dataset_root, "pkl", save_dir)

    for split in splits:
        package_training_or_test(
            dataset_root,
            split,
            save_dir,
            include_maps=True,
            include_refined_instance=not args.only_maps,
        )
        if (not args.only_maps) and args.include_per_frame_points_ply:
            package_per_frame_points_ply(dataset_root, split, save_dir)

    print(f"[DONE] All zip files are saved in: {save_dir}")


if __name__ == "__main__":
    main()
