#!/usr/bin/env python3
"""
Interactive downloader + unzipper for ScanNet-SG releases.

Usage:
    python download_and_upzip.py /path/to/download_root
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from zip_files_unpack import extract_zip_files


DATASET_URL = "https://data.4tu.nl/datasets/bebe8bd4-cf91-4f86-a28a-87cb870f6cea"
README_URL = "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/c0041825-39bb-4534-9332-f84fc814b621"
TERMS_URL = "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/36b4919e-8906-4435-bb6d-887f88053d39"

KB = 1024
MB = 1024**2
GB = 1024**3


@dataclass(frozen=True)
class FileSpec:
    filename: str
    url: Optional[str]
    compressed_size: int
    uncompressed_size: int
    group: str  # core, pkl, refined, per_frame


SUBSET_FILES: Dict[str, List[FileSpec]] = {
    "ScanNet-SG-509": [
        FileSpec(
            "509_meta_data.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/bf66805b-5ffa-4c39-9b32-aa6beeb74f5f",
            12 * KB,
            12 * KB,
            "core",
        ),
        FileSpec(
            "509_pkl.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/f08ce37f-542a-496e-8b34-5c4f11b6e970",
            int(4.8 * GB),
            int(11.4 * GB),
            "pkl",
        ),
        FileSpec(
            "509_training_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/7f4ec966-9940-4ee8-a109-7370c324ab79",
            int(11.2 * GB),
            int(22.8 * GB),
            "core",
        ),
        FileSpec(
            "509_training_refined_instance.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/fcab5771-67ea-427e-b0fd-5d1acabf23bd",
            int(15.2 * GB),
            int(32.7 * GB),
            "refined",
        ),
        FileSpec(
            "509_test_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/a07dd136-3130-449c-a33a-9e32ba69c25b",
            int(2.3 * GB),
            int(4.6 * GB),
            "core",
        ),
        FileSpec(
            "509_test_refined_instance.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/3e1910a8-5019-46fe-a56e-e1eb7bc5dcc3",
            int(3.3 * GB),
            int(7.0 * GB),
            "refined",
        ),
        FileSpec(
            "509_test_per_frame_points_ply.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/0b8bb7d8-2205-4e94-9c2d-7434221605a8",
            int(175.7 * GB),
            int(592.0 * GB),
            "per_frame",
        ),
    ],
    "ScanNet-SG-GPT": [
        FileSpec(
            "GPT_meta_data.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/5e5c40a0-f9fe-4f4a-afa3-2761a48bdb0f",
            int(19.1 * KB),
            int(19.1 * KB),
            "core",
        ),
        FileSpec(
            "GPT_pkl.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/5f5880fb-c49d-42ee-85cf-b800b915516f",
            int(1.2 * GB),
            int(2.1 * GB),
            "pkl",
        ),
        FileSpec(
            "GPT_training_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/637b4dda-9496-4673-bbf9-3cf1824f45ce",
            int(1.7 * GB),
            int(4.1 * GB),
            "core",
        ),
        FileSpec(
            "GPT_training_refined_instance.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/758d1493-a651-4d2a-81a9-5471affad0cf",
            int(4.2 * GB),
            int(11.2 * GB),
            "refined",
        ),
        FileSpec(
            "GPT_test_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/aa6de2b9-e64e-49c3-9058-6f3a2a1e3f16",
            int(650 * MB),
            int(1.6 * GB),
            "core",
        ),
        FileSpec(
            "GPT_test_refined_instance.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/c88073b4-9113-40d4-bcca-28fe1157b574",
            int(1.5 * GB),
            int(3.9 * GB),
            "refined",
        ),
        FileSpec(
            "GPT_test_per_frame_points_ply.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/29cbf66e-aa69-45f1-a16a-077ac533c112",
            int(36.8 * GB),
            int(117.0 * GB),
            "per_frame",
        ),
    ],
    "ScanNet-SG-Subscan": [
        FileSpec(
            "Subscan_meta_data.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/397779e2-b4d6-46c8-9ee5-4355b0da598b",
            int(125.1 * KB),
            int(125.1 * KB),
            "core",
        ),
        FileSpec(
            "Subscan_pkl.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/97e2ccdf-f234-4fbc-9ffa-9409c40db023",
            int(1.3 * GB),
            int(2.2 * GB),
            "pkl",
        ),
        FileSpec(
            "Subscan_training_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/c87bc59a-871a-4c7e-a32b-c3c7c7c62650",
            int(6.0 * GB),
            int(9.9 * GB),
            "core",
        ),
        FileSpec(
            "Subscan_test_maps.zip",
            "https://data.4tu.nl/file/bebe8bd4-cf91-4f86-a28a-87cb870f6cea/5d712f19-0d75-4d74-8b48-867fb7199a4f",
            int(1.2 * GB),
            int(2.0 * GB),
            "core",
        ),
    ],
}

SUBSET_ORDER = ["ScanNet-SG-509", "ScanNet-SG-GPT", "ScanNet-SG-Subscan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively download and unzip ScanNet-SG datasets."
    )
    parser.add_argument(
        "download_path",
        type=Path,
        help="Target directory where subset folders will be created.",
    )
    return parser.parse_args()


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def ask_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


def prompt_subset_selection() -> List[str]:
    print("Choose subsets to download (default: all subsets).")
    print("Options:")
    for idx, name in enumerate(SUBSET_ORDER, start=1):
        print(f"  {idx}) {name}")
    raw = input(
        "Enter numbers or names separated by comma (or press Enter for all): "
    ).strip()
    if not raw:
        return list(SUBSET_ORDER)

    selected: List[str] = []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    for token in tokens:
        if token.isdigit():
            i = int(token) - 1
            if 0 <= i < len(SUBSET_ORDER):
                selected.append(SUBSET_ORDER[i])
            continue
        for name in SUBSET_ORDER:
            if token.lower() == name.lower():
                selected.append(name)
                break
    selected_unique = [name for name in SUBSET_ORDER if name in set(selected)]
    if not selected_unique:
        print("[WARN] Invalid selection, defaulting to all subsets.")
        return list(SUBSET_ORDER)
    return selected_unique


def files_by_group(subsets: Sequence[str], group: str) -> List[tuple[str, FileSpec]]:
    selected: List[tuple[str, FileSpec]] = []
    for subset in subsets:
        for spec in SUBSET_FILES[subset]:
            if spec.group == group:
                selected.append((subset, spec))
    return selected


def summarize_specs(specs: Sequence[tuple[str, FileSpec]]) -> tuple[int, int]:
    compressed = sum(spec.compressed_size for _, spec in specs if spec.url)
    uncompressed = sum(spec.uncompressed_size for _, spec in specs if spec.url)
    return compressed, uncompressed


def print_option_size_message(
    title: str,
    specs: Sequence[tuple[str, FileSpec]],
    default_yes: bool,
) -> bool:
    available = [item for item in specs if item[1].url]
    unavailable = [item for item in specs if item[1].url is None]
    comp, uncomp = summarize_specs(available)
    print(
        f"{title}: adds {format_bytes(comp)} compressed, "
        f"{format_bytes(uncomp)} decompressed."
    )
    if unavailable:
        names = ", ".join(f"{subset}/{spec.filename}" for subset, spec in unavailable)
        print(f"  [INFO] Not yet downloadable (missing URL): {names}")
    return ask_yes_no("Include this group?", default_yes=default_yes)


def choose_group_with_prompt(
    title: str,
    specs: Sequence[tuple[str, FileSpec]],
    default_yes: bool,
) -> bool:
    """
    Ask include/exclude only if selected subsets contain this data group.
    """
    if not specs:
        print(f"{title}: not available in selected subset(s), skipping.")
        return False
    return print_option_size_message(title, specs, default_yes)


def build_download_plan(
    subsets: Sequence[str],
    include_pkl: bool,
    include_refined: bool,
    include_per_frame: bool,
) -> Dict[str, List[FileSpec]]:
    plan: Dict[str, List[FileSpec]] = {}
    selected_groups = {"core"}
    if include_pkl:
        selected_groups.add("pkl")
    if include_refined:
        selected_groups.add("refined")
    if include_per_frame:
        selected_groups.add("per_frame")

    for subset in subsets:
        subset_specs: List[FileSpec] = []
        for spec in SUBSET_FILES[subset]:
            if spec.group not in selected_groups:
                continue
            if spec.url is None:
                continue
            subset_specs.append(spec)
        plan[subset] = subset_specs
    return plan


def print_download_progress(prefix: str, done: int, total: Optional[int]) -> None:
    bar_width = 30
    if total is None or total <= 0:
        sys.stdout.write(f"\r{prefix} {format_bytes(done)} downloaded")
        sys.stdout.flush()
        return
    ratio = min(done / total, 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(
        f"\r{prefix} |{bar}| {ratio * 100:5.1f}% ({format_bytes(done)}/{format_bytes(total)})"
    )
    sys.stdout.flush()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        header_size = response.headers.get("Content-Length")
        total_size = int(header_size) if header_size and header_size.isdigit() else None

        if out_path.exists() and total_size is not None and out_path.stat().st_size == total_size:
            print(f"[SKIP] {out_path.name} already exists with matching size.")
            return

        print(f"[DOWNLOAD] {out_path.name}")
        with out_path.open("wb") as f:
            downloaded = 0
            chunk_size = 8 * MB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print_download_progress("  progress", downloaded, total_size)
        print()


def main() -> None:
    args = parse_args()
    download_root = args.download_path.expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    print("ScanNet-SG interactive downloader")
    print(f"Dataset page: {DATASET_URL}")
    print(f"Readme: {README_URL}")
    print(f"Terms: {TERMS_URL}")
    print()

    selected_subsets = prompt_subset_selection()
    print(f"Selected subsets: {', '.join(selected_subsets)}")
    print()

    include_pkl = choose_group_with_prompt(
        "Packed pkl files for training and test in OpenSGA",
        files_by_group(selected_subsets, "pkl"),
        default_yes=True,
    )
    include_refined = choose_group_with_prompt(
        "Instance images",
        files_by_group(selected_subsets, "refined"),
        default_yes=True,
    )
    include_per_frame = choose_group_with_prompt(
        "Per-frame point clouds (Huge. Only frames in test set will be downloaded. The per-frame point clouds can be generated with instance images optitionally downloaded in the previous step and depth images in ScanNet dataset.)",
        files_by_group(selected_subsets, "per_frame"),
        default_yes=False,
    )
    print()

    plan = build_download_plan(
        selected_subsets,
        include_pkl=include_pkl,
        include_refined=include_refined,
        include_per_frame=include_per_frame,
    )

    all_specs = [(subset, spec) for subset, specs in plan.items() for spec in specs]
    total_compressed, total_uncompressed = summarize_specs(all_specs)
    total_needed = total_compressed + total_uncompressed

    print("Planned download:")
    for subset in selected_subsets:
        specs = plan.get(subset, [])
        comp = sum(spec.compressed_size for spec in specs)
        uncomp = sum(spec.uncompressed_size for spec in specs)
        print(
            f"  - {subset}: {len(specs)} files, "
            f"{format_bytes(comp)} compressed, {format_bytes(uncomp)} decompressed"
        )
    print()
    print(f"Required zip space: {format_bytes(total_compressed)}")
    print(f"Required extracted space: {format_bytes(total_uncompressed)}")
    print(f"Total required space (zip + extracted): {format_bytes(total_needed)}")
    print()

    if not ask_yes_no("Confirm and start download?", default_yes=True):
        print("[ABORT] Download canceled.")
        return

    print()
    print("[START] Downloading selected files...")

    for subset in selected_subsets:
        subset_root = download_root / subset
        zip_dir = subset_root / "zips"
        extract_dir = subset_root
        zip_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        specs = plan.get(subset, [])
        if not specs:
            print(f"[SKIP] {subset}: no downloadable files selected.")
            continue

        downloaded_paths: List[Path] = []
        for spec in specs:
            assert spec.url is not None
            zip_path = zip_dir / spec.filename
            download_file(spec.url, zip_path)
            downloaded_paths.append(zip_path)

        print(f"[UNZIP] {subset} ({len(downloaded_paths)} archives)")
        archives, members = extract_zip_files(downloaded_paths, extract_dir)
        print(f"[DONE] {subset}: extracted {archives} archives, {members} files")
        print()

    print(f"[ALL DONE] Data is downloaded and extracted under: {download_root}")


if __name__ == "__main__":
    main()
