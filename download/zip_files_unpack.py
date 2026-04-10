#!/usr/bin/env python3
"""
Unpack dataset zip files and restore folder structure.

This script extracts all .zip files from a source directory into one output
directory. The original relative paths stored inside each zip are preserved.

Note:
- topology_map.json is kept as-is (no rename back to topology_map_cleaned.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unzip dataset archives and restore original folder structure."
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        required=True,
        help="Directory containing zip files created by zip_files.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where files are extracted.",
    )
    return parser.parse_args()


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


def extract_zip_files(zip_files: Sequence[Path], output_dir: Path) -> tuple[int, int]:
    """Extract provided zip files into output_dir."""
    total_archives = 0
    total_members = 0
    all_archives = len(zip_files)

    for archive_idx, zip_path in enumerate(zip_files, start=1):
        with ZipFile(zip_path, "r") as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
            member_total = len(members)
            print_progress(f"[UNZIP] {zip_path.name}", 0, member_total)
            for member_idx, member in enumerate(members, start=1):
                zf.extract(member, output_dir)
                print_progress(f"[UNZIP] {zip_path.name}", member_idx, member_total)
            print()
        total_archives += 1
        total_members += member_total
        print(
            f"[OK] Extracted {zip_path.name} ({member_total} files) "
            f"[{archive_idx}/{all_archives}]"
        )

    return total_archives, total_members


def extract_all_zips_in_dir(zip_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Extract all zip files under zip_dir into output_dir."""
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        print(f"[WARN] No zip files found in: {zip_dir}")
        return 0, 0
    return extract_zip_files(zip_files, output_dir)


def main() -> None:
    args = parse_args()
    zip_dir = args.zip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_dir.is_dir():
        raise FileNotFoundError(f"Zip directory does not exist: {zip_dir}")

    total_archives, total_members = extract_all_zips_in_dir(zip_dir, output_dir)
    if total_archives == 0:
        return

    print(
        f"[DONE] Extracted {total_archives} archives, {total_members} files into: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
