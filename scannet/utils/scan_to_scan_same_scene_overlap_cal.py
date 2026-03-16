import re
from pathlib import Path
import pandas as pd
import tqdm


def read_instance_ids(instance_name_map_csv: Path) -> set[int]:
    """
    Read instance_name_map.csv and return the set of instance IDs from the FIRST column.
    Works with/without a header; ignores non-integer entries.
    """
    if not instance_name_map_csv.is_file():
        return set()

    # Try normal read (header inferred)
    try:
        df = pd.read_csv(instance_name_map_csv)
        if df.shape[1] >= 1:
            col = df.columns[0]
            vals = df[col].tolist()
            out = set()
            for v in vals:
                try:
                    out.add(int(v))
                except Exception:
                    pass
            if out:
                return out
    except Exception:
        pass

    # Fallback: no-header, take first column
    try:
        df = pd.read_csv(instance_name_map_csv, header=None)
        out = set()
        for v in df.iloc[:, 0].tolist():
            try:
                out.add(int(v))
            except Exception:
                pass
        return out
    except Exception:
        return set()


def read_correspondences(matched_csv: Path) -> dict[int, int]:
    """
    Read matched_instance_correspondence_to_00.csv and return mapping:
      instance_id_in_yy -> instance_id_in_00
    Works with/without a header; ignores non-integer rows.
    """
    if not matched_csv.is_file():
        return {}

    # Try normal read (header inferred)
    try:
        df = pd.read_csv(matched_csv)
        if df.shape[1] >= 2:
            c0, c1 = df.columns[0], df.columns[1]
            mapping = {}
            for a, b in zip(df[c0].tolist(), df[c1].tolist()):
                try:
                    mapping[int(a)] = int(b)
                except Exception:
                    pass
            if mapping:
                return mapping
    except Exception:
        pass

    # Fallback: no-header
    try:
        df = pd.read_csv(matched_csv, header=None)
        mapping = {}
        for a, b in zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()):
            try:
                mapping[int(a)] = int(b)
            except Exception:
                pass
        return mapping
    except Exception:
        return {}


def compute_overlap_metrics(
    inst_ids_00: set[int],
    inst_ids_yy: set[int],
    corr_yy_to_00: dict[int, int],
    scan_tag: str,
    *,
    treat_invalid_mapped_as_unmatched: bool = True
) -> dict:
    """
    Robust "do they share the same objects?" metric.

    We form a canonical set for yy:
      - matched yy instances -> mapped 00 instance id (int)
      - unmatched yy instances -> unique token ("unmatched", scan_tag, inst_yy)
      - (optional) if mapped id not present in inst_ids_00, treat as unmatched

    Then:
      inter = |I00 ∩ canon(yy)|
      union = |I00 ∪ canon(yy)|
      iou = inter/union

    Also outputs recall00 and precisionyy for easier interpretation.
    """
    yy_canon = set()
    unmatched_yy = 0
    matched_pairs = 0
    mapped_but_invalid = 0  # mapped to an id not present in I00

    for inst_yy in inst_ids_yy:
        if inst_yy in corr_yy_to_00:
            mapped_00 = corr_yy_to_00[inst_yy]
            matched_pairs += 1

            if treat_invalid_mapped_as_unmatched and (mapped_00 not in inst_ids_00):
                yy_canon.add(("unmatched", scan_tag, inst_yy))
                unmatched_yy += 1
                mapped_but_invalid += 1
            else:
                yy_canon.add(mapped_00)  # int in 00 ID space
        else:
            yy_canon.add(("unmatched", scan_tag, inst_yy))
            unmatched_yy += 1

    inter = len(inst_ids_00.intersection(yy_canon))
    union = len(inst_ids_00.union(yy_canon))
    iou = (inter / union) if union > 0 else 0.0

    recall00 = (inter / len(inst_ids_00)) if inst_ids_00 else 0.0
    precisionyy = (inter / len(yy_canon)) if yy_canon else 0.0

    return {
        "num_instances_00": len(inst_ids_00),
        "num_instances_yy": len(inst_ids_yy),
        "yy_canon_size": len(yy_canon),
        "matched_pairs_in_file": len(corr_yy_to_00),
        "matched_pairs_used": matched_pairs,
        "unmatched_yy": unmatched_yy,
        "mapped_but_invalid": mapped_but_invalid,
        "intersection": inter,
        "union": union,
        "iou": iou,
        "recall00": recall00,
        "precisionyy": precisionyy,
    }


def main(
    root_dir: str,
    out_csv: str = "scene_scan_object_overlap_to_00.csv",
    *,
    treat_invalid_mapped_as_unmatched: bool = True
) -> None:
    """
    Walk root_dir recursively for folders named sceneXXXX_YY.
    For each sceneXXXX_YY where YY != 00:
      - load instance_name_map.csv of both YY and 00
      - load matched_instance_correspondence_to_00.csv in YY
      - compute robust overlap metrics
    Save results to a CSV under root_dir/out_csv.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    # Match folder names like scene0101_00
    pat = re.compile(r"^(scene\d{4})_(\d{2})$")

    # Collect all scene->scan->path
    # scene_to_scans: dict[str, dict[str, Path]] = {}
    # for p in tqdm.tqdm(root.rglob("*"), desc="Scanning scenes"):
    #     if not p.is_dir():
    #         continue
    #     m = pat.match(p.name)
    #     if not m:
    #         continue
    #     scene, scan = m.group(1), m.group(2)
    #     scene_to_scans.setdefault(scene, {})[scan] = p

    scene_to_scans: dict[str, dict[str, Path]] = {}
    subfolders = [p for p in root.iterdir() if p.is_dir()]
    for subfolder in tqdm.tqdm(subfolders, desc="Processing scenes"):
        if not subfolder.is_dir():
            continue
        m = pat.match(subfolder.name)
        if not m:
            continue
        scene, scan = m.group(1), m.group(2)
        scene_to_scans.setdefault(scene, {})[scan] = subfolder

    rows = []
    for scene, scans in tqdm.tqdm(sorted(scene_to_scans.items()), desc="Processing scenes"):
        if "00" not in scans:
            continue

        dir_00 = scans["00"]
        inst_00_csv = dir_00 / "instance_name_map.csv"
        inst_ids_00 = read_instance_ids(inst_00_csv)

        for scan, dir_yy in sorted(scans.items()):
            if scan == "00":
                continue

            inst_yy_csv = dir_yy / "instance_name_map.csv"
            corr_csv = dir_yy / "matched_instance_correspondence_to_00.csv"

            inst_ids_yy = read_instance_ids(inst_yy_csv)
            corr = read_correspondences(corr_csv)

            metrics = compute_overlap_metrics(
                inst_ids_00,
                inst_ids_yy,
                corr,
                scan_tag=scan,
                treat_invalid_mapped_as_unmatched=treat_invalid_mapped_as_unmatched,
            )

            rows.append({
                "scan_folder_yy": dir_yy.name,
                "scan_folder_00": dir_00.name,
                "intersection": metrics["intersection"],
                "union": metrics["union"],
                "iou": metrics["iou"],
            })

    out_path = root / out_csv
    df_out = pd.DataFrame(rows)

    if not df_out.empty:
        # Sort by scan folder names (scenexxxx_yy)
        df_out = df_out.sort_values("scan_folder_yy", ignore_index=True)

    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(df_out)})")


if __name__ == "__main__":
    # CHANGE THIS
    ROOT_DIR = r"/media/cc/Expansion/scannet/processed/scans"

    # If True: if a correspondence maps to an instance_id_in_00 that is NOT in sceneXXXX_00/instance_name_map.csv,
    #          we count it as unmatched (safer / more conservative).
    # If False: we still count it as matched-to-00-ID-space even if 00's instance_name_map.csv doesn't list it.
    TREAT_INVALID_MAPPED_AS_UNMATCHED = True

    main(
        ROOT_DIR,
        out_csv="scene_scan_object_overlap_to_00.csv",
        treat_invalid_mapped_as_unmatched=TREAT_INVALID_MAPPED_AS_UNMATCHED,
    )
