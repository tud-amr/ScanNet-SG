import os
import pandas as pd
from collections import Counter
import argparse


def collect_csv_paths(root_dir, target_name="instance_name_map.csv"):
    csv_paths = []

    # root level
    for f in os.listdir(root_dir):
        p = os.path.join(root_dir, f)
        if os.path.isfile(p) and f == target_name:
            csv_paths.append(p)

    # first-layer subfolders
    for f in os.listdir(root_dir):
        sub = os.path.join(root_dir, f)
        if os.path.isdir(sub):
            csv = os.path.join(sub, target_name)
            if os.path.isfile(csv):
                csv_paths.append(csv)

    return csv_paths


def count_instance_names(root_dir, name_col="instance_name", top_n=20, output_csv=None):
    csv_paths = collect_csv_paths(root_dir)
    print(f"Found {len(csv_paths)} CSV files")

    counter = Counter()

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            if name_col not in df.columns:
                raise ValueError(f"Column '{name_col}' not found in {csv_path}")
            counter.update(df[name_col].astype(str))
        except Exception as e:
            print(f"[WARN] Skip {csv_path}: {e}")

    print(f"Total number of unique instance names: {len(counter)}")
    print("\nTop instance names:")
    for name, cnt in counter.most_common(top_n):
        print(f"{name:30s} {cnt}")
    
    # Export to CSV
    if output_csv is None:
        # Default output path if not specified
        output_csv = os.path.join(root_dir, 'unique_instance_names_counts.csv')
    
    # Create DataFrame with all unique names and counts, sorted by count (descending)
    result_df = pd.DataFrame({
        'instance_name': [name for name, _ in counter.most_common()],
        'count': [cnt for _, cnt in counter.most_common()]
    })
    result_df.to_csv(output_csv, index=False)
    print(f"\nExported unique names and counts to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory")
    parser.add_argument("--col", default="name", help="Instance name column")
    parser.add_argument("--top", type=int, default=20, help="Top N names")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path (default: root_dir/unique_instance_names_counts.csv)")

    args = parser.parse_args()
    count_instance_names(args.root_dir, args.col, args.top, args.output)
