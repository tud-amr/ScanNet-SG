import os
import argparse
import csv
import shutil
import open3d as o3d

PLY_FILENAME = "aligned_cloud_with_scan_00.ply"
CSV_NAME = "to_examine.csv"
CSV_LAST_NAME = "to_examine_last.csv"

def find_ply_scenes(root_dir):
    scenes = []
    for scene_name in sorted(os.listdir(root_dir)):
        if not scene_name.endswith("_00"):
            scene_path = os.path.join(root_dir, scene_name)
            ply_path = os.path.join(scene_path, PLY_FILENAME)
            if os.path.isfile(ply_path):
                scenes.append(scene_name)
    return scenes

def save_csv(scenes, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for scene in scenes:
            writer.writerow([scene])

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        reader = csv.reader(f)
        return [row[0] for row in reader if row]

def backup_existing_csv(csv_path, backup_path):
    if os.path.exists(csv_path):
        shutil.copy(csv_path, backup_path)
        print(f"Backed up existing CSV to: {backup_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize .ply files from dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to processed/scans")
    parser.add_argument("--new", action="store_true", help="Generate new CSV and examine all valid scenes")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    # Check if the dataset_dir is a valid directory
    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} is not a valid directory")
        return

    csv_path = os.path.join(dataset_dir, CSV_NAME)
    csv_backup_path = os.path.join(dataset_dir, CSV_LAST_NAME)

    if args.new:
        scenes = find_ply_scenes(dataset_dir)
        print(f"Found {len(scenes)} scenes with PLY files.")
        backup_existing_csv(csv_path, csv_backup_path)
        save_csv(scenes, csv_path)
    else:
        scenes = load_csv(csv_path)
        print(f"Loaded {len(scenes)} scenes from {csv_path}")

    updated_scenes = scenes.copy()
    for scene in scenes:
        ply_path = os.path.join(dataset_dir, scene, PLY_FILENAME)
        if not os.path.exists(ply_path):
            print(f"PLY file not found for scene {scene}, skipping.")
            continue

        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Viewing: {scene}. Press 'p' (positive), 'n' (negative), or 'q'/Esc (quit).")

        key_response = {}

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=scene)
        vis.add_geometry(pcd)

        vis.register_key_callback(ord("P"), lambda vis: (key_response.update({"key": "p"}), vis.close())[1])
        vis.register_key_callback(ord("N"), lambda vis: (key_response.update({"key": "n"}), vis.close())[1])
        vis.register_key_callback(ord("Q"), lambda vis: (key_response.update({"key": "q"}), vis.close())[1])
        vis.register_key_callback(256, lambda vis: (key_response.update({"key": "q"}), vis.close())[1])  # ESC

        vis.run()
        vis.destroy_window()

        key = key_response.get("key", "")
        if key == "p":
            print(f"[✓] Accepted: {scene}")
            updated_scenes.remove(scene)
            save_csv(updated_scenes, csv_path)
        elif key == "n":
            print(f"[✗] Rejected: {scene}")
        elif key == "q":
            print(f"[→] Quit early at scene: {scene}")
            break

if __name__ == "__main__":
    main()
