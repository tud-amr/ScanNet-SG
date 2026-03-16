import os
import sys
import json
import numpy as np
import tqdm
from fix_name_csv import fix_csv_format
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, ".."))


def process_scene(scene_folder, processed_dataset_dir, raw_images_parent_dir, 
                 exec_path, edge_distance_threshold, skip_existing, max_depth, subsample_factor):
    """Process a single scene folder"""
    processed_scene_folder = os.path.join(processed_dataset_dir, scene_folder)
    
    raw_images_parent_dir_with_quotes = '"' + raw_images_parent_dir + '"'
    processed_dataset_dir_with_quotes = '"' + processed_dataset_dir + '"'

    print(f"Processing scene: {scene_folder}")

    # Check if the processed scene folder exists
    if not os.path.exists(processed_scene_folder):
        print(f"Processed scene folder does not exist for {scene_folder}")
        print(f"Please generate the instance segmentation first. Skipping...")
        return

    # Run the openset_ply_map.cpp to generate the map
    if not os.path.exists(os.path.join(processed_scene_folder, "instance_cloud.ply")) or not skip_existing:
        original_cwd = os.getcwd()
        os.chdir(exec_path)
        print(f"running openset_ply_map for {scene_folder}")
        os.system("./openset_ply_map {} {} {} {} {} {}".format(
            scene_folder, 0, processed_dataset_dir_with_quotes, raw_images_parent_dir_with_quotes, 
            max_depth, subsample_factor))
        os.chdir(original_cwd)
    else:
        print(f"Instance cloud ply file already exists for {scene_folder}")

    ## Generate Topology Map for a scene
    original_cwd = os.getcwd()
    os.chdir(exec_path)
    print(f"running generate_json for {scene_folder}")
    ply_file = os.path.join(processed_scene_folder, "instance_cloud.ply")
    ply_file_with_quotes = '"' + ply_file + '"'
    os.system("./generate_json {} {} {} {}".format(ply_file_with_quotes, 0, 1, edge_distance_threshold))
    os.chdir(original_cwd)

    ## Fix the name of the instances in the instance_name_map.csv
    instance_name_map_path = os.path.join(processed_scene_folder, "instance_name_map.csv")
    if os.path.exists(instance_name_map_path):
        fix_csv_format(instance_name_map_path)
    
    print(f"Completed processing scene: {scene_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw_images_parent_dir', type=str, default='/media/cc/My Passport/dataset/scannet/images/scans')
    parser.add_argument('--processed_dataset_dir', type=str, default='/media/cc/Expansion/scannet/processed/openset_scans/ram/openset_scans')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--processing_threads', type=int, default=6)
    parser.add_argument('--edge_distance_threshold', type=float, default=2.0)
    parser.add_argument('--exec_path', type=str, default='/home/cc/chg_ws/ros_ws/topomap_ws/devel/lib/semantic_topo_map')
    parser.add_argument('--max_depth', type=float, default=0.0, 
                        help='Maximum depth threshold in meters (0 = no filtering, default: 0.0)')
    parser.add_argument('--subsample_factor', type=int, default=1,
                        help='Subsample factor for depth image rows/cols (1 = read all, 2 = read every other row/col, etc., default: 1)')
    args = parser.parse_args()

    raw_images_parent_dir = args.raw_images_parent_dir
    processed_dataset_dir = args.processed_dataset_dir
    skip_existing = args.skip_existing
    exec_path = args.exec_path
    edge_distance_threshold = args.edge_distance_threshold
    max_depth = args.max_depth
    subsample_factor = args.subsample_factor

    # Find the folder names that contain the word "scene" in the scans directory
    scene_folders = [f for f in os.listdir(processed_dataset_dir) if "scene" in f]
    print("Found {} scene folders".format(len(scene_folders)))

    # Process scenes using multiple threads
    with ThreadPoolExecutor(max_workers=args.processing_threads) as executor:
        # Submit all tasks directly
        future_to_scene = {
            executor.submit(
                process_scene, 
                scene_folder, 
                processed_dataset_dir, 
                raw_images_parent_dir, 
                exec_path, 
                edge_distance_threshold, 
                skip_existing,
                max_depth,
                subsample_factor
            ): scene_folder
            for scene_folder in scene_folders
        }

        # Process completed tasks with progress bar
        for future in tqdm.tqdm(as_completed(future_to_scene), total=len(scene_folders), desc="Processing scenes"):
            scene_folder = future_to_scene[future]
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as exc:
                print(f"Scene {scene_folder} generated an exception: {exc}")

    print("All scenes processed!")