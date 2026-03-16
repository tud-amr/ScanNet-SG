"""
Align the instances in the Scene scans to the first scan.
E.g. scene0000_01 is aligned to scene0000_00.

The output is a csv file that contains the instance id in the first scan and the instance id in the target scan.
The csv file is saved in the target scan folder with the name "matched_instance_correspondence_to_00.csv".
"""

import os
import argparse
import alignment_examine
file_path = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/cc/Expansion/scannet/processed/scans") # "/media/cc/My Passport/dataset/scannet/processed/scans"
    # parser.add_argument("--data_dir", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/openset_scans") # "/media/cc/My Passport/dataset/scannet/processed/scans"
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--use_scene_csv", action="store_true", default=False)
    parser.add_argument("--openset_scans", action="store_true", default=False)
    args = parser.parse_args()

    if args.openset_scans:
        # Check if the data_dir ends with "openset_scans"
        if not args.data_dir.endswith("openset_scans"):
            print("Error: The data_dir must end with 'openset_scans' when using openset_scans")
            exit()
    else:
        # Check if the data_dir ends with "scans"
        if not args.data_dir.endswith("scans"):
            print("Error: The data_dir must end with 'scans' when not using openset_scans")
            exit()

    if args.use_scene_csv:
        scene_csv_path = os.path.join(args.data_dir, alignment_examine.CSV_NAME)
        scene_folders = alignment_examine.load_csv(scene_csv_path)
    else:
        scene_folders = [f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))]

    print(f"Found {len(scene_folders)} scene folders in the source directory {args.data_dir}")
    if len(scene_folders) == 0:
        print(f"No scene folders to process found in the source directory")
        exit()
    
    # For each scene folder, align the instances
    for scene_folder in scene_folders:
        scan_id = scene_folder.split("_")[-1]
        scene_id = scene_folder.split("_")[0]
        if scan_id != "00":
            source_dir = os.path.join(args.data_dir, scene_id + "_00")
            target_dir = os.path.join(args.data_dir, scene_id + "_" + scan_id)
            print(f"Aligning instances from {target_dir} to {source_dir}")

            # Check if the source directory exists
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} does not exist")
                continue

            # Check if the result csv file exists
            result_csv_path = os.path.join(target_dir, "matched_instance_correspondence_to_00.csv")
            if os.path.exists(result_csv_path) and args.skip_existing:
                print(f"Result csv file {result_csv_path} already exists")
                continue
        
            os.chdir(file_path)
            if args.openset_scans:
                os.system(f"python align_instances.py --source_dir '{source_dir}' --target_dir '{target_dir}' --ori_pt_transform --use_bert_embeddings --three_channel_id")
            else:
                os.system(f"python align_instances.py --source_dir '{source_dir}' --target_dir '{target_dir}' --ori_pt_transform --use_bert_embeddings  --recalculate_bert_embeddings")
            
