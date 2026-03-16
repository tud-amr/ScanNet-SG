"""
This script is used to add pose and bbox to the frame json file.
The pose and bbox are from the per_frame_points json file.
"""

import argparse
import os
import json
import tqdm

def add_pose_bbox_to_frame_json(file_instance_json, file_ptc_json, output_file, openset_scans=False, skip_existing=False):
    if not os.path.exists(file_instance_json) or not os.path.exists(file_ptc_json):
        # print(f"File {file_instance_json} or {file_ptc_json} does not exist. Skipping...")
        return
    
    if skip_existing and os.path.exists(output_file):
        return

    # Load JSON files
    with open(file_instance_json, "r") as f:
        data_instance_json = json.load(f)

    with open(file_ptc_json, "r") as f:
        data_ptc_json = json.load(f)
    
    if data_instance_json is None or len(data_instance_json) == 0:
        print(f"No data in {file_instance_json}")
        return
    
    if data_ptc_json is None or len(data_ptc_json) == 0:
        print(f"No data in {file_ptc_json}")
        return

    # Build a lookup from file_ptc_json for quick matching
    lookup_ptc_json = {item["instance_id"]: item for item in data_ptc_json}

    merged_data = []
    for obj_instance_json in data_instance_json:
        if openset_scans:
            fid = obj_instance_json.get("frame_instance_id")
        else:
            fid = obj_instance_json.get("instance_id")

        if fid in lookup_ptc_json:
            obj_ptc_json = lookup_ptc_json[fid]
            # Merge dictionaries (obj_instance_json properties overwrite obj_ptc_json if same key)
            merged_obj = {**obj_ptc_json, **obj_instance_json}

            # Rearrange the keys to frame_instance_id, instance_id, object_name, object_description, confidence at the first
            if openset_scans:
                merged_obj = {
                    "frame_instance_id": merged_obj.pop("frame_instance_id"),
                    "instance_id": merged_obj.pop("instance_id"),
                    "object_name": merged_obj.pop("object_name"),
                    "object_description": merged_obj.pop("object_description"),
                    "confidence": merged_obj.pop("confidence"),
                    **merged_obj
                }
            else:
                # In non-openset scans, the instance_id is directly given by instance_mask. From 0 to 255. No need to add frame_instance_id.
                merged_obj = {
                    "instance_id": merged_obj.pop("instance_id"),
                    "object_name": merged_obj.pop("object_name"),
                    "confidence": merged_obj.pop("confidence"),
                    **merged_obj
                }
            merged_data.append(merged_obj)
        else:
            # print(f"No match found for {fid}. Possible reason: object has too few points.")
            continue

    # Save merged result
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    # print(f"Merged data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans_folder", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans")
    parser.add_argument("--openset_scans", action="store_true", help="Use the openset scans")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing files")
    args = parser.parse_args()

    # Iterate over the subfolders in the scans_folder
    for subfolder in tqdm.tqdm(os.listdir(args.scans_folder)):
        if os.path.isdir(os.path.join(args.scans_folder, subfolder)):
            # Check if per_frame_points folder and refined_instance folder exist in the subfolder
            per_frame_points_folder = os.path.join(args.scans_folder, subfolder, "per_frame_points")
            refined_instance_folder = os.path.join(args.scans_folder, subfolder, "refined_instance")
            
            if os.path.exists(per_frame_points_folder) and os.path.exists(refined_instance_folder):
                # Iterate over the files in the per_frame_points folder
                if args.openset_scans:
                    json_files = [f for f in os.listdir(refined_instance_folder) if f.endswith("updated_instance.json")]
                else:
                    json_files = [f for f in os.listdir(refined_instance_folder) if f.endswith(".json") and not f.endswith("instance.json")]

                for json_file in tqdm.tqdm(json_files, desc=f"Processing {subfolder}"):
                    file_instance_json = os.path.join(refined_instance_folder, json_file)
                    if args.openset_scans:
                        frame_id = json_file.split("_")[0]
                    else:
                        frame_id = json_file.split(".")[0]

                    frame_ptc_json = os.path.join(per_frame_points_folder, f"{frame_id}_instances.json")
                    output_file = os.path.join(refined_instance_folder, f"{frame_id}_final_instance.json")
                
                    add_pose_bbox_to_frame_json(file_instance_json, frame_ptc_json, output_file, args.openset_scans, args.skip_existing)
