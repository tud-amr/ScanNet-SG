import os
import sys
import json
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, ".."))

# Path to the processed dataset with instance segmentation and json files
processed_dataset_dir = "/media/cc/Expansion/scannet/processed"
exec_path = "/home/cc/chg_ws/ros_ws/topomap_ws/devel/lib/semantic_topo_map"
step = 3 # Processed every 3 frames in the processed dataset
edge_distance_threshold = 2.0

# Path to the processed scans directory
processed_scans_dir = os.path.join(processed_dataset_dir, "scans")

# Find the folder names that contain the word "scene" in the scans directory
scene_folders = [f for f in os.listdir(processed_scans_dir) if "scene" in f]
print("Found {} scene folders".format(len(scene_folders)))

# Generate map for all scenes
count = 0
for scene_folder in scene_folders:
    if "_00" not in scene_folder: #skip more scans for a scene
        continue
    processed_scene_folder = os.path.join(processed_scans_dir, scene_folder)
    processed_scene_folder_with_quotes = '"' + processed_scene_folder + '"'

    refined_instance_folder = os.path.join(processed_scans_dir, scene_folder, "refined_instance")
    refined_instance_folder_with_quotes = '"' + refined_instance_folder + '"'

    print("Processing scene: {}".format(scene_folder))

    # Check if the processed scene folder exists
    if not os.path.exists(processed_scene_folder):
        print("Processed scene folder does not exist for {}".format(scene_folder))
        print("Please generate the instance segmentation first. Skipping...")
        continue

    # Check if "averaged_instance_features.json" and "instance_bert_embeddings.json" and "topology_map.json" already exist
    averaged_instance_features_path = os.path.join(processed_scene_folder, "averaged_instance_features.json")
    instance_bert_embeddings_path = os.path.join(processed_scene_folder, "instance_bert_embeddings.json")
    topology_map_path = os.path.join(processed_scene_folder, "topology_map.json")

    instance_ply_background_path = os.path.join(processed_scene_folder, "instance_cloud_background.ply")
    if os.path.exists(instance_ply_background_path):
        print("Instance cloud background ply file already exists for {}. Skipping...".format(scene_folder))
        continue

    # if os.path.exists(averaged_instance_features_path) and os.path.exists(instance_bert_embeddings_path) and os.path.exists(topology_map_path):
    #     print("All files already exist for {}. Skipping...".format(scene_folder))
    #     continue

    # Find the end frame. Assume the frame number is 0.png, 1.png, 2.png, ..., 5001.png,...
    frame_names = os.listdir(refined_instance_folder)
    max_frame = 0
    for frame_name in frame_names:
        if "final" in frame_name:
            continue
        frame_idx = int(frame_name.split(".")[0])
        if frame_idx > max_frame:
            max_frame = frame_idx
    end_frame = max_frame
    print("End frame: {}".format(end_frame))

    ## Generate PLY map ./scannet_ply_map <scene_name> <start_frame> <end_frame> <step>
    os.chdir(exec_path)
    os.system("./scannet_ply_map {} {} {} {}".format(scene_folder, 0, end_frame, step))
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    ## Generate Instance to Object Name Map by running python get_instance_names.py --input_folder "/media/clarence/My Passport/dataset/scannet/processed/scans/scene0000_00" --scene_id "scene0000_00"
    # os.chdir(file_path)
    # os.system("python get_instance_names.py --input_folder {} --scene_id {}".format(processed_scene_folder_with_quotes, scene_folder))
    
    # ## Get Fused Visual-Language Features for Instances in a Scene
    # os.chdir(exec_path)
    # os.system("./get_fused_object_features {} {}".format(refined_instance_folder_with_quotes, processed_scene_folder_with_quotes))

    # ## Generate Topology Map for a scene
    # os.chdir(exec_path)
    # ply_file = os.path.join(processed_scene_folder, "instance_cloud.ply")
    # ply_file_with_quotes = '"' + ply_file + '"'
    # os.system("./generate_json {} {} {} {}".format(ply_file_with_quotes, 0, 0, edge_distance_threshold))

    # Generate 5 scenes and exit for testing
    # count += 1
    # if count > 5:
    #     exit()
