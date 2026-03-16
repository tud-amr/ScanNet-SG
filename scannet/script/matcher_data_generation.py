import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "..", "..", "script"))

from include.topology_map import *
import alignment_examine

def correctPositionRange(pos_x_range, pos_y_range, pos_z_range, pos_x_range_to_correct, pos_y_range_to_correct, pos_z_range_to_correct):
    """
    Correct the position range to the global range.
    """
    for global_range, range_to_correct in [(pos_x_range, pos_x_range_to_correct), 
                                          (pos_y_range, pos_y_range_to_correct), 
                                          (pos_z_range, pos_z_range_to_correct)]:
        range_to_correct[0] = min(global_range[0], range_to_correct[0])
        range_to_correct[1] = max(global_range[1], range_to_correct[1])
    
    return pos_x_range_to_correct, pos_y_range_to_correct, pos_z_range_to_correct


def loadIdCorrectionCsv(csv_path: str):
    """
    Load the id correction csv file. This is because scene0000_01 is initially not aligned with scene0000_00, so the instance id in scene0000_01 is not the same as the instance id in scene0000_00.
    """
    df = pd.read_csv(csv_path)
    return {row['instance_id']: row['instance_id_in_00'] for _, row in df.iterrows()}


def parseInstanceJson(json_file_path: str):
    """
    Parse the instance json file.
    """
    with open(json_file_path, 'r') as f:
        # If the content is "null", return an empty dictionary
        content = f.read().strip()        
        if not content or content == "null":
            return {}
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {json_file_path}: {e}")
            return {}

        # Create a dictionary mapping instance_id to a dictionary with feature, name, and confidence
        frame_feature_dict = {
            entry['instance_id']: {
                'feature': np.array(entry['feature'], dtype=np.float32),
                'object_name': entry['object_name'],
                'confidence': entry['confidence'],
                'bbox_size': entry['bbox_size'],
                'center': entry['center']
            }
            for entry in data
        }
        # print(f"frame_feature_dict: {frame_feature_dict}")
    return frame_feature_dict


def addMarginToRange(range, ratio=0.1):
    """
    Add a margin to the range.
    """
    return [range[0] - ratio * (range[1] - range[0]), range[1] + ratio * (range[1] - range[0])]


def gentlyScaleTo01(array):
    """
    Slightly bias and scale array so all values are in [0, 1],
    changing as little as possible.
    """
    array = np.asarray(array, dtype=float)
    min_val = array.min()
    max_val = array.max()
    
    # No change if already within range
    if min_val >= 0 and max_val <= 1:
        return array.copy()
    
    # Smallest adjustment to fit into [0,1]
    shift = 0
    scale = 1
    
    if min_val < 0:
        shift = -min_val
    if max_val + shift > 1:
        scale = 1 / (max_val + shift)
    
    return (array + shift) * scale

def normalizePositionArray(position_array, range):
    """
    Normalize the position array to the range of 0 to 1.
    """
    return gentlyScaleTo01((position_array - range[0]) / (range[1] - range[0]))


def normalizeBboxArray(bbox_array, range):
    """
    Normalize the bbox array to the range of 0 to 1. Clip to [0, 1] if exceeds.
    """
    return np.clip(bbox_array / (range[1] - range[0]), 0, 1)


def generate_self_matching_data(scene_folder, map_folder, frame_pose_dir, args, 
                               keypoints1_normalized, descriptors1, bbox1_normalized,
                               ids1, scene_range_all_with_margin, text_embedding1=None, text_model=None):
    """
    Generate self-matching data for a scene.
    
    Args:
        scene_folder: The scene folder name
        map_folder: The folder containing topology maps
        frame_pose_dir: Directory containing frame poses
        args: Command line arguments
        keypoints1_normalized: Normalized keypoints from topology map
        descriptors1: Descriptors from topology map
        bbox1_normalized: Normalized bbox data from topology map
        ids1: Instance IDs from topology map
        scene_range_all_with_margin: Scene range with margin for normalization
    
    Returns:
        data_list: List of self-matching data dictionaries
        data_info_list: List of self-matching data info strings
    """
    data_list = []
    data_info_list = []
    
    # Get all the json files in the scene folder
    refined_instance_folder = os.path.join(map_folder, scene_folder, "refined_instance")
    json_files = [f for f in os.listdir(refined_instance_folder) if f.endswith("final_instance.json")]

    for json_file in json_files:
        # print(f"Processing json file: {json_file}")
        frame_feature_dict = parseInstanceJson(os.path.join(refined_instance_folder, json_file))

        frame_id = json_file.split("_")[0]

        # Only consider the frame with more than 2 instances
        frame_instance_num = len(frame_feature_dict)
        if frame_instance_num < args.min_instance_num_frame:
            # print(f"Less than 2 instances found for scene {scene_folder} in frame {json_file}")
            continue
            
        # Find the matching nodes in the topology map by id
        descriptors0 = np.zeros((frame_instance_num, 256)) # 256 is the dimension of the visual embedding
        matches0 = (np.ones(frame_instance_num) * -1).astype(int) # -1 means no match. Initialize with -1
        names0 = []

        if not args.graph_generated_position:
            keypoints0 = np.zeros((frame_instance_num, 3))

        if args.add_bbox_data:
            bbox0_height = np.zeros((frame_instance_num, 1))
            bbox0_width = np.zeros((frame_instance_num, 1))
            bbox0_length = np.zeros((frame_instance_num, 1))
            
        for i in range(frame_instance_num):
            instance_id = list(frame_feature_dict.keys())[i]
            descriptors0[i] = frame_feature_dict[instance_id]['feature']

            if not args.graph_generated_position:
                keypoints0[i] = [frame_feature_dict[instance_id]['center']['x'], frame_feature_dict[instance_id]['center']['y'], frame_feature_dict[instance_id]['center']['z']]

            if args.add_bbox_data:
                bbox0_height[i] = frame_feature_dict[instance_id]['bbox_size']['y']
                bbox0_width[i] = frame_feature_dict[instance_id]['bbox_size']['x']
                bbox0_length[i] = frame_feature_dict[instance_id]['bbox_size']['z']
                bbox0 = np.stack([bbox0_height, bbox0_width, bbox0_length], axis=1)
                # REmove the last dimension, which is always 1
                bbox0 = np.squeeze(bbox0, axis=-1)
                bbox0_normalized = normalizeBboxArray(bbox0, scene_range_all_with_margin)

            if args.add_text_feature_data:
                names0.append(frame_feature_dict[instance_id]['object_name'])

            if instance_id in ids1:
                matches0[i] = np.where(ids1 == instance_id)[0][0]

        if args.graph_generated_position:
            #### Now set keypoints0 using the matches0 and add some normalization and random noise
            keypoints0 = np.array([keypoints1_normalized[i] for i in matches0])
            min_keypoints0 = np.min(keypoints0, axis=0)
            keypoints0 = keypoints0 - min_keypoints0
            keypoints0 = keypoints0 + abs(np.random.normal(0, args.position_noise_std, keypoints0.shape))
        
        if args.add_text_feature_data:
            text_embedding0 = text_model.encode(names0)
            # print(f"text_embedding0: {text_embedding0.shape}")
            # print(f"text_embedding1: {text_embedding1.shape}")
            # exit()

        # Normalize the keypoints0
        keypoints0_normalized = normalizePositionArray(keypoints0, scene_range_all_with_margin)
        
        # Add to data_list
        if not args.add_bbox_data or bbox1_normalized is None:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "matches0": matches0,
            })
        elif args.add_bbox_data and not args.add_text_feature_data:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
            })
        else:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "text_embedding0": text_embedding0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "text_embedding1": text_embedding1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
                "frame_id": frame_id,
                "frame_scene": scene_folder,
                "scene_graph_id": scene_folder,
            })

        if args.generate_frame_pose_txt:
            # Add global pose data. Frame pose file e.g. frame-000000.pose.txt
            frame_id = json_file.split("_")[0]
            frame_id_six_digits = frame_id.zfill(6)
            frame_pose_file = os.path.join(frame_pose_dir, f"frame-{frame_id_six_digits}.pose.txt")
            with open(frame_pose_file, "r") as f:
                frame_pose = f.read()
            
            # Convert the pose matrix to a single line string
            pose_lines = frame_pose.strip().split('\n')
            pose_single_line = ' '.join(pose_lines)
            
            data_info_list.append(f"{scene_folder} {scene_folder} {frame_id} {pose_single_line}")
    
    return data_list, data_info_list


def generate_cross_matching_data(source_scene_folder, target_scene_folder, map_folder, frame_pose_dir, args,
                                keypoints1_normalized, descriptors1, bbox1_normalized, ids1, 
                                scene_range_all_with_margin, scene_x_range, scene_y_range, scene_z_range,
                                id_correction_dict, scan_transformation, text_embedding1=None, text_model=None, if_print=False):
    """
    Generate cross-matching data between two scenes.
    
    Args:
        source_scene_folder: The source scene folder (e.g., more_scans scene)
        target_scene_folder: The target scene folder (e.g., first scan scene)
        map_folder: The folder containing topology maps
        frame_pose_dir: Directory containing frame poses
        args: Command line arguments
        keypoints1_normalized: Normalized keypoints from target topology map
        descriptors1: Descriptors from target topology map
        bbox1_normalized: Normalized bbox data from target topology map
        ids1: Instance IDs from target topology map
        scene_range_all_with_margin: Scene range with margin for normalization
        scene_x_range, scene_y_range, scene_z_range: Current scene ranges to be updated
        id_correction_dict: ID correction dictionary (loaded in main)
        scan_transformation: Scan transformation matrix (loaded in main)
        text_embedding1: Text embeddings from target topology map (optional)
        text_model: Text model for encoding (optional)
    
    Returns:
        data_list: List of cross-matching data dictionaries
        data_info_list: List of cross-matching data info strings
        updated_scene_ranges: Updated scene ranges
    """
    data_list = []
    data_info_list = []

    dynamic_data_list = []
    dynamic_data_info_list = []

    if if_print:
        print(f"source_scene_folder: {source_scene_folder}")
        print(f"target_scene_folder: {target_scene_folder}")
    
    # Load the topology map of the source scene to find the position of the keypoints0 when graph_generated_position is True
    map_path = os.path.join(map_folder, source_scene_folder, "topology_map.json")
    with open(map_path, "r") as f:
        topology_map_source = TopologyMap()
        topology_map_source.read_from_json(f.read())
    
    valid_nodes_source = [node for node_id, node in topology_map_source.object_nodes.nodes.items() if node.name != "unknown"]
    keypoints_source = np.array([node.position for node in valid_nodes_source])
    ids_source = np.array([node.id for node in valid_nodes_source]).astype(int)
    
    # Correct the scene range all with keypoints_source
    scene_x_range_source = [min(keypoints_source[:, 0]), max(keypoints_source[:, 0])]
    scene_y_range_source = [min(keypoints_source[:, 1]), max(keypoints_source[:, 1])]
    scene_z_range_source = [min(keypoints_source[:, 2]), max(keypoints_source[:, 2])]
    scene_x_range, scene_y_range, scene_z_range = correctPositionRange(
        scene_x_range_source, scene_y_range_source, scene_z_range_source, 
        scene_x_range, scene_y_range, scene_z_range
    )
    scene_range_all = [min(scene_x_range[0], scene_y_range[0], scene_z_range[0]), 
                       max(scene_x_range[1], scene_y_range[1], scene_z_range[1])]
    scene_range_all_with_margin = addMarginToRange(scene_range_all, 0.1)
                  
    # Get the json files in the source scene folder
    json_files = [f for f in os.listdir(os.path.join(map_folder, source_scene_folder, "refined_instance")) if f.endswith("final_instance.json")]
    frame_pose_dir_source = os.path.join(frame_pose_dir, source_scene_folder)

    # Process each json file
    for json_file in json_files:
        json_path = os.path.join(map_folder, source_scene_folder, "refined_instance", json_file)
        if if_print:
            print(f"Cross-matching: Processing json file: {json_path}")
        frame_feature_dict = parseInstanceJson(json_path)

        frame_id = json_file.split("_")[0]
        
        # Only consider the frame with more than 2 instances
        frame_instance_num = len(frame_feature_dict)
        if frame_instance_num < args.min_instance_num_frame:
            continue

        names_this_scan = np.array([frame_feature_dict[instance_id]['object_name'] for instance_id in frame_feature_dict.keys()])
        
        # Initialize names0 for text embeddings
        names0 = []

        # Find the matching nodes in the topology map target by id
        descriptors0 = np.zeros((frame_instance_num, 256)) # 256 is the dimension of the visual embedding
        matches0 = (np.ones(frame_instance_num) * -1).astype(int)
        matches_this_scan = (np.ones(frame_instance_num) * -1).astype(int)

        if not args.graph_generated_position:
            keypoints0 = np.zeros((frame_instance_num, 3))

        if args.add_bbox_data:
            bbox0_height = np.zeros((frame_instance_num, 1))
            bbox0_width = np.zeros((frame_instance_num, 1))
            bbox0_length = np.zeros((frame_instance_num, 1))
            
        for i in range(frame_instance_num):
            instance_id = list(frame_feature_dict.keys())[i]
            descriptors0[i] = frame_feature_dict[instance_id]['feature']

            if not args.graph_generated_position:
                keypoints0[i] = [frame_feature_dict[instance_id]['center']['x'], frame_feature_dict[instance_id]['center']['y'], frame_feature_dict[instance_id]['center']['z']]

            if args.add_bbox_data:
                bbox0_height[i] = frame_feature_dict[instance_id]['bbox_size']['y']
                bbox0_width[i] = frame_feature_dict[instance_id]['bbox_size']['x']
                bbox0_length[i] = frame_feature_dict[instance_id]['bbox_size']['z']
                bbox0 = np.stack([bbox0_height, bbox0_width, bbox0_length], axis=1)
                bbox0 = np.squeeze(bbox0, axis=-1)
                bbox0_normalized = normalizeBboxArray(bbox0, scene_range_all_with_margin)

            if args.add_text_feature_data:
                names0.append(frame_feature_dict[instance_id]['object_name'])

            if instance_id in id_correction_dict:
                # Correct the instance id to the instance id in the target scene
                instance_id_in_target = id_correction_dict[instance_id]
                # Find the matching node in the topology map target
                if instance_id_in_target in ids1:
                    matches0[i] = np.where(ids1 == instance_id_in_target)[0][0]
                if instance_id in ids_source:
                    matches_this_scan[i] = np.where(ids_source == instance_id)[0][0]
        
        if args.add_text_feature_data:
            text_embedding0 = text_model.encode(names0)
            # print(f"text_embedding0: {text_embedding0.shape}")
            # print(f"text_embedding1: {text_embedding1.shape}")
            # exit()

        if args.graph_generated_position:
            #### Now set keypoints0 and descriptors0 using the matches_this_scan
            keypoints0 = np.array([keypoints_source[i] for i in matches_this_scan])
            min_keypoints0 = np.min(keypoints0, axis=0)
            keypoints0 = keypoints0 - min_keypoints0
            keypoints0 = keypoints0 + abs(np.random.normal(0, args.position_noise_std, keypoints0.shape))
        
        # Normalize the keypoints0
        keypoints0_normalized = normalizePositionArray(keypoints0, scene_range_all_with_margin)

        # Add to data_list
        if not args.add_bbox_data or bbox1_normalized is None:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "matches0": matches0,
            })
        elif args.add_bbox_data and not args.add_text_feature_data:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
            })
        else:
            data_list.append({
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "text_embedding0": text_embedding0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "text_embedding1": text_embedding1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
                "frame_id": frame_id,
                "frame_scene": target_scene_folder,
                "scene_graph_id": source_scene_folder,
            })

        data_info_list_to_add = []
        if args.generate_frame_pose_txt:
            # Add global pose data
            frame_id = json_file.split("_")[0]
            frame_id_six_digits = frame_id.zfill(6)
            frame_pose_file = os.path.join(frame_pose_dir_source, f"frame-{frame_id_six_digits}.pose.txt")
            with open(frame_pose_file, "r") as f:
                frame_pose = f.read()
            
            # Transform the frame pose to the target scene
            frame_pose = np.fromstring(frame_pose, dtype=np.float32, sep=' ').reshape(4, 4)
            frame_pose = np.matmul(scan_transformation, frame_pose)
            frame_pose = frame_pose.flatten().astype(str)
            pose_single_line = ' '.join(frame_pose)
            data_info_list_to_add = f"{target_scene_folder} {source_scene_folder} {frame_id} {pose_single_line}"
            data_info_list.append(data_info_list_to_add)

        if args.add_dynamic_data:
                # keypoints0_normalized_dynamic = keypoints0_normalized.copy()
                # descriptors0_dynamic = descriptors0.copy()
                # matches0_dynamic = matches0.copy()
                keypoints0_normalized_dynamic = []
                descriptors0_dynamic = []
                matches0_dynamic = []
                
                # Initialize dynamic bbox and text embedding lists
                bbox0_normalized_dynamic = []
                text_embedding0_dynamic = []
                
                for i in range(frame_instance_num):
                    # 0.2 probability to move keypoints0 to a random direction by a Gaussian distribution. Std is max_bbox_size * 1.5
                    if np.random.rand() < 0.2:
                        stddev = np.max(bbox0_normalized[i]) * 1.5 # Take 1.5 times the max bbox size as the stddev for the random direction
                        random_direction = np.random.normal(0, stddev, keypoints0_normalized[i].shape)
                        # keypoints0_normalized_dynamic[i] = keypoints0_normalized_dynamic[i] + random_direction
                        # keypoints0_normalized_dynamic[i] = np.clip(keypoints0_normalized_dynamic[i], 0, 1)
                        keypoint_normalized = keypoints0_normalized[i] + random_direction
                        keypoint_normalized = np.clip(keypoint_normalized, 0, 1)
                        keypoints0_normalized_dynamic.append(keypoint_normalized)
                        descriptors0_dynamic.append(descriptors0[i])
                        matches0_dynamic.append(matches0[i])
                        
                        # Add corresponding bbox and text embedding data
                        if args.add_bbox_data:
                            bbox0_normalized_dynamic.append(bbox0_normalized[i])
                        if args.add_text_feature_data:
                            text_embedding0_dynamic.append(text_embedding0[i])
                        continue

                    # 0.2 probability to remove the instance if the instance is not moved
                    if np.random.rand() < 0.2:
                        continue
                    else:
                        # No change
                        keypoints0_normalized_dynamic.append(keypoints0_normalized[i])
                        descriptors0_dynamic.append(descriptors0[i])
                        matches0_dynamic.append(matches0[i])
                        
                        # Add corresponding bbox and text embedding data
                        if args.add_bbox_data:
                            bbox0_normalized_dynamic.append(bbox0_normalized[i])
                        if args.add_text_feature_data:
                            text_embedding0_dynamic.append(text_embedding0[i])
                
                # Add only frames with more than args.min_instance_num_frame instances
                if len(keypoints0_normalized_dynamic) >= args.min_instance_num_frame:
                    # Create dynamic data dictionary based on available features
                    if not args.add_bbox_data or bbox1_normalized is None:
                        dynamic_data_list.append({
                            "keypoints0": keypoints0_normalized_dynamic,
                            "descriptors0": descriptors0_dynamic,
                            "keypoints1": keypoints1_normalized,
                            "descriptors1": descriptors1,
                            "matches0": matches0_dynamic,
                        })
                    elif args.add_bbox_data and not args.add_text_feature_data:
                        dynamic_data_list.append({
                            "keypoints0": keypoints0_normalized_dynamic,
                            "descriptors0": descriptors0_dynamic,
                            "bbox0": bbox0_normalized_dynamic,
                            "keypoints1": keypoints1_normalized,
                            "descriptors1": descriptors1,
                            "bbox1": bbox1_normalized,
                            "matches0": matches0_dynamic,
                        })
                    else:
                        dynamic_data_list.append({
                            "keypoints0": keypoints0_normalized_dynamic,
                            "descriptors0": descriptors0_dynamic,
                            "bbox0": bbox0_normalized_dynamic,
                            "text_embedding0": text_embedding0_dynamic,
                            "keypoints1": keypoints1_normalized,
                            "descriptors1": descriptors1,
                            "text_embedding1": text_embedding1,
                            "bbox1": bbox1_normalized,
                            "matches0": matches0_dynamic,
                            "frame_id": frame_id,
                            "frame_scene": target_scene_folder,
                            "scene_graph_id": source_scene_folder,
                        })

                    if args.generate_frame_pose_txt:
                        # The same as data_info_list_to_add 
                        dynamic_data_info_list.append(data_info_list_to_add)

    updated_scene_ranges = (scene_x_range, scene_y_range, scene_z_range)
    return data_list, data_info_list, dynamic_data_list, dynamic_data_info_list, updated_scene_ranges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_folder", type=str, help="The folder that contains scenes with topology maps", default="/media/cc/Expansion/scannet/processed/scans")
    parser.add_argument("--frame_pose_dir", type=str, help="The directory that contains the frame global pose", default="/media/cc/My Passport/dataset/scannet/images/scans")
    
    parser.add_argument("--graph_generated_position", action="store_true", help="Use the graph generated position instead of the observed position in each frame")
    parser.add_argument("--position_noise_std", type=float, help="The standard deviation of the position noise", default=0.3)

    parser.add_argument("--data_output_dir", type=str, help="The output directory", default="/media/cc/Expansion/scannet/processed")
    parser.add_argument("--add_cross_scan", type=bool, help="Add cross-scan data", default=True)
    parser.add_argument("--add_dynamic_data", type=bool, help="Add the dynamic data", default=True)
    parser.add_argument("--add_bbox_data", type=bool, help="Add the bbox data", default=True)
    parser.add_argument("--add_text_feature_data", type=bool, help="Add the text feature data", default=True)

    parser.add_argument("--generate_frame_pose_txt", type=bool, help="Generate the frame pose txt file", default=True)

    parser.add_argument("--scene_exclude_csv_path", type=str, help="The path to the scene csv file to exclude", default="/media/cc/Expansion/scannet/processed/excluded.csv")
    parser.add_argument("--save_every_n_scenes", type=int, help="Save the data every n scenes", default=100)
    parser.add_argument("--min_instance_num_frame", type=int, help="The minimum number of instances in a frame", default=2)
    
    parser.add_argument("--add_cross_scan_from_first_scan", type=bool, help="Add the cross-scan data from the first scan", default=False)
    
    args = parser.parse_args()


    print(f"args: {args}")

    if args.add_dynamic_data:
        if not args.add_bbox_data:
            raise ValueError("add_dynamic_data requires add_bbox_data to be True")

    excluded_scene_folders = []
    if args.scene_exclude_csv_path:
        excluded_scene_folders = alignment_examine.load_csv(args.scene_exclude_csv_path)
        print(f"Found {len(excluded_scene_folders)} excluded scene folders")
    
    # Get all the scene folders
    scene_folders = [f for f in os.listdir(args.map_folder) if os.path.isdir(os.path.join(args.map_folder, f))]
    print(f"Found {len(scene_folders)} scene folders")

    # Split the scene folders into first scans and more scans
    first_scans = [f for f in scene_folders if f.endswith("_00")]
    more_scans = [f for f in scene_folders if not f.endswith("_00")]
    print(f"Found {len(first_scans)} first scans and {len(more_scans)} more scans")

    # Sort the scene folders by the scene id
    first_scans.sort(key=lambda x: int(x.split("_")[0].split("scene")[1]))
    more_scans.sort(key=lambda x: int(x.split("_")[0].split("scene")[1]))

    # Initialization
    feature_dim_all = 256
    data_list_self_matching = []
    data_info_list_self_matching = []
    data_list_cross_matching = []
    data_list_cross_matching_from_first_scan = []
    data_info_list_cross_matching = []
    data_info_list_cross_matching_from_first_scan = []
    dynamic_data_list = []
    dynamic_data_info_list = []
    current_scene_seq = 0
    start_scene_num = -1

    if args.add_text_feature_data:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        text_model = None

    for scene_folder in tqdm(first_scans): # Generate self-matching data for only first scans
        print(f"Processing scene {scene_folder}")
        map_path = os.path.join(args.map_folder, scene_folder, "topology_map.json")
        
        current_scene_num = int(scene_folder.split("_")[0].split("scene")[1])
        if start_scene_num == -1:
            start_scene_num = current_scene_num
            
        current_scene_seq += 1

        # Check if the map file exists
        if not os.path.exists(map_path):
            print(f"Warning: Map file not found for scene {scene_folder}")
            continue

        # Read the topology map and let keypoints0 be the nodes in the topology map that are not named "unknown"
        with open(map_path, "r") as f:
            topology_map = TopologyMap()
            topology_map.read_from_json(f.read())
        
        # Get the nodes that are not named "unknown"
        valid_nodes = [node for node_id, node in topology_map.object_nodes.nodes.items() if node.name != "unknown"]
        keypoints1 = np.array([node.position for node in valid_nodes])
        descriptors1 = np.array([node.visual_embedding for node in valid_nodes])
        names1 = np.array([node.name for node in valid_nodes])
        
        # Continue if the descriptors1 is empty
        if descriptors1.shape[0] == 0:
            print(f"Warning: No valid nodes found for scene {scene_folder}")
            continue
        
        # Normalize the position
        scene_x_range = [min(keypoints1[:, 0]), max(keypoints1[:, 0])]
        scene_y_range = [min(keypoints1[:, 1]), max(keypoints1[:, 1])]
        scene_z_range = [min(keypoints1[:, 2]), max(keypoints1[:, 2])]
        scene_range_all = [min(scene_x_range[0], scene_y_range[0], scene_z_range[0]), max(scene_x_range[1], scene_y_range[1], scene_z_range[1])]
        scene_range_all_with_margin = addMarginToRange(scene_range_all, 0.1)
        keypoints1_normalized = normalizePositionArray(keypoints1, scene_range_all_with_margin)
        
        # Add the bbox data if args.add_bbox_data is True
        if args.add_bbox_data:
            bbox1_height = np.array([node.shape.height for node in valid_nodes])
            bbox1_width = np.array([node.shape.width for node in valid_nodes])
            bbox1_length = np.array([node.shape.length for node in valid_nodes])
            # Stack the bbox1_height, bbox1_width, bbox1_length to bbox1 and normalize the bbox1
            bbox1 = np.stack([bbox1_height, bbox1_width, bbox1_length], axis=1)
            bbox1_normalized = normalizeBboxArray(bbox1, scene_range_all_with_margin)
        
        if args.add_text_feature_data:
            text_embedding1 = text_model.encode(names1)
        else:
            text_embedding1 = None

        feature_dim_all = descriptors1.shape[1]
        ids1 = np.array([node.id for node in valid_nodes]).astype(int)

        ########### Add self-matching data ###########
        frame_pose_dir = os.path.join(args.frame_pose_dir, scene_folder)
        self_matching_data, self_matching_info = generate_self_matching_data(
            scene_folder, args.map_folder, frame_pose_dir, args,
            keypoints1_normalized, descriptors1, bbox1_normalized if args.add_bbox_data else None,
            ids1, scene_range_all_with_margin, text_embedding1=text_embedding1, text_model=text_model
        )
        data_list_self_matching.extend(self_matching_data)
        data_info_list_self_matching.extend(self_matching_info)

        ########### Add more scans cross-matching data ###########
        if scene_folder in first_scans and args.add_cross_scan:
            for more_scans_scene_folder in more_scans:
                scene_id = more_scans_scene_folder.split("_")[0]
                if scene_id != scene_folder.split("_")[0]:
                    continue
                else:
                    print(f"Found more scans scene {more_scans_scene_folder} for scene {scene_folder}")
                    if more_scans_scene_folder in excluded_scene_folders:
                        print(f"Skipping {more_scans_scene_folder} because it is in the excluded scene folders")
                        continue
                    
                    # Load the id correction csv file and transformation matrix for this more_scans scene
                    id_correction_csv_path = os.path.join(args.map_folder, more_scans_scene_folder, "matched_instance_correspondence_to_00.csv")
                    id_correction_dict = loadIdCorrectionCsv(id_correction_csv_path)
                    
                    scan_transformation_path = os.path.join(args.map_folder, more_scans_scene_folder, "transformation.npy")
                    if os.path.exists(scan_transformation_path):
                        scan_transformation = np.load(scan_transformation_path)
                        inv_scan_transformation = np.linalg.inv(scan_transformation)
                    else:
                        raise ValueError(f"Transformation matrix not found for scene {more_scans_scene_folder}")
                    
                    # Generate cross-matching data from more_scans to first_scan
                    cross_matching_data, cross_matching_info, dynamic_data, dynamic_data_info, updated_scene_ranges = generate_cross_matching_data(
                        more_scans_scene_folder, scene_folder, args.map_folder, args.frame_pose_dir, args,
                        keypoints1_normalized, descriptors1, bbox1_normalized if args.add_bbox_data else None,
                        ids1, scene_range_all_with_margin, scene_x_range, scene_y_range, scene_z_range,
                        id_correction_dict, inv_scan_transformation, text_embedding1=text_embedding1, text_model=text_model
                    )
                    
                    data_list_cross_matching.extend(cross_matching_data)
                    data_info_list_cross_matching.extend(cross_matching_info)

                    if args.add_dynamic_data:
                        dynamic_data_list.extend(dynamic_data)
                        dynamic_data_info_list.extend(dynamic_data_info)

                    # Update scene ranges
                    scene_x_range, scene_y_range, scene_z_range = updated_scene_ranges

        ########### Add first scan to more scans cross-matching data ###########
        if scene_folder in first_scans and args.add_cross_scan and args.add_cross_scan_from_first_scan:
            for more_scans_scene_folder in more_scans:
                scene_id = more_scans_scene_folder.split("_")[0]
                if scene_id != scene_folder.split("_")[0]:
                    continue
                else:
                    print(f"Found more scans scene {more_scans_scene_folder} for scene {scene_folder}")
                    if more_scans_scene_folder in excluded_scene_folders:
                        print(f"Skipping {more_scans_scene_folder} because it is in the excluded scene folders")
                        continue
                    
                    # Use the already loaded id_correction_dict and scan_transformation from the previous loop
                    # Load the topology map of the more_scans scene to get keypoints1, descriptors1, etc.
                    map_path_more_scans = os.path.join(args.map_folder, more_scans_scene_folder, "topology_map.json")
                    with open(map_path_more_scans, "r") as f:
                        topology_map_more_scans = TopologyMap()
                        topology_map_more_scans.read_from_json(f.read())

                    # Load the id correction csv file and transformation matrix for this more_scans scene
                    id_correction_csv_path = os.path.join(args.map_folder, more_scans_scene_folder, "matched_instance_correspondence_to_00.csv")
                    id_correction_dict = loadIdCorrectionCsv(id_correction_csv_path)
                    id_correction_dict_reverse = {v: k for k, v in id_correction_dict.items()}

                    scan_transformation_path = os.path.join(args.map_folder, more_scans_scene_folder, "transformation.npy")
                    if os.path.exists(scan_transformation_path):
                        scan_transformation = np.load(scan_transformation_path)
                    else:
                        raise ValueError(f"Transformation matrix not found for scene {more_scans_scene_folder}")
                    
                    # Get the nodes that are not named "unknown" from more_scans topology map
                    valid_nodes_more_scans = [node for node_id, node in topology_map_more_scans.object_nodes.nodes.items() if node.name != "unknown"]
                    keypoints1_more_scans = np.array([node.position for node in valid_nodes_more_scans])
                    descriptors1_more_scans = np.array([node.visual_embedding for node in valid_nodes_more_scans])
                    ids1_more_scans = np.array([node.id for node in valid_nodes_more_scans]).astype(int)
                    
                    # Normalize the keypoints1 from more_scans topology map
                    scene_x_range_more_scans = [min(keypoints1_more_scans[:, 0]), max(keypoints1_more_scans[:, 0])]
                    scene_y_range_more_scans = [min(keypoints1_more_scans[:, 1]), max(keypoints1_more_scans[:, 1])]
                    scene_z_range_more_scans = [min(keypoints1_more_scans[:, 2]), max(keypoints1_more_scans[:, 2])]
                    scene_x_range, scene_y_range, scene_z_range = correctPositionRange(
                        scene_x_range_more_scans, scene_y_range_more_scans, scene_z_range_more_scans, 
                        scene_x_range, scene_y_range, scene_z_range
                    )
                    scene_range_all_more_scans = [min(scene_x_range[0], scene_y_range[0], scene_z_range[0]), 
                                                max(scene_x_range[1], scene_y_range[1], scene_z_range[1])]
                    scene_range_all_with_margin_more_scans = addMarginToRange(scene_range_all_more_scans, 0.1)
                    keypoints1_normalized_more_scans = normalizePositionArray(keypoints1_more_scans, scene_range_all_with_margin_more_scans)
                    
                    # Add the bbox data if args.add_bbox_data is True
                    if args.add_bbox_data:
                        bbox1_height_more_scans = np.array([node.shape.height for node in valid_nodes_more_scans])
                        bbox1_width_more_scans = np.array([node.shape.width for node in valid_nodes_more_scans])
                        bbox1_length_more_scans = np.array([node.shape.length for node in valid_nodes_more_scans])
                        bbox1_more_scans = np.stack([bbox1_height_more_scans, bbox1_width_more_scans, bbox1_length_more_scans], axis=1)
                        bbox1_more_scans = np.squeeze(bbox1_more_scans, axis=-1)
                        bbox1_normalized_more_scans = normalizeBboxArray(bbox1_more_scans, scene_range_all_with_margin_more_scans)
                    else:
                        bbox1_normalized_more_scans = None
                    
                    # Add text embedding data if args.add_text_feature_data is True
                    if args.add_text_feature_data:
                        names1_more_scans = np.array([node.name for node in valid_nodes_more_scans])
                        text_embedding1_more_scans = text_model.encode(names1_more_scans)
                    else:
                        text_embedding1_more_scans = None
                    
                    # Generate cross-matching data from first_scan to more_scans
                    cross_matching_data, cross_matching_info, dynamic_data, dynamic_data_info, updated_scene_ranges = generate_cross_matching_data(
                        scene_folder, more_scans_scene_folder, args.map_folder, args.frame_pose_dir, args,
                        keypoints1_normalized_more_scans, descriptors1_more_scans, bbox1_normalized_more_scans,
                        ids1_more_scans, scene_range_all_with_margin_more_scans, scene_x_range, scene_y_range, scene_z_range,
                        id_correction_dict_reverse, scan_transformation, text_embedding1=text_embedding1_more_scans, text_model=text_model
                    )
                    data_list_cross_matching_from_first_scan.extend(cross_matching_data)
                    data_info_list_cross_matching_from_first_scan.extend(cross_matching_info)

                    # if args.add_dynamic_data: #Ignore dynamic data from first scan to more scans for now
                    #     dynamic_data_list.extend(dynamic_data)
                    #     dynamic_data_info_list.extend(dynamic_data_info)

                    # Update scene ranges
                    scene_x_range, scene_y_range, scene_z_range = updated_scene_ranges

        # Show the first data in the data_list_self_matching and data_list_cross_matching
        if current_scene_seq % args.save_every_n_scenes == 0 or current_scene_seq == len(first_scans):
            print(f"Showing the first data in the data_list_self_matching and data_list_cross_matching")
            print(f"Data length self-matching: {len(data_list_self_matching)}")
            if data_list_self_matching:
                print(f"Data self-matching: {data_list_self_matching[0]}")
                   
        if current_scene_seq % args.save_every_n_scenes == 0 or current_scene_seq == len(first_scans):
            print(f"Saving data to {args.data_output_dir}/data_list_{start_scene_num}_{current_scene_num}.pkl")
            print(f"Data length self-matching: {len(data_list_self_matching)}")
            print(f"Data length cross-matching: {len(data_list_cross_matching)}")
            with open(os.path.join(args.data_output_dir, f"data_list_self_matching_{start_scene_num}_{current_scene_num}.pkl"), "wb") as f:
                pickle.dump(data_list_self_matching, f)
            with open(os.path.join(args.data_output_dir, f"data_list_cross_matching_{start_scene_num}_{current_scene_num}.pkl"), "wb") as f:
                pickle.dump(data_list_cross_matching, f)
            
            if args.add_cross_scan_from_first_scan:
                with open(os.path.join(args.data_output_dir, f"data_list_cross_matching_from_first_scan_{start_scene_num}_{current_scene_num}.pkl"), "wb") as f:
                    pickle.dump(data_list_cross_matching_from_first_scan, f)

            if args.generate_frame_pose_txt:
                # Save the data info list to a txt file
                with open(os.path.join(args.data_output_dir, f"data_pose_list_self_matching_{start_scene_num}_{current_scene_num}.txt"), "w") as f:
                    for data_info in data_info_list_self_matching:
                        f.write(data_info + "\n")
                with open(os.path.join(args.data_output_dir, f"data_pose_list_cross_matching_{start_scene_num}_{current_scene_num}.txt"), "w") as f:
                    for data_info in data_info_list_cross_matching:
                        f.write(data_info + "\n")
                if args.add_cross_scan_from_first_scan:
                    with open(os.path.join(args.data_output_dir, f"data_pose_list_cross_matching_from_first_scan_{start_scene_num}_{current_scene_num}.txt"), "w") as f:
                        for data_info in data_info_list_cross_matching_from_first_scan:
                            f.write(data_info + "\n")

            if args.add_dynamic_data:
                with open(os.path.join(args.data_output_dir, f"data_list_dynamic_{start_scene_num}_{current_scene_num}.pkl"), "wb") as f:
                    pickle.dump(dynamic_data_list, f)
                if args.generate_frame_pose_txt:
                    with open(os.path.join(args.data_output_dir, f"data_pose_list_dynamic_{start_scene_num}_{current_scene_num}.txt"), "w") as f:
                        for data_info in dynamic_data_info_list:
                            f.write(data_info + "\n")

            # Check the range of the keypoints0 and keypoints1 to correct the position ranges
            print("Correcting the position ranges by checking the keypoints0 and keypoints1 ...")
            for data in data_list_self_matching + data_list_cross_matching:
                keypoints0 = data["keypoints0"]
                keypoints1 = data["keypoints1"]

            # Reset the data list
            data_list_self_matching = []
            data_list_cross_matching = []
            data_list_cross_matching_from_first_scan = []
            data_info_list_self_matching = []
            data_info_list_cross_matching = []
            data_info_list_cross_matching_from_first_scan = []
            dynamic_data_list = []
            dynamic_data_info_list = []
            start_scene_num = -1 # Will be reset in the next iteration

    # Save the position ranges to a json file
    meta_data_save_path = os.path.join(args.data_output_dir, "meta_data.json")
    with open(meta_data_save_path, "w") as f:
        json.dump({
            "x": [0, 1],
            "y": [0, 1],
            "z": [0, 1],
            "feature_dim": feature_dim_all
        }, f)
    print(f"Saved position ranges to {meta_data_save_path}")

