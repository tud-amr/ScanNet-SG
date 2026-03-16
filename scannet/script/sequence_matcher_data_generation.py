import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "..", "..", "script"))

from include.topology_map import *
import alignment_examine

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


def calculateBboxVolume(bbox_size):
    """
    Calculate the volume of a bounding box.
    """
    return bbox_size['x'] * bbox_size['y'] * bbox_size['z']


def mergeSequenceObjects(frame_sequence_data):
    """
    Merge objects with the same instance_id across a sequence of frames.
    
    Args:
        frame_sequence_data: List of dictionaries, each containing frame_feature_dict from parseInstanceJson
    
    Returns:
        merged_objects: Dictionary mapping instance_id to merged object data
    """
    merged_objects = {}
    
    # Group objects by instance_id across all frames
    instance_groups = {}
    for frame_data in frame_sequence_data:
        for instance_id, obj_data in frame_data.items():
            if instance_id not in instance_groups:
                instance_groups[instance_id] = []
            instance_groups[instance_id].append(obj_data)
    
    # Merge each group
    for instance_id, obj_list in instance_groups.items():
        # Average the feature vectors
        features = np.array([obj['feature'] for obj in obj_list])
        avg_feature = np.mean(features, axis=0)
        
        # Find the object with the biggest bbox
        max_volume = -1
        best_obj = None
        for obj in obj_list:
            volume = calculateBboxVolume(obj['bbox_size'])
            if volume > max_volume:
                max_volume = volume
                best_obj = obj
        
        # Use bbox, center, and object_name from the one with biggest bbox
        merged_objects[instance_id] = {
            'feature': avg_feature,
            'object_name': best_obj['object_name'],
            'confidence': best_obj['confidence'],  # Use confidence from biggest bbox
            'bbox_size': best_obj['bbox_size'],
            'center': best_obj['center']
        }
    
    return merged_objects


def selectFrameSequences(json_files, num_sequences, min_frames=50, max_frames=200):
    """
    Randomly select consistent (consecutive) frame sequences from json files.
    
    Args:
        json_files: List of json file names
        num_sequences: Number of sequences to select
        min_frames: Minimum frames per sequence
        max_frames: Maximum frames per sequence
    
    Returns:
        sequences: List of lists, each containing json file names for consecutive frames
    """
    if len(json_files) == 0:
        return []
    
    # Extract frame IDs and sort them to ensure consecutive sequences
    frame_ids = []
    for json_file in json_files:
        frame_id_str = json_file.split("_")[0]
        try:
            frame_id_int = int(frame_id_str)
            frame_ids.append((frame_id_int, json_file))
        except ValueError:
            continue
    
    # Sort by frame ID to ensure we can create consecutive sequences
    frame_ids.sort(key=lambda x: x[0])
    
    if len(frame_ids) < min_frames:
        return []
    
    sequences = []
    
    for _ in range(num_sequences):
        if len(frame_ids) < min_frames:
            break
        
        # Randomly select sequence length
        seq_length = random.randint(min_frames, min(max_frames, len(frame_ids)))
        
        # Randomly select starting position (ensuring we can get consecutive frames)
        max_start = len(frame_ids) - seq_length
        if max_start < 0:
            break
        
        start_idx = random.randint(0, max_start)
        
        # Create consecutive sequence
        sequence = [frame_ids[start_idx + i][1] for i in range(seq_length)]
        sequences.append(sequence)
    
    return sequences


def generateMapLevelMatchingData(source_scene_folder, target_scene_folder, map_folder, args,
                                  keypoints1_normalized, descriptors1, bbox1_normalized, ids1,
                                  scene_range_all_with_margin, id_correction_dict,
                                  text_embedding1=None, text_model=None, names1=None):
    """
    Generate map-level cross-matching data using topology map nodes from more scans.
    
    Args:
        source_scene_folder: The source scene folder (more_scans scene)
        target_scene_folder: The target scene folder (first scan scene)
        map_folder: The folder containing topology maps
        args: Command line arguments
        keypoints1_normalized: Normalized keypoints from target topology map
        descriptors1: Descriptors from target topology map
        bbox1_normalized: Normalized bbox data from target topology map
        ids1: Instance IDs from target topology map
        scene_range_all_with_margin: Scene range with margin for normalization
        id_correction_dict: ID correction dictionary
        text_embedding1: Text embeddings from target topology map (optional)
        text_model: Text model for encoding (optional)
        names1: Names from target topology map (optional)
    Returns:
        data_dict: A single matching data dictionary or None
    """
    # Read the topology map of the source scene (more_scans)
    source_map_path = os.path.join(map_folder, source_scene_folder, "topology_map.json")
    if not os.path.exists(source_map_path):
        print(f"Warning: Map file not found for source scene {source_scene_folder}")
        return None
    
    with open(source_map_path, "r") as f:
        source_topology_map = TopologyMap()
        source_topology_map.read_from_json(f.read())
    
    # Get the nodes that are not named "unknown"
    valid_source_nodes = [node for node_id, node in source_topology_map.object_nodes.nodes.items() 
                          if node.name != "unknown"]
    
    if len(valid_source_nodes) < args.min_instance_num_frame:
        print(f"Warning: Not enough valid nodes in source scene {source_scene_folder}")
        return None
    
    # Prepare data arrays
    num_instances = len(valid_source_nodes)
    keypoints0 = np.array([node.position for node in valid_source_nodes])
    descriptors0 = np.array([node.visual_embedding for node in valid_source_nodes])
    source_ids = np.array([node.id for node in valid_source_nodes]).astype(int)
    matches0 = (np.ones(num_instances) * -1).astype(int)
    
    if args.add_bbox_data:
        bbox0_height = np.array([node.shape.height for node in valid_source_nodes])
        bbox0_width = np.array([node.shape.width for node in valid_source_nodes])
        bbox0_length = np.array([node.shape.length for node in valid_source_nodes])
        bbox0 = np.stack([bbox0_height, bbox0_width, bbox0_length], axis=1)
        bbox0_normalized = normalizeBboxArray(bbox0, scene_range_all_with_margin)
    
    if args.add_text_feature_data:
        names0 = [node.name for node in valid_source_nodes]
        text_embedding0 = text_model.encode(names0)
    
    # Normalize keypoints
    keypoints0_normalized = normalizePositionArray(keypoints0, scene_range_all_with_margin)
    
    # Build matches using id_correction_dict
    for i, source_id in enumerate(source_ids):
        if source_id in id_correction_dict:
            target_id = id_correction_dict[source_id]
            if target_id in ids1:
                matches0[i] = np.where(ids1 == target_id)[0][0]
    
    # Create matches1 (reverse matches)
    matches1 = (np.ones(len(ids1)) * -1).astype(int)
    for i, match_idx in enumerate(matches0):
        if match_idx >= 0:
            matches1[match_idx] = i
    
    # Create the data dictionary
    data_dict = {
        "keypoints0": keypoints0_normalized,
        "keypoints1": keypoints1_normalized,
        "descriptors0": descriptors0,
        "descriptors1": descriptors1,
        "matches0": matches0,
        "matches1": matches1,
        "source_scene": source_scene_folder,
        "target_scene": target_scene_folder,
    }
    
    if args.add_bbox_data:
        data_dict["bbox0"] = bbox0_normalized
        data_dict["bbox1"] = bbox1_normalized
    
    if args.add_text_feature_data:
        data_dict["text_embedding0"] = text_embedding0
        data_dict["text_embedding1"] = text_embedding1
    
    return data_dict


def generateSequenceMatchingData(source_scene_folder, target_scene_folder, map_folder, args,
                                 keypoints1_normalized, descriptors1, bbox1_normalized, ids1,
                                 scene_range_all_with_margin, id_correction_dict, 
                                 text_embedding1=None, text_model=None, names1=None, sequence_ratio=0.3):
    """
    Generate sequence-based cross-matching data between two scenes.
    
    Args:
        source_scene_folder: The source scene folder (more_scans scene)
        target_scene_folder: The target scene folder (first scan scene)
        map_folder: The folder containing topology maps
        args: Command line arguments
        keypoints1_normalized: Normalized keypoints from target topology map
        descriptors1: Descriptors from target topology map
        bbox1_normalized: Normalized bbox data from target topology map
        ids1: Instance IDs from target topology map
        scene_range_all_with_margin: Scene range with margin for normalization
        id_correction_dict: ID correction dictionary
        text_embedding1: Text embeddings from target topology map (optional)
        text_model: Text model for encoding (optional)
        names1: Names from target topology map (optional)
    Returns:
        data_list: List of sequence matching data dictionaries
    """
    data_list = []
    
    # Get all json files in the source scene folder
    refined_instance_folder = os.path.join(map_folder, source_scene_folder, "refined_instance")
    if not os.path.exists(refined_instance_folder):
        print(f"Warning: Refined instance folder not found: {refined_instance_folder}")
        return data_list
    
    json_files = [f for f in os.listdir(refined_instance_folder) if f.endswith("final_instance.json")]
    
    # Filter valid frames (with at least min_instance_num_frame instances)
    valid_json_files = []
    for json_file in json_files:
        json_path = os.path.join(refined_instance_folder, json_file)
        frame_feature_dict = parseInstanceJson(json_path)
        if len(frame_feature_dict) >= args.min_instance_num_frame:
            valid_json_files.append(json_file)
    
    if len(valid_json_files) == 0:
        print(f"Warning: No valid frames found for scene {source_scene_folder}")
        return data_list
    
    # Calculate number of sequences to select (sequence_ratio * N)
    num_sequences = max(1, int(sequence_ratio * len(valid_json_files)))
    
    # Select frame sequences
    sequences = selectFrameSequences(valid_json_files, num_sequences, min_frames=args.min_frames, max_frames=args.max_frames)
    
    print(f"Selected {len(sequences)} sequences from {len(valid_json_files)} valid frames in {source_scene_folder}")
    
    # Process each sequence
    for seq_idx, sequence in enumerate(sequences):
        # Load all frames in the sequence
        frame_sequence_data = []
        for json_file in sequence:
            json_path = os.path.join(refined_instance_folder, json_file)
            frame_feature_dict = parseInstanceJson(json_path)
            if len(frame_feature_dict) > 0:
                frame_sequence_data.append(frame_feature_dict)
        
        if len(frame_sequence_data) == 0:
            continue
        
        # Merge objects with same instance_id
        merged_objects = mergeSequenceObjects(frame_sequence_data)
        
        if len(merged_objects) < args.min_instance_num_frame:
            continue
        
        # Prepare data arrays
        num_instances = len(merged_objects)
        descriptors0 = np.zeros((num_instances, 256))  # 256 is the dimension of the visual embedding
        matches0 = (np.ones(num_instances) * -1).astype(int)
        keypoints0 = np.zeros((num_instances, 3))  # Initialize for both cases
        
        if args.add_bbox_data:
            bbox0_height = np.zeros((num_instances, 1))
            bbox0_width = np.zeros((num_instances, 1))
            bbox0_length = np.zeros((num_instances, 1))
        
        names0 = []
        
        # Process each merged object
        for i, (instance_id, obj_data) in enumerate(merged_objects.items()):
            descriptors0[i] = obj_data['feature']
            
            if not args.graph_generated_position:
                keypoints0[i] = [
                    obj_data['center']['x'],
                    obj_data['center']['y'],
                    obj_data['center']['z']
                ]
            
            if args.add_bbox_data:
                bbox0_height[i] = obj_data['bbox_size']['y']
                bbox0_width[i] = obj_data['bbox_size']['x']
                bbox0_length[i] = obj_data['bbox_size']['z']
            
            if args.add_text_feature_data:
                names0.append(obj_data['object_name'])
            
            # Map instance_id to target scene using id_correction_dict
            if instance_id in id_correction_dict:
                instance_id_in_target = id_correction_dict[instance_id]
                if instance_id_in_target in ids1:
                    matches0[i] = np.where(ids1 == instance_id_in_target)[0][0]
        
        if args.add_bbox_data:
            bbox0 = np.stack([bbox0_height, bbox0_width, bbox0_length], axis=1)
            bbox0 = np.squeeze(bbox0, axis=-1)
            bbox0_normalized = normalizeBboxArray(bbox0, scene_range_all_with_margin)
        
        if args.add_text_feature_data:
            text_embedding0 = text_model.encode(names0)
        
        if args.graph_generated_position:
            # For graph generated position, we need to load the source scene topology map
            map_path = os.path.join(map_folder, source_scene_folder, "topology_map.json")
            if os.path.exists(map_path):
                with open(map_path, "r") as f:
                    topology_map_source = TopologyMap()
                    topology_map_source.read_from_json(f.read())
                
                valid_nodes_source = [node for node_id, node in topology_map_source.object_nodes.nodes.items() if node.name != "unknown"]
                keypoints_source = np.array([node.position for node in valid_nodes_source])
                ids_source = np.array([node.id for node in valid_nodes_source]).astype(int)
                
                # Map to source scene topology
                matches_source = (np.ones(num_instances) * -1).astype(int)
                for i, (instance_id, _) in enumerate(merged_objects.items()):
                    if instance_id in ids_source:
                        matches_source[i] = np.where(ids_source == instance_id)[0][0]
                
                # Set keypoints0 using matches_source
                keypoints0 = np.array([keypoints_source[i] if i >= 0 else np.zeros(3) for i in matches_source])
                # Find minimum of non-zero keypoints
                valid_mask = np.any(keypoints0 != 0, axis=1)
                if np.any(valid_mask):
                    min_keypoints0 = np.min(keypoints0[valid_mask], axis=0)
                else:
                    min_keypoints0 = np.zeros(3)
                keypoints0 = keypoints0 - min_keypoints0
                # keypoints0 = keypoints0 + abs(np.random.normal(0, args.position_noise_std, keypoints0.shape))
        
        # Normalize keypoints0
        keypoints0_normalized = normalizePositionArray(keypoints0, scene_range_all_with_margin)
        
        # Create data dictionary
        num_frames_in_sequence = len(sequence)
        if not args.add_bbox_data or bbox1_normalized is None:
            data_dict = {
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "matches0": matches0,
                "num_frames": num_frames_in_sequence,
            }
        elif args.add_bbox_data and not args.add_text_feature_data:
            data_dict = {
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
                "num_frames": num_frames_in_sequence,
            }
        else:
            data_dict = {
                "keypoints0": keypoints0_normalized,
                "descriptors0": descriptors0,
                "bbox0": bbox0_normalized,
                "text_embedding0": text_embedding0,
                "keypoints1": keypoints1_normalized,
                "descriptors1": descriptors1,
                "text_embedding1": text_embedding1,
                "bbox1": bbox1_normalized,
                "matches0": matches0,
                "frame_scene": target_scene_folder,
                "scene_graph_id": source_scene_folder,
                "sequence_id": seq_idx,
                "num_frames": num_frames_in_sequence,
            }        

        data_list.append(data_dict)
    
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_folder", type=str, help="The folder that contains scenes with topology maps", 
                       default="/media/cc/Expansion/scannet/processed/scans")
    
    parser.add_argument("--graph_generated_position", action="store_true", 
                       help="Use the graph generated position instead of the observed position in each frame")
    
    parser.add_argument("--data_output_dir", type=str, help="The output directory", 
                       default="/media/cc/Expansion/scannet/processed/data/squences_small_sample")
    parser.add_argument("--add_bbox_data", type=bool, help="Add the bbox data", default=True)
    parser.add_argument("--add_text_feature_data", type=bool, help="Add the text feature data", default=True)
    
    parser.add_argument("--scene_exclude_csv_path", type=str, 
                       help="The path to the scene csv file to exclude", 
                       default="/media/cc/Expansion/scannet/processed/excluded.csv")
    parser.add_argument("--save_every_n_scenes", type=int, help="Save the data every n scenes", default=20)
    parser.add_argument("--min_instance_num_frame", type=int, 
                       help="The minimum number of instances in a frame", default=2)
    
    parser.add_argument("--min_frames", type=int, 
                       help="Minimum number of frames in a sequence", default=2)
    parser.add_argument("--max_frames", type=int, 
                       help="Maximum number of frames in a sequence", default=200)
    parser.add_argument("--sequence_ratio", type=float,
                       help="Ratio of sequences to select from valid frames (0.3 means 30%%)", default=0.3)
    
    parser.add_argument("--map_level", action="store_true",
                       help="Use map-level matching: match topology map nodes from more scans with first scan")
    
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility", default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print(f"args: {args}")
    
    excluded_scene_folders = []
    if args.scene_exclude_csv_path and os.path.exists(args.scene_exclude_csv_path):
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
    data_list_sequence_matching = []
    current_scene_seq = 0
    start_scene_num = -1
    
    if args.add_text_feature_data:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        text_model = None
    
    # Process each first scan scene
    for scene_folder in tqdm(first_scans):
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
        
        # Read the topology map
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
        scene_range_all = [min(scene_x_range[0], scene_y_range[0], scene_z_range[0]), 
                          max(scene_x_range[1], scene_y_range[1], scene_z_range[1])]
        scene_range_all_with_margin = addMarginToRange(scene_range_all, 0.1)
        keypoints1_normalized = normalizePositionArray(keypoints1, scene_range_all_with_margin)
        
        # Add the bbox data if args.add_bbox_data is True
        if args.add_bbox_data:
            bbox1_height = np.array([node.shape.height for node in valid_nodes])
            bbox1_width = np.array([node.shape.width for node in valid_nodes])
            bbox1_length = np.array([node.shape.length for node in valid_nodes])
            bbox1 = np.stack([bbox1_height, bbox1_width, bbox1_length], axis=1)
            bbox1_normalized = normalizeBboxArray(bbox1, scene_range_all_with_margin)
        else:
            bbox1_normalized = None
        
        if args.add_text_feature_data:
            text_embedding1 = text_model.encode(names1)
        else:
            text_embedding1 = None
        
        feature_dim_all = descriptors1.shape[1]
        ids1 = np.array([node.id for node in valid_nodes]).astype(int)
        
        # Process more scans for this first scan
        for more_scans_scene_folder in more_scans:
            scene_id = more_scans_scene_folder.split("_")[0]
            if scene_id != scene_folder.split("_")[0]:
                continue
            
            print(f"Found more scans scene {more_scans_scene_folder} for scene {scene_folder}")
            if more_scans_scene_folder in excluded_scene_folders:
                print(f"Skipping {more_scans_scene_folder} because it is in the excluded scene folders")
                continue
            
            # Load the id correction csv file
            id_correction_csv_path = os.path.join(args.map_folder, more_scans_scene_folder, 
                                                  "matched_instance_correspondence_to_00.csv")
            if not os.path.exists(id_correction_csv_path):
                print(f"Warning: ID correction CSV not found: {id_correction_csv_path}")
                continue
            
            id_correction_dict = loadIdCorrectionCsv(id_correction_csv_path)
            
            # Generate matching data based on the mode
            if args.map_level:
                # Map-level matching: use topology map nodes from more_scans
                map_data = generateMapLevelMatchingData(
                    more_scans_scene_folder, scene_folder, args.map_folder, args,
                    keypoints1_normalized, descriptors1, bbox1_normalized, ids1,
                    scene_range_all_with_margin, id_correction_dict,
                    text_embedding1=text_embedding1, text_model=text_model, names1=names1
                )
                if map_data is not None:
                    data_list_sequence_matching.append(map_data)
            else:
                # Sequence-level matching: use sequences of frames from more_scans
                sequence_data = generateSequenceMatchingData(
                    more_scans_scene_folder, scene_folder, args.map_folder, args,
                    keypoints1_normalized, descriptors1, bbox1_normalized, ids1,
                    scene_range_all_with_margin, id_correction_dict,
                    text_embedding1=text_embedding1, text_model=text_model, names1=names1,
                    sequence_ratio=args.sequence_ratio
                )
                data_list_sequence_matching.extend(sequence_data)
        
        # Save periodically
        if current_scene_seq % args.save_every_n_scenes == 0 or current_scene_seq == len(first_scans):
            print(f"Saving data to {args.data_output_dir}/data_list_sequence_matching_{start_scene_num}_{current_scene_num}.pkl")
            print(f"Data length sequence matching: {len(data_list_sequence_matching)}")
            
            os.makedirs(args.data_output_dir, exist_ok=True)
            with open(os.path.join(args.data_output_dir, 
                                 f"data_list_sequence_matching_{start_scene_num}_{current_scene_num}.pkl"), "wb") as f:
                pickle.dump(data_list_sequence_matching, f)
            
            # Reset the data list
            data_list_sequence_matching = []
            start_scene_num = -1  # Will be reset in the next iteration

            # exit()
    
    # Save the position ranges to a json file
    meta_data_save_path = os.path.join(args.data_output_dir, "meta_data_sequence.json")
    with open(meta_data_save_path, "w") as f:
        json.dump({
            "x": [0, 1],
            "y": [0, 1],
            "z": [0, 1],
            "feature_dim": feature_dim_all
        }, f)
    print(f"Saved position ranges to {meta_data_save_path}")

