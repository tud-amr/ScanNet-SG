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


def loadIdCorrectionCsv(csv_path: str):
    """
    Load the id correction csv file. Maps instance_id to instance_id_in_00.
    """
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {int(row['instance_id']): int(row['instance_id_in_00']) for _, row in df.iterrows()}


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


def loadTopologyMap(topology_map_path):
    """
    Load topology map from JSON file.
    
    Returns:
        topology_map: TopologyMap object, or None if failed
    """
    if not os.path.exists(topology_map_path):
        return None
    
    try:
        with open(topology_map_path, "r") as f:
            topology_map = TopologyMap()
            topology_map.read_from_json(f.read())
        return topology_map
    except Exception as e:
        print(f"Warning: Could not load topology map from {topology_map_path}: {e}")
        return None


def getValidNodes(topology_map):
    """
    Get valid nodes from topology map (excluding "unknown" nodes).
    
    Returns:
        valid_nodes: List of ObjectNode objects
        node_ids: Array of node IDs
        node_positions: Array of node positions
        node_descriptors: Array of visual embeddings
        node_names: Array of node names
        node_bboxes: Array of bbox data (height, width, length)
    """
    valid_nodes = [node for node_id, node in topology_map.object_nodes.nodes.items() 
                   if node.name != "unknown"]
    
    if len(valid_nodes) == 0:
        return [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    node_ids = np.array([int(node.id) for node in valid_nodes]).astype(int)
    node_positions = np.array([node.position for node in valid_nodes])
    node_descriptors = np.array([node.visual_embedding for node in valid_nodes])
    node_names = np.array([node.name for node in valid_nodes])
    
    # Extract bbox data
    node_bboxes = []
    for node in valid_nodes:
        bbox_height = node.shape.height
        bbox_width = node.shape.width
        bbox_length = node.shape.length
        node_bboxes.append([bbox_height, bbox_width, bbox_length])
    node_bboxes = np.array(node_bboxes)
    
    return valid_nodes, node_ids, node_positions, node_descriptors, node_names, node_bboxes


def correctNodeIds(node_ids, id_correction_dict):
    """
    Correct node IDs using the correction dictionary.
    Only nodes that exist in the correction dictionary are corrected.
    Nodes without corrections are treated as unmatched.
    
    Args:
        node_ids: Array of node IDs
        id_correction_dict: Dictionary mapping original ID to corrected ID
    
    Returns:
        corrected_ids: Array of corrected node IDs (only for nodes with valid corrections)
        valid_mask: Boolean mask indicating which nodes have valid corrections
    """
    corrected_ids = []
    valid_mask = []
    for node_id in node_ids:
        node_id_int = int(node_id)
        if node_id_int in id_correction_dict:
            corrected_id = id_correction_dict[node_id_int]
            corrected_ids.append(int(corrected_id))
            valid_mask.append(True)
        else:
            # Node without correction - treat as unmatched
            valid_mask.append(False)
    return np.array(corrected_ids), np.array(valid_mask)


def calculateOverlapRatio(ids0, ids1):
    """
    Calculate overlap ratio between two sets of IDs.
    
    Args:
        ids0: Array of IDs from first subscan
        ids1: Array of IDs from second subscan
    
    Returns:
        overlap_ratio: Number of common IDs / min(len(ids0), len(ids1))
        common_ids: Set of common IDs
    """
    if len(ids0) == 0 or len(ids1) == 0:
        return 0.0, set()
    
    set0 = set(ids0)
    set1 = set(ids1)
    common_ids = set0.intersection(set1)
    
    min_size = min(len(set0), len(set1))
    if min_size == 0:
        return 0.0, common_ids
    
    overlap_ratio = len(common_ids) / min_size
    return overlap_ratio, common_ids


def createMatchingData(topology_map0, topology_map1, id_correction_dict, text_model, min_overlap_ratio=0.1):
    """
    Create matching data dictionary from two topology maps.
    
    Args:
        topology_map0: TopologyMap from scene_00 (keypoints0)
        topology_map1: TopologyMap from scene_bb (keypoints1)
        id_correction_dict: ID correction dictionary for scene_bb
        text_model: SentenceTransformer model for text embeddings
        min_overlap_ratio: Minimum overlap ratio to consider a match (default: 0.1)
    
    Returns:
        data_dict: Dictionary with matching data, or None if failed
        overlap_ratio: Overlap ratio between the two maps
    """
    # Get valid nodes from both maps
    valid_nodes0, ids0, positions0, descriptors0, names0, bboxes0 = getValidNodes(topology_map0)
    valid_nodes1, ids1, positions1, descriptors1, names1, bboxes1 = getValidNodes(topology_map1)
    
    if len(ids0) == 0 or len(ids1) == 0:
        return None, 0.0
    
    # Correct IDs for map1 - only nodes with valid corrections are included
    ids1_corrected, valid_mask = correctNodeIds(ids1, id_correction_dict)
    
    # Filter to only include nodes with valid corrections for overlap calculation
    if len(ids1_corrected) == 0:
        # No nodes have valid corrections, so no overlap
        return None, 0.0
    
    # Calculate overlap ratio using only nodes with valid corrections
    overlap_ratio, common_ids = calculateOverlapRatio(ids0, ids1_corrected)
    
    if overlap_ratio < min_overlap_ratio:  # Less than minimum overlap
        return None, overlap_ratio
    
    # Create matches0: for each node in map0, find its match in map1 (by corrected ID)
    # Only match against nodes with valid corrections
    matches0 = np.ones(len(ids0), dtype=int) * -1  # -1 means no match
    for i, id0 in enumerate(ids0):
        matches = np.where(ids1_corrected == id0)[0]
        if len(matches) > 0:
            # Map back to original indices in ids1 (accounting for valid_mask filtering)
            valid_indices = np.where(valid_mask)[0]
            matches0[i] = valid_indices[matches[0]]  # Take first match if multiple
    
    # Only keep nodes that have matches (or keep all? Let's keep all for now)
    # Filter to only include matched nodes for keypoints0, descriptors0, etc.
    # Actually, let's keep all nodes and use matches0 to indicate matches
    
    # Normalize positions
    # Combine both maps to get full range
    all_positions = np.vstack([positions0, positions1])
    scene_x_range = [min(all_positions[:, 0]), max(all_positions[:, 0])]
    scene_y_range = [min(all_positions[:, 1]), max(all_positions[:, 1])]
    scene_z_range = [min(all_positions[:, 2]), max(all_positions[:, 2])]
    scene_range_all = [min(scene_x_range[0], scene_y_range[0], scene_z_range[0]), 
                       max(scene_x_range[1], scene_y_range[1], scene_z_range[1])]
    scene_range_all_with_margin = addMarginToRange(scene_range_all, 0.1)
    
    keypoints0_normalized = normalizePositionArray(positions0, scene_range_all_with_margin)
    keypoints1_normalized = normalizePositionArray(positions1, scene_range_all_with_margin)
    
    # Normalize bboxes
    bbox0_normalized = normalizeBboxArray(bboxes0, scene_range_all_with_margin)
    bbox1_normalized = normalizeBboxArray(bboxes1, scene_range_all_with_margin)
    
    # Generate text embeddings
    text_embedding0 = text_model.encode(names0.tolist())
    text_embedding1 = text_model.encode(names1.tolist())
    
    # Create data dictionary
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
        "overlap_ratio": overlap_ratio
    }
    
    return data_dict, overlap_ratio


def findSubscanFolders(scene_folder_path):
    """
    Find all subscan folders in a scene folder.
    Subscan folders are named like "frame_42_to_708".
    
    Returns:
        subscan_folders: List of subscan folder names
    """
    if not os.path.exists(scene_folder_path):
        return []
    
    subscan_folders = []
    for item in os.listdir(scene_folder_path):
        item_path = os.path.join(scene_folder_path, item)
        if os.path.isdir(item_path) and item.startswith("frame_") and "_to_" in item:
            subscan_folders.append(item)
    
    return sorted(subscan_folders)


def processScenePair(scene_00_folder, scene_bb_folder, subscans_folder, processed_scans_folder, text_model, min_overlap_ratio=0.1):
    """
    Process a pair of scenes (scene_00 and scene_bb) and find matching subscans.
    
    Args:
        scene_00_folder: Name of scene_00 folder (e.g., "scene0000_00")
        scene_bb_folder: Name of scene_bb folder (e.g., "scene0000_01")
        subscans_folder: Path to folder containing scene folders with subscans
        processed_scans_folder: Path to folder containing matched_instance_correspondence_to_00.csv files
        text_model: SentenceTransformer model
        min_overlap_ratio: Minimum overlap ratio to consider a match (default: 0.1)
    
    Returns:
        data_list: List of matching data dictionaries
        overlap_ratios: List of overlap ratios
    """
    data_list = []
    overlap_ratios = []
    
    # Load ID correction dictionary
    id_correction_csv = os.path.join(processed_scans_folder, scene_bb_folder, 
                                     "matched_instance_correspondence_to_00.csv")
    id_correction_dict = loadIdCorrectionCsv(id_correction_csv)
    
    if len(id_correction_dict) == 0:
        print(f"  Warning: No ID correction dictionary found for {scene_bb_folder}")
        return data_list, overlap_ratios
    
    # Find subscan folders in both scenes
    scene_00_path = os.path.join(subscans_folder, scene_00_folder)
    scene_bb_path = os.path.join(subscans_folder, scene_bb_folder)
    
    subscan_folders_00 = findSubscanFolders(scene_00_path)
    subscan_folders_bb = findSubscanFolders(scene_bb_path)
    
    if len(subscan_folders_00) == 0 or len(subscan_folders_bb) == 0:
        return data_list, overlap_ratios
    
    print(f"  Found {len(subscan_folders_00)} subscans in {scene_00_folder} and {len(subscan_folders_bb)} in {scene_bb_folder}")
    
    # Compare all pairs of subscans
    for subscan_00 in subscan_folders_00:
        topology_map_00_path = os.path.join(scene_00_path, subscan_00, "topology_map.json")
        topology_map_00 = loadTopologyMap(topology_map_00_path)
        
        if topology_map_00 is None:
            print(f"    Skipping {subscan_00}: topology map not found or invalid")
            continue
        
        for subscan_bb in subscan_folders_bb:
            topology_map_bb_path = os.path.join(scene_bb_path, subscan_bb, "topology_map.json")
            topology_map_bb = loadTopologyMap(topology_map_bb_path)
            
            if topology_map_bb is None:
                print(f"    Skipping {subscan_bb}: topology map not found or invalid")
                continue
            
            # Create matching data
            data_dict, overlap_ratio = createMatchingData(
                topology_map_00, topology_map_bb, id_correction_dict, text_model, min_overlap_ratio
            )
            
            # Print overlap ratio for every check
            print(f"    Checking {subscan_00} vs {subscan_bb}: overlap_ratio = {overlap_ratio:.4f} ({'MATCH' if data_dict is not None else 'NO MATCH'})")
            
            if data_dict is not None:
                # Add scene and frame information
                data_dict["keypoints0_scene"] = scene_00_folder
                data_dict["keypoints1_scene"] = scene_bb_folder
                data_dict["keypoints0_frames"] = subscan_00
                data_dict["keypoints2_frames"] = subscan_bb
                
                data_list.append(data_dict)
                overlap_ratios.append(overlap_ratio)
    
    return data_list, overlap_ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate matcher data from subscans")
    parser.add_argument("--subscans_folder", type=str,
                       help="The folder that contains scenes with subscans",
                       default="/media/cc/DATA/dataset/scannet/subscan_sample")
    parser.add_argument("--processed_scans_folder", type=str,
                       help="The folder that contains matched_instance_correspondence_to_00.csv files",
                       default="/media/cc/Expansion/scannet/processed/scans")
    parser.add_argument("--data_output_dir", type=str,
                       help="The output directory for pkl files and CSV",
                       default="/media/cc/DATA/dataset/scannet/subscan_output_test")
    parser.add_argument("--save_every_n_scenes", type=int,
                       help="Save the data every n scenes",
                       default=20)
    parser.add_argument("--min_overlap_ratio", type=float,
                       help="Minimum overlap ratio to consider a match",
                       default=0.05)
    
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    # Initialize text model
    print("Loading text model...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get all scene folders
    scene_folders = [f for f in os.listdir(args.subscans_folder) 
                    if os.path.isdir(os.path.join(args.subscans_folder, f)) and "scene" in f]
    scene_folders.sort()
    
    print(f"Found {len(scene_folders)} scene folders")
    
    # Split into first scans (_00) and more scans (_bb where bb != 00)
    first_scans = [f for f in scene_folders if f.endswith("_00")]
    more_scans = [f for f in scene_folders if not f.endswith("_00")]
    
    print(f"Found {len(first_scans)} first scans and {len(more_scans)} more scans")
    
    # Initialize data storage
    all_data_list = []
    all_overlap_ratios = []
    all_scene_info = []  # Store scene and frame info for CSV
    
    current_scene_seq = 0
    start_scene_num = -1
    
    # Process each first scan
    for scene_00_folder in tqdm(first_scans):
        scene_id = scene_00_folder.split("_")[0]  # e.g., "scene0000"
        
        current_scene_num = int(scene_id.split("scene")[1])
        if start_scene_num == -1:
            start_scene_num = current_scene_num
        
        current_scene_seq += 1
        
        print(f"\nProcessing scene {scene_00_folder}")
        
        # Find matching more_scans scenes
        matching_more_scans = [f for f in more_scans if f.startswith(scene_id + "_")]
        
        if len(matching_more_scans) == 0:
            print(f"  No matching more_scans found for {scene_00_folder}")
            continue
        
        # Process each matching more_scans scene
        for scene_bb_folder in matching_more_scans:
            print(f"  Processing pair: {scene_00_folder} <-> {scene_bb_folder}")
            
            data_list, overlap_ratios = processScenePair(
                scene_00_folder, scene_bb_folder, args.subscans_folder, 
                args.processed_scans_folder, text_model, args.min_overlap_ratio
            )
            
            # Add to accumulated data
            all_data_list.extend(data_list)
            
            # Store overlap ratios and scene info for CSV
            for i, data_dict in enumerate(data_list):
                all_overlap_ratios.append(overlap_ratios[i])
                all_scene_info.append({
                    "keypoints0_scene": data_dict["keypoints0_scene"],
                    "keypoints1_scene": data_dict["keypoints1_scene"],
                    "keypoints0_frames": data_dict["keypoints0_frames"],
                    "keypoints2_frames": data_dict["keypoints2_frames"],
                    "overlap_ratio": overlap_ratios[i]
                })
            
            print(f"  Found {len(data_list)} matching subscan pairs")
        
        # Save periodically
        if current_scene_seq % args.save_every_n_scenes == 0 or current_scene_seq == len(first_scans):
            print(f"\nSaving data (scenes {start_scene_num} to {current_scene_num})...")
            print(f"Total data entries: {len(all_data_list)}")
            
            # Save pkl file
            pkl_filename = f"data_list_subscan_matching_{start_scene_num}_{current_scene_num}.pkl"
            pkl_path = os.path.join(args.data_output_dir, pkl_filename)
            with open(pkl_path, "wb") as f:
                pickle.dump(all_data_list, f)
            print(f"Saved data to {pkl_path}")
            
            # Print first data sample
            if len(all_data_list) > 0:
                print(f"\nFirst data sample:")
                first_sample = all_data_list[0]
                print(f"  first sample: {first_sample}")
                print(f"  keypoints0_scene: {first_sample.get('keypoints0_scene', 'N/A')}")
                print(f"  keypoints1_scene: {first_sample.get('keypoints1_scene', 'N/A')}")
                print(f"  keypoints0_frames: {first_sample.get('keypoints0_frames', 'N/A')}")
                print(f"  keypoints2_frames: {first_sample.get('keypoints2_frames', 'N/A')}")
                print(f"  overlap_ratio: {first_sample.get('overlap_ratio', 'N/A'):.4f}")
                print(f"  keypoints0 shape: {first_sample.get('keypoints0', np.array([])).shape}")
                print(f"  keypoints1 shape: {first_sample.get('keypoints1', np.array([])).shape}")
                print(f"  descriptors0 shape: {first_sample.get('descriptors0', np.array([])).shape}")
                print(f"  descriptors1 shape: {first_sample.get('descriptors1', np.array([])).shape}")
                print(f"  text_embedding0 shape: {first_sample.get('text_embedding0', np.array([])).shape}")
                print(f"  text_embedding1 shape: {first_sample.get('text_embedding1', np.array([])).shape}")
                print(f"  bbox0 shape: {first_sample.get('bbox0', np.array([])).shape}")
                print(f"  bbox1 shape: {first_sample.get('bbox1', np.array([])).shape}")
                print(f"  matches0 shape: {first_sample.get('matches0', np.array([])).shape}")
                matches_count = np.sum(first_sample.get('matches0', np.array([])) != -1)
                print(f"  number of matches: {matches_count}/{len(first_sample.get('matches0', np.array([])))}")
                print()
            
            # Save CSV file
            csv_filename = f"subscan_overlap_ratios_{start_scene_num}_{current_scene_num}.csv"
            csv_path = os.path.join(args.data_output_dir, csv_filename)
            df = pd.DataFrame(all_scene_info)
            df.to_csv(csv_path, index=False)
            print(f"Saved overlap ratios to {csv_path}")
            
            # Reset for next batch
            all_data_list = []
            all_overlap_ratios = []
            all_scene_info = []
            start_scene_num = -1
    
    print("\nDone!")

