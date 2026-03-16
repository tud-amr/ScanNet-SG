import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import tempfile
import open3d as o3d

# Add paths for imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from script.include.topology_map import TopologyMap
from script.utils.result_visualization import visualize_inference_results_points, generate_instance_colors


def load_topology_map(map_path: str):
    """
    Load topology map from JSON file.
    
    Args:
        map_path: Path to the topology map JSON file
        
    Returns:
        TopologyMap object
    """
    with open(map_path, 'r') as f:
        topology_map = TopologyMap()
        topology_map.read_from_json(f.read())
    return topology_map


def load_gt_alignment_dict(gt_alignment_dict_path: str):
    """
    Load the ground truth alignment dictionary from the csv file.
    Maps instance_id in scene_bb to instance_id in scene_00.
    """
    if not os.path.exists(gt_alignment_dict_path):
        return {}
    
    with open(gt_alignment_dict_path, 'r') as f:
        gt_alignment_dict = {}
        for line in f:  # skip the first line
            if line.startswith('i'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                id, id_scene_00 = parts[0], parts[1]
                try:
                    gt_alignment_dict[int(id)] = int(id_scene_00)
                except ValueError:
                    continue
    return gt_alignment_dict


def check_ply_topology_map_mismatch(ply_path, position_dict, subscan_name=""):
    """
    Check for mismatches between instance IDs in PLY file and topology map.
    Prints warnings similar to recalculate_centers_from_ply in inference_subscan.py.
    
    Args:
        ply_path: Path to the PLY file
        position_dict: Dictionary mapping instance_id to position (from topology map)
        subscan_name: Name of subscan for logging (e.g., "subscan0" or "subscan1")
    """
    if not os.path.exists(ply_path):
        return
    
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) == 0:
            return
        
        # Extract instances from point cloud (using R channel as instance ID)
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Handle both normalized [0,1] and integer [0,255] color formats
        if colors.max() <= 1.0:
            rgb_int = (colors * 255).astype(np.uint8)
        else:
            rgb_int = colors.astype(np.uint8)
        
        # Extract instance_id from RGB (R channel for single-channel encoding, or R=G=B)
        instance_ids = rgb_int[:, 0].astype(int)
        
        # Debug: print unique instance IDs found
        unique_ids = np.unique(instance_ids)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        expected_ids = set(position_dict.keys())
        found_ids = set(unique_ids)
        matching_ids = expected_ids.intersection(found_ids)
        missing_in_ply = expected_ids - found_ids
        extra_in_ply = found_ids - expected_ids
        
        prefix = f"  [{subscan_name}]" if subscan_name else " "
        print(f"{prefix} Found {len(unique_ids)} unique instance IDs in PLY")
        print(f"{prefix} Expected {len(expected_ids)} instance IDs from topology map")
        print(f"{prefix} Matching IDs: {len(matching_ids)}")
        if missing_in_ply:
            print(f"{prefix} Warning: {len(missing_in_ply)} IDs in topology map but not in PLY: {sorted(list(missing_in_ply))[:10]}{'...' if len(missing_in_ply) > 10 else ''}")
        if extra_in_ply:
            print(f"{prefix} Warning: {len(extra_in_ply)} IDs in PLY but not in topology map: {sorted(list(extra_in_ply))[:10]}{'...' if len(extra_in_ply) > 10 else ''}")
        
        # Group points by instance ID
        instance_points = {}
        for iid in unique_ids:
            mask = instance_ids == iid
            instance_points[iid] = pts[mask]
        
        print(f"{prefix} Found {len(instance_points)} instances in PLY file")
    except Exception as e:
        print(f"{prefix} Warning: Failed to check PLY/topology map mismatch: {e}")


def prepare_gt_visualization_data(subscan0_path, subscan1_path, gt_alignment_dict):
    """
    Prepare data for ground truth visualization by loading topology maps and creating GT matches.
    
    Args:
        subscan0_path: Path to first subscan folder (contains topology_map.json)
        subscan1_path: Path to second subscan folder (contains topology_map.json)
        gt_alignment_dict: Ground truth alignment dictionary mapping instance_id in subscan1 to instance_id in subscan0
        
    Returns:
        dict: Prepared data for visualization with ground truth matches
        keypoints0, keypoints1: Original keypoints for visualization
        subscan0_positions_dict, subscan1_positions_dict: Position dictionaries
    """
    print(f"Loading topology map 0 from: {subscan0_path}")
    topology_map0_path = os.path.join(subscan0_path, "topology_map.json")
    if not os.path.exists(topology_map0_path):
        raise ValueError(f"Topology map not found: {topology_map0_path}")
    topology_map0 = load_topology_map(topology_map0_path)
    
    print(f"Loading topology map 1 from: {subscan1_path}")
    topology_map1_path = os.path.join(subscan1_path, "topology_map.json")
    if not os.path.exists(topology_map1_path):
        raise ValueError(f"Topology map not found: {topology_map1_path}")
    topology_map1 = load_topology_map(topology_map1_path)
    
    # Get valid nodes from both topology maps (excluding "unknown" objects)
    valid_nodes0 = [node for node_id, node in topology_map0.object_nodes.nodes.items() if node.name != "unknown"]
    valid_nodes1 = [node for node_id, node in topology_map1.object_nodes.nodes.items() if node.name != "unknown"]
    
    if len(valid_nodes0) == 0:
        raise ValueError("No valid nodes found in topology map 0")
    if len(valid_nodes1) == 0:
        raise ValueError("No valid nodes found in topology map 1")
    
    # Extract keypoints and IDs from topology maps
    keypoints0 = np.array([node.position for node in valid_nodes0])
    ids0 = np.array([node.id for node in valid_nodes0]).astype(int)
    
    keypoints1 = np.array([node.position for node in valid_nodes1])
    ids1 = np.array([node.id for node in valid_nodes1]).astype(int)
    
    subscan0_positions_dict = {int(node.id): node.position for node in valid_nodes0}
    subscan1_positions_dict = {int(node.id): node.position for node in valid_nodes1}
    
    print(f"Subscan 0 has {len(valid_nodes0)} valid nodes")
    print(f"Subscan 1 has {len(valid_nodes1)} valid nodes")
    
    # Check for PLY/topology map mismatches (similar to inference_subscan.py)
    print(f"\nChecking PLY/topology map consistency...")
    subscan0_ply_path = os.path.join(subscan0_path, "instance_cloud_cleaned.ply")
    if not os.path.exists(subscan0_ply_path):
        subscan0_ply_path = os.path.join(subscan0_path, "instance_cloud.ply")
    if os.path.exists(subscan0_ply_path):
        check_ply_topology_map_mismatch(subscan0_ply_path, subscan0_positions_dict, "subscan0")
    
    subscan1_ply_path = os.path.join(subscan1_path, "instance_cloud_cleaned.ply")
    if not os.path.exists(subscan1_ply_path):
        subscan1_ply_path = os.path.join(subscan1_path, "instance_cloud.ply")
    if os.path.exists(subscan1_ply_path):
        check_ply_topology_map_mismatch(subscan1_ply_path, subscan1_positions_dict, "subscan1")
    
    # Create ground truth matches from gt_alignment_dict
    # IMPORTANT: Nodes in different maps use different ID systems.
    # The CSV file (matched_instance_correspondence_to_00.csv) provides the ONLY way to establish
    # correspondences between nodes. If two nodes are NOT in the CSV, even if they have the same
    # ID number, they are NOT the same object.
    # 
    # For subscan-to-subscan matching, we match nodes in subscan1 to nodes in subscan0
    # using the ground truth alignment dictionary from the CSV file.
    gt_matches0 = np.full(len(valid_nodes0), -1, dtype=int)
    gt_match_count = 0
    
    if gt_alignment_dict is not None:
        print(f"Creating ground truth matches from alignment dictionary ({len(gt_alignment_dict)} entries)...")
        # Create a mapping from subscan1 node IDs to subscan0 node IDs
        # First, map subscan1 node IDs to scene_00 IDs using gt_alignment_dict
        # Then match to subscan0 node IDs (assuming subscan0 is from scene_00)
        for i, id1 in enumerate(ids1):
            if id1 in gt_alignment_dict:
                # Map id1 to scene_00 ID (only if explicitly in CSV)
                id_scene_00 = gt_alignment_dict[id1]
                # Find matching node in subscan0
                matches = np.where(ids0 == id_scene_00)[0]
                if len(matches) > 0:
                    # matches0[i] means: subscan0 node i matches to subscan1 node matches0[i]
                    # So we need to find which subscan0 node matches to this subscan1 node
                    gt_matches0[matches[0]] = i
                    gt_match_count += 1
        
        print(f"Created {gt_match_count} ground truth matches")
        print(f"  Subscan0 nodes with GT matches: {np.sum(gt_matches0 != -1)}")
        print(f"  Subscan0 nodes without GT matches: {np.sum(gt_matches0 == -1)}")
        # Note: Only matches where BOTH nodes exist in topology maps AND are in CSV are included
        # If a CSV entry references a node that doesn't exist, it's skipped (no correspondence created)
        # If nodes are not in CSV, even with same ID, they are NOT considered the same object
    else:
        print("Warning: No ground truth alignment dictionary provided. No matches will be visualized.")
    
    # Create result dictionary in the format expected by visualization function
    # The visualization function expects results with predicted_matches0, matching_scores0, etc.
    # We'll use ground truth matches as "predicted" matches for visualization
    result_dict = {
        "predicted_matches0": gt_matches0,  # Use GT matches as "predicted" for visualization
        "predicted_matches1": np.full(len(valid_nodes1), -1, dtype=int),  # Not used but required
        "matching_scores0": np.ones(len(valid_nodes0), dtype=np.float32),  # All GT matches have score 1.0
        "matching_scores1": np.zeros(len(valid_nodes1), dtype=np.float32),  # Not used but required
        "valid_matches": gt_match_count,
        "total_keypoints": len(valid_nodes0),
        "match_ratio": gt_match_count / len(valid_nodes0) if len(valid_nodes0) > 0 else 0.0,
        "data": {
            "keypoints0": keypoints0,  # Use original (non-normalized) keypoints for visualization
            "keypoints1": keypoints1,  # Use original (non-normalized) keypoints for visualization
            "frame_instance_ids": ids0,  # Subscan0 instance IDs (treated as frame)
            "map_node_ids": ids1,  # Subscan1 instance IDs (treated as map)
        },
        "topk_matches0": np.expand_dims(gt_matches0, axis=1) if len(gt_matches0) > 0 else np.array([]),  # Required format
        "topk_matches1": np.full((len(valid_nodes1), 1), -1, dtype=int),  # Required format
        "topk_scores0": np.ones((len(valid_nodes0), 1), dtype=np.float32),  # Required format
        "topk_scores1": np.zeros((len(valid_nodes1), 1), dtype=np.float32),  # Required format
    }
    
    return result_dict, keypoints0, keypoints1, subscan0_positions_dict, subscan1_positions_dict


def main():
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Matching for Two Subscans")
    
    # Data arguments
    parser.add_argument("--processed_data_path", type=str, default="/media/cc/Expansion/scannet/processed/scans",
                       help="Path to the processed data (for transformation files and ground truth CSV)")
    parser.add_argument("--subscan_data_path", type=str, default="/media/cc/DATA/dataset/scannet/subscans",
                       help="Path to the subscan data (contains scene folders with subscan folders)")
    parser.add_argument("--keypoints0_scene", type=str, default="",
                       help="Scene ID of the first subscan (e.g., scene0001_00). Optional, defaults to empty string.")
    parser.add_argument("--keypoints1_scene", type=str, default="scene0001_01",
                       help="Scene ID of the second subscan (e.g., scene0001_01)")
    parser.add_argument("--keypoints0_frames", type=str, default="",
                       help="Subscan folder name for first subscan (e.g., frame_12_to_50). Optional, defaults to empty string.")
    parser.add_argument("--keypoints1_frames", type=str, default="",
                       help="Subscan folder name for second subscan (e.g., frame_12_to_50)")
    
    # Visualization arguments
    parser.add_argument("--bias_meter", type=float, default=0.0,
                       help="Bias distance for visualization (default: 0.0 for subscans since both are in world coordinates)")
    parser.add_argument("--angle_bias", type=float, default=None,
                       help="Rotation angle in degrees to rotate bias direction around z-axis")
    parser.add_argument("--match_line_radius", type=float, default=0.03,
                       help="Radius of match connection lines/cylinders (default: 0.03)")
    parser.add_argument("--keypoint_radius", type=float, default=0.15,
                       help="Radius of keypoint spheres (default: 0.15)")
    parser.add_argument("--view_front", type=float, nargs=3, default=None,
                       help="Default front vector for camera view [x, y, z]")
    parser.add_argument("--view_lookat", type=float, nargs=3, default=None,
                       help="Default lookat point for camera view [x, y, z]")
    parser.add_argument("--view_up", type=float, nargs=3, default=None,
                       help="Default up vector for camera view [x, y, z]")
    parser.add_argument("--view_zoom", type=float, default=None,
                       help="Default zoom level for camera view")
    parser.add_argument("--hide_view_angles", action="store_true", default=False,
                       help="Hide view angle information in console output")
    parser.add_argument("--hide_coordinate_frame", action="store_true", default=False,
                       help="Hide the xyz coordinate frame axes in visualization")
    
    args = parser.parse_args()
    
    # Construct subscan paths (handle empty strings by filtering them out)
    subscan0_path_parts = [args.subscan_data_path]
    if args.keypoints0_scene:
        subscan0_path_parts.append(args.keypoints0_scene)
    if args.keypoints0_frames:
        subscan0_path_parts.append(args.keypoints0_frames)
    subscan0_path = os.path.join(*subscan0_path_parts)
    
    subscan1_path_parts = [args.subscan_data_path]
    if args.keypoints1_scene:
        subscan1_path_parts.append(args.keypoints1_scene)
    if args.keypoints1_frames:
        subscan1_path_parts.append(args.keypoints1_frames)
    subscan1_path = os.path.join(*subscan1_path_parts)
    
    # Check if subscan folders exist
    if not os.path.exists(subscan0_path):
        print(f"Error: Subscan 0 folder not found: {subscan0_path}")
        return
    
    if not os.path.exists(subscan1_path):
        print(f"Error: Subscan 1 folder not found: {subscan1_path}")
        return
    
    # Load ground truth alignment dictionary if scenes are different
    gt_alignment_dict = None
    if args.keypoints0_scene != args.keypoints1_scene:
        gt_alignment_dict_path = os.path.join(args.processed_data_path, args.keypoints1_scene, "matched_instance_correspondence_to_00.csv")
        if os.path.exists(gt_alignment_dict_path):
            gt_alignment_dict = load_gt_alignment_dict(gt_alignment_dict_path)
            print(f"Loaded {len(gt_alignment_dict)} ground truth alignments from {gt_alignment_dict_path}")
        else:
            print(f"Warning: Ground truth alignment file not found: {gt_alignment_dict_path}")
            print("No ground truth matches will be visualized.")
    else:
        # Same scene, create identity mapping
        print(f"Same scene ({args.keypoints0_scene if args.keypoints0_scene else '(empty)'}), using identity mapping for ground truth")
        # For same scene, we can create identity matches if both subscans have the same instance IDs
        # But we'll still need to load the subscans first to see what IDs they have
        gt_alignment_dict = {}  # Will be populated after loading subscans
    
    # Prepare visualization data with ground truth matches
    print("\n" + "="*60)
    print("Preparing Ground Truth Visualization Data")
    print("="*60)
    results, ori_keypoints0, ori_keypoints1, subscan0_positions_dict, subscan1_positions_dict = prepare_gt_visualization_data(
        subscan0_path, subscan1_path, gt_alignment_dict
    )
    
    # Handle same scene case - create identity matches for matching instance IDs
    # NOTE: Even for same scene, if CSV exists, we should use it. But if CSV is empty/not found,
    # we assume same IDs mean same objects (only for same scene subscans).
    if args.keypoints0_scene == args.keypoints1_scene and gt_alignment_dict == {}:
        print("\nSame scene detected. Creating identity matches for matching instance IDs...")
        print("  Note: Using identity matching (same ID = same object) only because CSV is empty.")
        print("  If CSV exists, it should be used even for same scene (nodes may use different ID systems).")
        ids0 = results['data']['frame_instance_ids']
        ids1 = results['data']['map_node_ids']
        
        gt_matches0 = np.full(len(ids0), -1, dtype=int)
        gt_match_count = 0
        
        for i, id0 in enumerate(ids0):
            matches = np.where(ids1 == id0)[0]
            if len(matches) > 0:
                gt_matches0[i] = matches[0]
                gt_match_count += 1
        
        results['predicted_matches0'] = gt_matches0
        results['valid_matches'] = gt_match_count
        results['match_ratio'] = gt_match_count / len(ids0) if len(ids0) > 0 else 0.0
        results['topk_matches0'] = np.expand_dims(gt_matches0, axis=1) if len(gt_matches0) > 0 else np.array([])
        
        print(f"Created {gt_match_count} identity matches for same scene")
    
    print(f"\nGround Truth Match Statistics:")
    print(f"  Total subscan0 nodes: {results['total_keypoints']}")
    print(f"  Ground truth matches: {results['valid_matches']}")
    print(f"  Match ratio: {results['match_ratio']:.4f}")
    
    # Use original keypoints directly (no filtering/recalculation)
    results['data']['keypoints0'] = ori_keypoints0
    results['data']['keypoints1'] = ori_keypoints1
    
    # Visualize ground truth matches
    print(f"\n{'='*60}")
    print(f"Visualizing Ground Truth Matches")
    print(f"{'='*60}")
    
    # Get PLY file paths
    subscan0_ply_path = os.path.join(subscan0_path, "instance_cloud_cleaned.ply")
    subscan1_ply_path = os.path.join(subscan1_path, "instance_cloud_cleaned.ply")
    
    # Check if PLY files exist, fallback to regular instance_cloud.ply
    if not os.path.exists(subscan0_ply_path):
        subscan0_ply_path = os.path.join(subscan0_path, "instance_cloud.ply")
    if not os.path.exists(subscan1_ply_path):
        subscan1_ply_path = os.path.join(subscan1_path, "instance_cloud.ply")
    
    # Load transformation matrix if scenes are different (for alignment between scenes)
    align_matrix = None
    if args.keypoints0_scene != args.keypoints1_scene:
        transformation_path = os.path.join(args.processed_data_path, args.keypoints1_scene, "transformation.npy")
        if os.path.exists(transformation_path):
            align_matrix = np.load(transformation_path)
            align_matrix = np.linalg.inv(align_matrix)
            print(f"Loaded transformation matrix from: {transformation_path}")
            print(f"Transformation matrix:\n{align_matrix}")
        else:
            print(f"Warning: Transformation matrix not found at {transformation_path}, using identity")
    else:
        print("Same scene, no transformation matrix needed")
    
    # Create temporary identity transformation file for subscan1 (required by visualization function)
    temp_pose_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    identity_matrix = np.eye(4)
    np.savetxt(temp_pose_file.name, identity_matrix)
    temp_pose_file.close()
    subscan1_pose_path = temp_pose_file.name
    
    try:
        # Generate instance colors
        instance_colors = generate_instance_colors(0, 255)
        
        # Create match_success_list - all GT matches are True (correct)
        match_success_list = []
        for i in range(len(results['data']['frame_instance_ids'])):
            if results['predicted_matches0'][i] != -1:
                match_success_list.append(True)  # All GT matches are correct
            else:
                match_success_list.append(None)  # No match (not evaluated)
        
        match_success_list_vis = [match_success_list] if match_success_list else []
        # Don't pass gt_alignment_dict - we're only showing GT matches, so no need to compare against GT
        # This prevents false negatives (orange lines) from being drawn.
        # 
        # Note: Even if we passed gt_alignment_dict, false negatives would only be drawn if:
        # - Both nodes exist in topology maps (visualization checks this)
        # - But predicted_matches0 doesn't have the match
        # Since predicted_matches0 already contains all valid GT matches (where both nodes exist),
        # there would be no false negatives anyway. But we skip the comparison entirely since
        # we're only visualizing GT matches, not comparing predictions against GT.
        gt_alignment_dict_list = None
        
        print(f"Visualization settings:")
        if align_matrix is not None:
            print(f"  align_matrix: Applied (transforms frame to align with map)")
        else:
            print(f"  align_matrix: None (no transformation - same scene or matrix not found)")
        print(f"  pose_matrix: Identity (no pose transformation for subscans)")
        print(f"  bias_meter: {args.bias_meter} (visual separation only)")
        print(f"  Ground truth matches: {results['valid_matches']} (all shown as correct)")
        print(f"  Note: gt_alignment_dict not passed to visualization (showing GT only, no false negatives)")
        
        # NOTE: Visualization function expects:
        # - map_ply_path with keypoints1 (subscan1)
        # - frame_ply_path with keypoints0 (subscan0)
        visualize_inference_results_points(
            [results], 
            subscan1_ply_path,  # Map uses keypoints1 (subscan1)
            [subscan0_ply_path],  # Frame uses keypoints0 (subscan0)
            [subscan1_pose_path],  # Use temporary identity pose file
            align_matrix=[align_matrix] if align_matrix is not None else None,  # Apply transformation if available
            instance_colors=instance_colors, 
            match_success_list=match_success_list_vis, 
            gt_alignment_dict=gt_alignment_dict_list,  # None - don't compare GT against GT
            bias_meter=[args.bias_meter],
            angle_bias=[args.angle_bias] if args.angle_bias is not None else None,
            filter_frame_outliers=False,  # No filtering
            map_background_ply_path=None,  # No background points for subscans
            match_line_radius=args.match_line_radius,  # Line thickness for match connections
            keypoint_radius=args.keypoint_radius,  # Radius of keypoint spheres
            default_front=args.view_front,
            default_lookat=args.view_lookat,
            default_up=args.view_up,
            default_zoom=args.view_zoom,
            show_view_angles=not args.hide_view_angles,
            show_coordinate_frame=not args.hide_coordinate_frame
        )
    finally:
        # Clean up temporary pose file
        if os.path.exists(subscan1_pose_path):
            os.remove(subscan1_pose_path)
    
    # Print final GT correspondences count
    print(f"\n{'='*60}")
    print(f"FINAL GROUND TRUTH CORRESPONDENCES")
    print(f"{'='*60}")
    print(f"Number of GT correspondences: {results['valid_matches']}")
    print(f"Total subscan0 nodes: {results['total_keypoints']}")
    print(f"GT match ratio: {results['match_ratio']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

