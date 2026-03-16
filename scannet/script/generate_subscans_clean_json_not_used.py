import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Error: open3d is required for this script")
    sys.exit(1)


def count_points_per_instance(ply_path, min_points=10):
    """
    Count points per instance_id from a PLY file.
    
    Args:
        ply_path: Path to the PLY file
        min_points: Minimum number of points required (for filtering)
    
    Returns:
        instance_point_counts: Dictionary mapping instance_id (int) to point count
        valid_instance_ids: Set of instance_ids with >= min_points points
    """
    if not os.path.exists(ply_path):
        print(f"  Warning: PLY file not found: {ply_path}")
        return {}, set()
    
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) == 0:
            print(f"  Warning: PLY file is empty: {ply_path}")
            return {}, set()
        
        # Extract instance_id from RGB values
        # In PLY files from scannet_ply_map, RGB values represent instance_id
        # R channel is used as instance_id
        colors = np.asarray(pcd.colors)
        rgb_int = (colors * 255).astype(np.uint8)
        instance_ids = rgb_int[:, 0]  # Use R channel as instance_id
        
        # Count points per instance_id
        unique_ids, counts = np.unique(instance_ids, return_counts=True)
        instance_point_counts = dict(zip(unique_ids.tolist(), counts.tolist()))
        
        # Find valid instance_ids (>= min_points)
        valid_instance_ids = set(uid for uid, count in instance_point_counts.items() if count >= min_points)
        
        return instance_point_counts, valid_instance_ids
        
    except Exception as e:
        print(f"  Error: Could not read PLY file {ply_path}: {e}")
        return {}, set()


def filter_topology_map(topology_map_path, valid_instance_ids):
    """
    Filter topology map to remove nodes with instance_ids not in valid_instance_ids.
    Also removes edges connected to filtered nodes.
    
    Args:
        topology_map_path: Path to the topology_map.json file
        valid_instance_ids: Set of valid instance_ids (as integers)
    
    Returns:
        filtered_data: Filtered topology map data, or None if error
        total_nodes: Total number of nodes before filtering
        nodes_removed: Number of nodes removed
        edges_removed: Number of edges removed
    """
    if not os.path.exists(topology_map_path):
        print(f"  Warning: Topology map not found: {topology_map_path}")
        return None, 0, 0, 0
    
    try:
        with open(topology_map_path, 'r') as f:
            topology_map_data = json.load(f)
        
        # Count total nodes before filtering
        total_nodes = 0
        if 'object_nodes' in topology_map_data and 'nodes' in topology_map_data['object_nodes']:
            total_nodes = len(topology_map_data['object_nodes']['nodes'])
        
        # Convert valid_instance_ids to strings (node IDs are strings in JSON)
        valid_node_ids = set(str(instance_id) for instance_id in valid_instance_ids)
        
        # Find nodes to remove (nodes not in valid_node_ids)
        nodes_to_remove = set()
        if 'object_nodes' in topology_map_data and 'nodes' in topology_map_data['object_nodes']:
            for node_id, node_data in list(topology_map_data['object_nodes']['nodes'].items()):
                # Node ID should match instance_id
                if node_id not in valid_node_ids:
                    nodes_to_remove.add(node_id)
                    del topology_map_data['object_nodes']['nodes'][node_id]
        
        nodes_removed = len(nodes_to_remove)
        
        # Remove edges that reference removed nodes
        edges_removed = 0
        if 'edge_hypotheses' in topology_map_data:
            for hypothesis_id, hypothesis_data in topology_map_data['edge_hypotheses'].items():
                if 'edges' in hypothesis_data:
                    edges_to_remove = []
                    for edge_id, edge_data in hypothesis_data['edges'].items():
                        source_id = edge_data.get('source_id', '')
                        target_id = edge_data.get('target_id', '')
                        # Remove edge if either source or target is a removed node
                        if source_id in nodes_to_remove or target_id in nodes_to_remove:
                            edges_to_remove.append(edge_id)
                    
                    for edge_id in edges_to_remove:
                        del hypothesis_data['edges'][edge_id]
                        edges_removed += 1
        
        return topology_map_data, total_nodes, nodes_removed, edges_removed
        
    except Exception as e:
        print(f"  Error: Could not process topology map {topology_map_path}: {e}")
        return None, 0, 0, 0


def process_subscan_folder(subscan_folder, min_points=10):
    """
    Process a single subscan folder: filter topology_map.json based on point counts.
    
    Args:
        subscan_folder: Path to the subscan folder
        min_points: Minimum number of points required for a node to be kept
    
    Returns:
        success: True if processing was successful, False otherwise
    """
    instance_cloud_cleaned_path = os.path.join(subscan_folder, "instance_cloud_cleaned.ply")
    topology_map_path = os.path.join(subscan_folder, "topology_map.json")
    topology_map_cleaned_path = os.path.join(subscan_folder, "topology_map_cleaned.json")
    
    # Check if required files exist
    if not os.path.exists(instance_cloud_cleaned_path):
        print(f"  Warning: instance_cloud_cleaned.ply not found in {subscan_folder}, skipping...")
        return False
    
    if not os.path.exists(topology_map_path):
        print(f"  Warning: topology_map.json not found in {subscan_folder}, skipping...")
        return False
    
    # Check if already processed
    if os.path.exists(topology_map_cleaned_path):
        print(f"  Already processed: {subscan_folder}, skipping...")
        return True
    
    # Count points per instance
    instance_point_counts, valid_instance_ids = count_points_per_instance(
        instance_cloud_cleaned_path, min_points
    )
    
    if len(valid_instance_ids) == 0:
        print(f"  Warning: No instances with >= {min_points} points found in {subscan_folder}")
        return False
    
    # Filter topology map
    filtered_data, total_nodes, nodes_removed, edges_removed = filter_topology_map(
        topology_map_path, valid_instance_ids
    )
    
    if filtered_data is None:
        return False
    
    # Save filtered topology map
    try:
        with open(topology_map_cleaned_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"  Processed {subscan_folder}:")
        print(f"    Total instances in PLY: {len(instance_point_counts)}")
        print(f"    Valid instances (>= {min_points} points): {len(valid_instance_ids)}")
        print(f"    Total nodes in topology map: {total_nodes}")
        print(f"    Nodes removed: {nodes_removed}")
        print(f"    Nodes remaining: {total_nodes - nodes_removed}")
        print(f"    Edges removed: {edges_removed}")
        
        return True
        
    except Exception as e:
        print(f"  Error: Could not save filtered topology map: {e}")
        return False


def find_all_subscan_folders(root_dir):
    """
    Find all subscan folders in the directory structure.
    
    Expected structure:
        root_dir/
            scene0601_00/
                frame_120_to_1350/
                    instance_cloud_cleaned.ply
                    topology_map.json
                frame_XXX_to_YYY/
                    ...
            sceneXXXX_XX/
                ...
    
    Args:
        root_dir: Root directory containing scene folders (e.g., 'aaa' in 'aaa/scene0601_00/frame_120_to_1350')
    
    Returns:
        subscan_folders: List of paths to subscan folders
    """
    subscan_folders = []
    
    # Look for scene folders (e.g., scene0601_00, scene0000_00)
    if not os.path.exists(root_dir):
        print(f"Error: Root directory not found: {root_dir}")
        return subscan_folders
    
    scene_folders = [f for f in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, f)) and "scene" in f]
    
    print(f"Found {len(scene_folders)} scene folders")
    
    for scene_folder in scene_folders:
        scene_path = os.path.join(root_dir, scene_folder)
        
        # Look for subscan folders (frame_XX_to_YY)
        for item in os.listdir(scene_path):
            item_path = os.path.join(scene_path, item)
            if os.path.isdir(item_path) and item.startswith("frame_") and "_to_" in item:
                subscan_folders.append(item_path)
    
    return subscan_folders


def remove_cleaned_topology_maps(subscan_folders):
    """
    Remove all topology_map_cleaned.json files from subscan folders.
    
    Args:
        subscan_folders: List of paths to subscan folders
    
    Returns:
        removed_count: Number of files successfully removed
    """
    removed_count = 0
    
    for subscan_folder in tqdm(subscan_folders, desc="Removing cleaned topology maps"):
        topology_map_cleaned_path = os.path.join(subscan_folder, "topology_map_cleaned.json")
        
        if os.path.exists(topology_map_cleaned_path):
            try:
                os.remove(topology_map_cleaned_path)
                removed_count += 1
            except Exception as e:
                print(f"  Error: Could not remove {topology_map_cleaned_path}: {e}")
    
    return removed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter topology maps based on point counts from cleaned PLY files"
    )
    parser.add_argument(
        "--root_dir", type=str,
        help="Root directory containing scene folders with subscans (e.g., 'aaa' for structure 'aaa/scene0601_00/frame_120_to_1350')",
        required=True
    )
    parser.add_argument(
        "--min_points", type=int,
        help="Minimum number of points required for a node to be kept (default: 10)",
        default=10
    )
    parser.add_argument(
        "--scene_filter", type=str,
        help="Optional: Process only scenes matching this pattern (e.g., 'scene0000_00')",
        default=None
    )
    parser.add_argument(
        "--remove_only", action="store_true",
        help="Only remove existing topology_map_cleaned.json files without generating new ones"
    )
    args = parser.parse_args()
    
    if args.remove_only:
        # Only remove existing topology_map_cleaned.json files
        print(f"Finding all subscan folders in: {args.root_dir}")
        subscan_folders = find_all_subscan_folders(args.root_dir)
        
        if args.scene_filter:
            subscan_folders = [f for f in subscan_folders if args.scene_filter in f]
        
        print(f"Found {len(subscan_folders)} subscan folders")
        
        if len(subscan_folders) == 0:
            print("No subscan folders found. Exiting.")
            sys.exit(0)
        
        removed_count = remove_cleaned_topology_maps(subscan_folders)
        print(f"\nDone! Removed {removed_count} topology_map_cleaned.json files")
    else:
        # Normal processing: generate cleaned topology maps
        if not HAS_OPEN3D:
            print("Error: open3d is required for this script")
            sys.exit(1)
        
        print(f"Finding all subscan folders in: {args.root_dir}")
        subscan_folders = find_all_subscan_folders(args.root_dir)
        
        if args.scene_filter:
            subscan_folders = [f for f in subscan_folders if args.scene_filter in f]
        
        print(f"Found {len(subscan_folders)} subscan folders")
        
        if len(subscan_folders) == 0:
            print("No subscan folders found. Exiting.")
            sys.exit(0)
        
        # Process each subscan folder
        success_count = 0
        for subscan_folder in tqdm(subscan_folders, desc="Processing subscans"):
            if process_subscan_folder(subscan_folder, args.min_points):
                success_count += 1
        
        print(f"\nDone! Successfully processed {success_count}/{len(subscan_folders)} subscans")

