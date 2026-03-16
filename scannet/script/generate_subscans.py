import os
import sys
import argparse
import json
import numpy as np
import random
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import torch

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available, will use file size check for PLY validation")

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "..", "..", "script"))

from sequence_matcher_data_generation import parseInstanceJson, selectFrameSequences

# Import PLY post-filtering functions
try:
    sys.path.append(file_dir)
    from map_ply_post_filter import filter_point_cloud_outliers
    HAS_PLY_FILTER = True
except ImportError:
    HAS_PLY_FILTER = False
    print("Warning: map_ply_post_filter not available, will skip PLY cleaning")


def generateInstanceNameMap(json_files, refined_instance_folder, output_csv):
    """
    Generate instance_name_map.csv from a list of json files.
    
    Args:
        json_files: List of json file names (e.g., ["0_final_instance.json", "1_final_instance.json"])
        refined_instance_folder: Path to the refined_instance folder
        output_csv: Output path for the CSV file
    """
    instance_name_counts = defaultdict(list)
    
    for json_file in json_files:
        json_path = os.path.join(refined_instance_folder, json_file)
        if not os.path.exists(json_path):
            continue
        
        frame_feature_dict = parseInstanceJson(json_path)
        for instance_id, obj_data in frame_feature_dict.items():
            # Keep background (id=0) out of name mapping artifacts.
            if int(instance_id) == 0:
                continue
            object_name = obj_data.get('object_name')
            if object_name:
                instance_name_counts[instance_id].append(object_name)
    
    # Apply max pooling (i.e., most common object_name) for each instance_id
    final_mapping = {
        instance_id: Counter(names).most_common(1)[0][0]
        for instance_id, names in instance_name_counts.items()
    }
    
    # Write the result to a CSV file
    import csv
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance_id', 'name'])
        for instance_id, name in sorted(final_mapping.items()):
            writer.writerow([instance_id, name])
    
    return final_mapping


def generateBertEmbeddings(name_mapping, output_json):
    """
    Generate BERT embeddings for instance names.
    
    Args:
        name_mapping: Dictionary mapping instance_id to name
        output_json: Output path for the JSON file
    """
    # Load the DistilBERT model and tokenizer (matching get_instance_names.py)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Get the embedding for each instance name
    instance_embeddings = {}
    for instance_id, name in sorted(name_mapping.items()):
        inputs = tokenizer(name, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        instance_embeddings[instance_id] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
    
    # Save the embeddings to a json file
    with open(output_json, 'w') as f:
        json.dump(instance_embeddings, f, indent=2)
    
    return instance_embeddings


def generateAveragedFeatures(json_files, refined_instance_folder, output_json):
    """
    Generate averaged_instance_features.json from a list of json files.
    This function replicates the behavior of get_fused_object_features.cpp but 
    processes only the specified JSON files (for subscans).
    
    Args:
        json_files: List of json file names (e.g., ["0_final_instance.json", "1_final_instance.json"])
        refined_instance_folder: Path to the refined_instance folder
        output_json: Output path for the JSON file
    """
    # Structure to accumulate features (matching C++ FeatureAccumulator)
    class FeatureAccumulator:
        def __init__(self):
            self.feature_sum = None
            self.count = 0
        
        def add(self, feature_vec):
            """Add a feature vector to the accumulator"""
            if self.feature_sum is None:
                self.feature_sum = np.zeros(len(feature_vec), dtype=np.float32)
            for i in range(len(feature_vec)):
                self.feature_sum[i] += feature_vec[i]
            self.count += 1
        
        def average(self):
            """Compute the average feature vector"""
            if self.count == 0 or self.feature_sum is None:
                return None
            return (self.feature_sum / self.count).tolist()
    
    # Accumulate features for each instance_id (matching C++ behavior)
    instance_features = {}
    
    for json_file in json_files:
        json_path = os.path.join(refined_instance_folder, json_file)
        if not os.path.exists(json_path):
            continue
        
        # Read JSON file (matching C++ behavior: reads JSON array directly)
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Process each item in the JSON array (matching C++: for (const auto& item : j))
            if isinstance(json_data, list):
                for item in json_data:
                    instance_id = item.get("instance_id")
                    feature = item.get("feature")
                    if instance_id is not None and feature is not None:
                        # Keep background (id=0) out of feature artifacts.
                        if int(instance_id) == 0:
                            continue
                        if instance_id not in instance_features:
                            instance_features[instance_id] = FeatureAccumulator()
                        instance_features[instance_id].add(np.array(feature, dtype=np.float32))
        except Exception as e:
            print(f"  Warning: Could not process {json_file}: {e}")
            continue
    
    # Generate result array (matching C++ output format)
    result = []
    for instance_id, accumulator in sorted(instance_features.items()):
        avg_feature = accumulator.average()
        if avg_feature is not None:
            result.append({
                "instance_id": int(instance_id),
                "feature": avg_feature,
                "occurance": accumulator.count  # Note: matching C++ typo "occurance" not "occurrence"
            })
    
    # Save to JSON file (matching C++ output format)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def extractFrameNumber(json_file):
    """
    Extract frame number from json file name.
    Expected format: "XX_final_instance.json" or "XX.json"
    """
    frame_str = json_file.split("_")[0]
    try:
        return int(frame_str)
    except ValueError:
        return None


def generateSubscan(scene_folder, sequence, output_map_folder, exec_path, 
                   raw_images_parent_dir, refined_instance_parent_dir, 
                   edge_distance_threshold, step=3, keep_background=False):
    """
    Generate a subscan (topology map and PLY) for a sequence of frames.
    
    Args:
        scene_folder: Scene folder name (e.g., "scene0000_00")
        sequence: List of json file names for the sequence
        output_map_folder: Path to the output map folder where results will be stored
        exec_path: Path to the executable directory
        raw_images_parent_dir: Path to raw images directory
        refined_instance_parent_dir: Path to refined instance directory
        edge_distance_threshold: Edge distance threshold for topology map
        step: Step size for frame processing (default 3)
        keep_background: Keep background points (instance_id=0) in filtered PLY
    
    Returns:
        subscan_folder: Path to the created subscan folder, or None if failed
    """
    # Extract frame numbers from sequence
    frame_numbers = []
    for json_file in sequence:
        frame_num = extractFrameNumber(json_file)
        if frame_num is not None:
            frame_numbers.append(frame_num)
    
    if len(frame_numbers) == 0:
        print(f"Warning: No valid frame numbers found in sequence")
        return None
    
    frame_numbers.sort()
    start_frame = frame_numbers[0]
    end_frame = frame_numbers[-1]
    
    # Verify that frames actually exist (check for PNG files)
    refined_instance_folder = os.path.join(refined_instance_parent_dir, scene_folder, "refined_instance")
    existing_frames = []
    for frame_num in frame_numbers:
        png_path = os.path.join(refined_instance_folder, f"{frame_num}.png")
        if os.path.exists(png_path):
            existing_frames.append(frame_num)
    
    if len(existing_frames) == 0:
        print(f"  Warning: No existing frames found in sequence, skipping subscan")
        return None
    
    if len(existing_frames) < len(frame_numbers) * 0.5:  # If less than 50% of frames exist
        print(f"  Warning: Only {len(existing_frames)}/{len(frame_numbers)} frames exist, skipping subscan")
        return None
    
    # Create subscan folder: output_map_folder/scene_xxxx_xx/frame_xx_to_xx
    subscan_folder = os.path.join(output_map_folder, scene_folder, f"frame_{start_frame}_to_{end_frame}")
    os.makedirs(subscan_folder, exist_ok=True)
    
    # Check if already processed
    topology_map_path = os.path.join(subscan_folder, "topology_map.json")
    instance_cloud_path = os.path.join(subscan_folder, "instance_cloud.ply")
    if os.path.exists(topology_map_path) and os.path.exists(instance_cloud_path):
        print(f"Subscan already exists: {subscan_folder}, skipping...")
        return subscan_folder
    
    print(f"Generating subscan: {subscan_folder} (frames {start_frame} to {end_frame}, {len(existing_frames)}/{len(frame_numbers)} exist)")
    
    # Step 1: Generate PLY file
    print(f"  Step 1/5: Generating instance_cloud.ply...")
    original_cwd = os.getcwd()
    os.chdir(exec_path)
    
    # scannet_ply_map processes frames from start_frame to end_frame with step
    # Missing frames will be skipped by scannet_ply_map.
    # The PLY will contain instances from existing frames in the range.
    # The topology map will only contain instances from our sequence (via instance_name_map.csv)
    cmd = f'./scannet_ply_map {scene_folder} {start_frame} {end_frame} {step}'
    
    ret = os.system(cmd)
    os.chdir(original_cwd)
    
    # Copy the generated PLY file to subscan folder and remove background
    # scannet_ply_map saves to: refined_instance_parent_dir/scene_folder/instance_cloud_background.ply
    source_ply = os.path.join(refined_instance_parent_dir, scene_folder, "instance_cloud_background.ply")
    
    # Check if PLY file was created (even if return code is non-zero, file might still exist)
    if not os.path.exists(source_ply):
        if ret != 0:
            print(f"  Error: Failed to generate PLY file (return code: {ret})")
        else:
            print(f"  Warning: PLY file not found at {source_ply}")
        return None
    
    # If return code is non-zero but file exists, warn but continue
    if ret != 0:
        print(f"  Warning: scannet_ply_map returned non-zero code ({ret}), but PLY file exists. Continuing...")
    
    if os.path.exists(source_ply):
        # Copy PLY file to subscan folder 
        import shutil
        shutil.copy2(source_ply, instance_cloud_path)
        print(f"  Copied PLY file to {instance_cloud_path}")
    else:
        print(f"  Warning: PLY file not found at {source_ply}")
        return None
    
    # Step 2: Generate instance_name_map.csv
    print(f"  Step 2/5: Generating instance_name_map.csv...")
    instance_name_map_path = os.path.join(subscan_folder, "instance_name_map.csv")
    name_mapping = generateInstanceNameMap(sequence, refined_instance_folder, instance_name_map_path)
    
    if len(name_mapping) == 0:
        print(f"  Warning: No instances found in sequence, skipping subscan")
        return None
    
    # Filter PLY file to keep only points with instance_ids in the name map
    print(f"  Filtering PLY file to keep only instances from name map...")
    valid_instance_ids = set(name_mapping.keys())
    if keep_background:
        valid_instance_ids.add(0)
    
    if HAS_OPEN3D:
        try:
            pcd = o3d.io.read_point_cloud(instance_cloud_path)
            if len(pcd.points) == 0:
                print(f"  Warning: PLY file is empty, skipping filtering")
            else:
                # In PLY files from scannet_ply_map, RGB values represent instance_id
                # For instance_id < 255, R=G=B=instance_id
                colors = np.asarray(pcd.colors)
                rgb_int = (colors * 255).astype(np.uint8)
                
                # Extract instance_id from RGB (assuming R channel is instance_id for single-channel encoding)
                # For multi-channel encoding, instance_id might be encoded differently
                # Check if R=G=B (single-channel encoding) or use R channel as instance_id
                instance_ids = rgb_int[:, 0]  # Use R channel as instance_id
                
                # Create mask for valid instance IDs
                valid_mask = np.isin(instance_ids, list(valid_instance_ids))
                
                if np.sum(valid_mask) == 0:
                    print(f"  Warning: No points found with valid instance IDs, skipping subscan")
                    return None
                
                # Filter points
                pcd_filtered = pcd.select_by_index(np.where(valid_mask)[0])
                
                print(f"  PLY filtering: {len(pcd.points)} points total, {len(pcd_filtered.points)} points after filtering")
                
                # Save filtered PLY file in ASCII format compatible with PCL
                points = np.asarray(pcd_filtered.points)
                colors_filtered = np.asarray(pcd_filtered.colors)
                rgb_int_filtered = (colors_filtered * 255).astype(np.uint8)
                
                with open(instance_cloud_path, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(points)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")
                    
                    for i in range(len(points)):
                        f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {rgb_int_filtered[i][0]} {rgb_int_filtered[i][1]} {rgb_int_filtered[i][2]}\n")
                
                print(f"  Saved filtered PLY file to {instance_cloud_path}")
        except Exception as e:
            print(f"  Warning: Could not filter PLY file: {e}")
            print(f"  Continuing with unfiltered PLY file...")
    else:
        print(f"  Warning: open3d not available, cannot filter PLY file")
    
    # Step 3: Generate averaged_instance_features.json
    # Using Python implementation that replicates get_fused_object_features.cpp behavior
    # but processes only the sequence frames (not all files in directory)
    print(f"  Step 3/5: Generating averaged_instance_features.json...")
    averaged_features_path = os.path.join(subscan_folder, "averaged_instance_features.json")
    generateAveragedFeatures(sequence, refined_instance_folder, averaged_features_path)
    
    # Step 4: Generate instance_bert_embeddings.json
    print(f"  Step 4/5: Generating instance_bert_embeddings.json...")
    bert_embeddings_path = os.path.join(subscan_folder, "instance_bert_embeddings.json")
    generateBertEmbeddings(name_mapping, bert_embeddings_path)
    
    # Step 5: Generate topology_map.json
    print(f"  Step 5/5: Generating topology_map.json...")
    original_cwd = os.getcwd()
    os.chdir(exec_path)
    # Keep background only in saved PLY artifacts. For topology JSON generation,
    # strip id=0 points when requested.
    ply_for_json = instance_cloud_path
    temp_ply_no_background = None
    if keep_background and HAS_OPEN3D:
        try:
            pcd_for_json = o3d.io.read_point_cloud(instance_cloud_path)
            if len(pcd_for_json.points) > 0:
                colors_for_json = np.asarray(pcd_for_json.colors)
                rgb_int_for_json = (colors_for_json * 255).astype(np.uint8)
                instance_ids_for_json = rgb_int_for_json[:, 0]
                non_background_mask = instance_ids_for_json != 0
                if np.sum(non_background_mask) > 0:
                    pcd_non_background = pcd_for_json.select_by_index(np.where(non_background_mask)[0])
                    temp_ply_no_background = os.path.join(subscan_folder, "instance_cloud_no_background_tmp.ply")
                    points = np.asarray(pcd_non_background.points)
                    colors_filtered = np.asarray(pcd_non_background.colors)
                    rgb_int_filtered = (colors_filtered * 255).astype(np.uint8)
                    with open(temp_ply_no_background, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(points)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write("end_header\n")
                        for i in range(len(points)):
                            f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {rgb_int_filtered[i][0]} {rgb_int_filtered[i][1]} {rgb_int_filtered[i][2]}\n")
                    ply_for_json = temp_ply_no_background
        except Exception as e:
            print(f"  Warning: Could not create non-background PLY for JSON generation: {e}")
            print(f"  Continuing with original PLY for JSON generation...")
    elif keep_background and not HAS_OPEN3D:
        print(f"  Warning: open3d not available; cannot strip background for JSON generation.")
        print(f"  Continuing with original PLY for JSON generation...")

    ply_file_with_quotes = '"' + ply_for_json + '"'
    cmd = f'./generate_json {ply_file_with_quotes} 0 0 {edge_distance_threshold}'
    ret = os.system(cmd)
    os.chdir(original_cwd)
    
    if ret != 0:
        if temp_ply_no_background is not None and os.path.exists(temp_ply_no_background):
            try:
                os.remove(temp_ply_no_background)
            except Exception as e:
                print(f"  Warning: Could not remove temp file {temp_ply_no_background}: {e}")
        print(f"  Error: Failed to generate topology map")
        return None
    
    # Check if topology map was generated
    if not os.path.exists(topology_map_path):
        if temp_ply_no_background is not None and os.path.exists(temp_ply_no_background):
            try:
                os.remove(temp_ply_no_background)
            except Exception as e:
                print(f"  Warning: Could not remove temp file {temp_ply_no_background}: {e}")
        print(f"  Warning: Topology map not found at {topology_map_path}")
        return None

    # Remove temporary non-background PLY used only for JSON generation.
    if temp_ply_no_background is not None and os.path.exists(temp_ply_no_background):
        try:
            os.remove(temp_ply_no_background)
        except Exception as e:
            print(f"  Warning: Could not remove temp file {temp_ply_no_background}: {e}")
    
    # Step 6: Remove "unknown" nodes and their edges from topology map
    print(f"  Step 6/6: Removing 'unknown' nodes and their edges from topology map...")
    try:
        with open(topology_map_path, 'r') as f:
            topology_map_data = json.load(f)
        
        # Find all "unknown" node IDs
        unknown_node_ids = set()
        if 'object_nodes' in topology_map_data and 'nodes' in topology_map_data['object_nodes']:
            for node_id, node_data in list(topology_map_data['object_nodes']['nodes'].items()):
                if node_data.get('name') == 'unknown':
                    unknown_node_ids.add(node_id)
                    del topology_map_data['object_nodes']['nodes'][node_id]
        
        print(f"  Removed {len(unknown_node_ids)} 'unknown' nodes: {unknown_node_ids}")
        
        # Remove edges that reference unknown nodes
        edges_removed = 0
        if 'edge_hypotheses' in topology_map_data:
            for hypothesis_id, hypothesis_data in topology_map_data['edge_hypotheses'].items():
                if 'edges' in hypothesis_data:
                    edges_to_remove = []
                    for edge_id, edge_data in hypothesis_data['edges'].items():
                        source_id = edge_data.get('source_id', '')
                        target_id = edge_data.get('target_id', '')
                        # Remove edge if either source or target is an unknown node
                        if source_id in unknown_node_ids or target_id in unknown_node_ids:
                            edges_to_remove.append(edge_id)
                    
                    for edge_id in edges_to_remove:
                        del hypothesis_data['edges'][edge_id]
                        edges_removed += 1
        
        print(f"  Removed {edges_removed} edges connected to 'unknown' nodes")
        
        # Save the filtered topology map
        with open(topology_map_path, 'w') as f:
            json.dump(topology_map_data, f, indent=2)
        
        print(f"  Saved filtered topology map to {topology_map_path}")
    except Exception as e:
        print(f"  Warning: Could not filter topology map: {e}")
        print(f"  Continuing with unfiltered topology map...")
    
    # Step 7: Generate instance_cloud_cleaned.ply using map_ply_post_filter
    print(f"  Step 7/7: Generating instance_cloud_cleaned.ply...")
    instance_cloud_cleaned_path = os.path.join(subscan_folder, "instance_cloud_cleaned.ply")
    
    if HAS_PLY_FILTER and HAS_OPEN3D:
        try:
            # Load the PLY file
            pcd = o3d.io.read_point_cloud(instance_cloud_path)
            if len(pcd.points) == 0:
                print(f"  Warning: PLY file is empty, skipping cleaning")
            else:
                pts = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                
                # Convert to 0–255 for discrete instance colors
                rgb_int = (colors * 255).astype(np.uint8)
                
                # Get unique colors (instances)
                unique_colors = np.unique(rgb_int, axis=0)
                
                merged_points = []
                merged_colors = []
                
                print(f"  Processing {len(unique_colors)} unique instances...")
                for uc in unique_colors:
                    mask = np.all(rgb_int == uc, axis=1)
                    pts_c = pts[mask]
                    
                    if len(pts_c) == 0:
                        continue

                    # Keep background points (id=0) untouched when requested.
                    # This avoids passing them through outlier filtering.
                    if keep_background and np.all(uc == 0):
                        merged_points.append(pts_c)
                        merged_colors.append(np.tile(uc / 255.0, (pts_c.shape[0], 1)))
                        continue
                    
                    # Filter these points (outlier removal + DBSCAN clustering)
                    filtered = filter_point_cloud_outliers(pts_c)
                    
                    if len(filtered) == 0:
                        continue
                    
                    # Add filtered points back with the same color
                    merged_points.append(filtered)
                    merged_colors.append(np.tile(uc / 255.0, (filtered.shape[0], 1)))
                
                # Merge all points
                if len(merged_points) == 0:
                    print(f"  Warning: No points after filtering, skipping cleaned PLY generation")
                else:
                    merged_points = np.concatenate(merged_points, axis=0)
                    merged_colors = np.concatenate(merged_colors, axis=0)
                    
                    # Create and save cleaned point cloud
                    out_pcd = o3d.geometry.PointCloud()
                    out_pcd.points = o3d.utility.Vector3dVector(merged_points)
                    out_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
                    
                    # Save in ASCII format compatible with PCL
                    points = np.asarray(out_pcd.points)
                    colors_filtered = np.asarray(out_pcd.colors)
                    rgb_int_filtered = (colors_filtered * 255).astype(np.uint8)
                    
                    with open(instance_cloud_cleaned_path, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(points)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write("end_header\n")
                        
                        for i in range(len(points)):
                            f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {rgb_int_filtered[i][0]} {rgb_int_filtered[i][1]} {rgb_int_filtered[i][2]}\n")
                    
                    print(f"  Saved cleaned PLY file: {len(pcd.points)} points → {len(points)} points")
        except Exception as e:
            print(f"  Warning: Could not generate cleaned PLY file: {e}")
            print(f"  Continuing without cleaned PLY file...")
    else:
        if not HAS_PLY_FILTER:
            print(f"  Warning: map_ply_post_filter not available, skipping PLY cleaning")
        if not HAS_OPEN3D:
            print(f"  Warning: open3d not available, skipping PLY cleaning")
    
    print(f"  Successfully generated subscan: {subscan_folder}")
    return subscan_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subscans with topology maps from frame sequences")
    parser.add_argument("--map_folder", type=str, 
                       help="The folder that contains scenes with topology maps (input)",
                       default="/media/cc/Expansion/scannet/processed/scans")
    parser.add_argument("--output_map_folder", type=str,
                       help="The folder where subscan results will be stored (output)",
                       default=None)
    parser.add_argument("--exec_path", type=str,
                       help="Path to the executable directory",
                       default="/home/cc/chg_ws/ros_ws/topomap_ws/devel/lib/semantic_topo_map")
    parser.add_argument("--raw_images_parent_dir", type=str,
                       help="Path to raw images directory",
                       default="/media/cc/Extreme SSD/dataset/scannet/images/scans/")
    parser.add_argument("--refined_instance_parent_dir", type=str,
                       help="Path to refined instance directory (defaults to map_folder if not specified). "
                            "Use this if refined_instance folders are in a different location than map_folder.",
                       default="/media/cc/Expansion/scannet/processed/scans/")
    parser.add_argument("--min_frames", type=int,
                       help="Minimum number of frames in a sequence",
                       default=50)
    parser.add_argument("--max_frames", type=int,
                       help="Maximum number of frames in a sequence",
                       default=300)
    parser.add_argument("--sequence_ratio", type=float,
                       help="Ratio of sequences to select compared to the number of valid frames (0.3 means 30%%)",
                       default=0.01)
    parser.add_argument("--min_sequences", type=int,
                       help="Minimum number of sequences to generate per scene (overrides sequence_ratio if calculated value is lower)",
                       default=1)
    parser.add_argument("--max_sequences", type=int,
                       help="Maximum number of sequences to generate per scene (overrides sequence_ratio if calculated value is higher)",
                       default=10)
    parser.add_argument("--edge_distance_threshold", type=float,
                       help="Edge distance threshold for topology map generation",
                       default=2.0)
    parser.add_argument("--step", type=int,
                       help="Step size for frame processing",
                       default=3)
    parser.add_argument("--keep_background", action="store_true",
                       help="Keep background points (instance_id=0) in filtered subscan PLY")
    parser.add_argument("--min_instance_num_frame", type=int,
                       help="Minimum number of instances in a frame",
                       default=2)
    parser.add_argument("--random_seed", type=int,
                       help="Random seed for reproducibility",
                       default=42)
    parser.add_argument("--scene_exclude_csv_path", type=str,
                       help="Path to CSV file with scenes to exclude",
                       default="/media/cc/Expansion/scannet/processed/excluded.csv")
    parser.add_argument("--start_scene_seq", type=int,
                       help="Start scene sequence",
                       default=None)
    parser.add_argument("--end_scene_seq", type=int,
                       help="End scene sequence",
                       default=None)
    args = parser.parse_args()

    start_scene_seq = 0
    end_scene_seq = 800
    if args.start_scene_seq is not None:
        start_scene_seq = args.start_scene_seq
    if args.end_scene_seq is not None:
        end_scene_seq = args.end_scene_seq
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print(f"Arguments: {args}")
    
    # Set output map folder (default to map_folder if not specified)
    output_map_folder = args.output_map_folder if args.output_map_folder else args.map_folder
    os.makedirs(output_map_folder, exist_ok=True)
    print(f"Results will be stored in: {output_map_folder}")
    
    # refined_instance is typically in map_folder/scenexxxx_xx/refined_instance
    # But can be overridden with --refined_instance_parent_dir if in a different location
    refined_instance_parent_dir = args.refined_instance_parent_dir if args.refined_instance_parent_dir else args.map_folder
    print(f"Using refined_instance_parent_dir: {refined_instance_parent_dir} (fallback: {args.map_folder if args.refined_instance_parent_dir else 'N/A - using map_folder'})")
    
    # Load excluded scenes if provided
    excluded_scene_folders = []
    if args.scene_exclude_csv_path and os.path.exists(args.scene_exclude_csv_path):
        import pandas as pd
        df = pd.read_csv(args.scene_exclude_csv_path)
        excluded_scene_folders = df.iloc[:, 0].tolist() if len(df.columns) > 0 else []
        print(f"Found {len(excluded_scene_folders)} excluded scene folders")
    
    # Get all scene folders
    scene_folders = [f for f in os.listdir(args.map_folder) 
                     if os.path.isdir(os.path.join(args.map_folder, f)) and "scene" in f]
    print(f"Found {len(scene_folders)} scene folders")
    
    # Process each scene folder
    for scene_folder in tqdm(scene_folders):
        if scene_folder in excluded_scene_folders:
            print(f"Skipping excluded scene: {scene_folder}")
            continue

        scene_seq = scene_folder.split("/")[-1].split("_")[0].split("scene")[-1]
        scene_seq = int(scene_seq)
        if scene_seq < start_scene_seq or scene_seq > end_scene_seq:
            print(f"Skipping scene: {scene_folder} because it is not in the range of {start_scene_seq} to {end_scene_seq}")
            continue
        
        print(f"\nProcessing scene: {scene_folder}")
        
        # Get refined instance folder
        refined_instance_folder = os.path.join(refined_instance_parent_dir, scene_folder, "refined_instance")
        if not os.path.exists(refined_instance_folder):
            print(f"  Warning: Refined instance folder not found: {refined_instance_folder}")
            continue
        
        # Get all json files
        json_files = [f for f in os.listdir(refined_instance_folder) 
                     if f.endswith("final_instance.json")]
        
        # Filter valid frames (with at least min_instance_num_frame instances)
        valid_json_files = []
        for json_file in json_files:
            json_path = os.path.join(refined_instance_folder, json_file)
            frame_feature_dict = parseInstanceJson(json_path)
            if len(frame_feature_dict) >= args.min_instance_num_frame:
                valid_json_files.append(json_file)
        
        if len(valid_json_files) == 0:
            print(f"  Warning: No valid frames found for scene {scene_folder}")
            continue
        
        # Calculate number of sequences to select
        num_sequences = int(args.sequence_ratio * len(valid_json_files))
        
        # Apply minimum and maximum bounds
        num_sequences = max(args.min_sequences, num_sequences)
        if args.max_sequences is not None:
            num_sequences = min(args.max_sequences, num_sequences)
        
        # Ensure at least 1 sequence if there are valid frames
        num_sequences = max(1, num_sequences)
        
        # Select frame sequences
        sequences = selectFrameSequences(valid_json_files, num_sequences, 
                                       min_frames=args.min_frames, 
                                       max_frames=args.max_frames)
        
        print(f"  Selected {len(sequences)} sequences from {len(valid_json_files)} valid frames")
        
        # Generate subscan for each sequence
        for seq_idx, sequence in enumerate(sequences):
            subscan_folder = generateSubscan(
                scene_folder, sequence, output_map_folder, args.exec_path,
                args.raw_images_parent_dir, refined_instance_parent_dir,
                args.edge_distance_threshold, args.step, args.keep_background
            )
            
            if subscan_folder is None:
                print(f"  Failed to generate subscan for sequence {seq_idx}")
            else:
                print(f"  Successfully generated subscan {seq_idx+1}/{len(sequences)}: {subscan_folder}")
    
    print("\nDone!")

