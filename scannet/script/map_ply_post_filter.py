import os
import sys
import numpy as np
import open3d as o3d
import shutil
import csv
import json
import re

# Add paths for imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.join(root_dir, "script")
scannet_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, "include"))
sys.path.append(scannet_script_dir)

from topology_map import TopologyMap

# Import functions from align_instances.py
from align_instances import (
    extract_instances,
    compute_overlap_score,
    load_instance_labels,
    load_instance_bert_embeddings,
    compute_all_bert_similarities
)


###############################################################
# Filter point cloud to remove outliers and keep main component
###############################################################
def filter_point_cloud_outliers(
    pts,
    nb_neighbors=20,
    std_ratio=2.0,
    eps=0.05,
    min_points=10
):
    """
    Filter point cloud to remove outliers and keep the main connected component.
    
    Args:
        pts: numpy array of shape (N, 3) containing point coordinates
    """
    if len(pts) < min_points:
        return pts
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    # Step 1: Statistical Outlier Removal (SOR)
    pcd_filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    if len(pcd_filtered.points) < min_points:
        return np.asarray(pcd_filtered.points)
    
    # Step 2: DBSCAN clustering to find connected components
    labels = np.array(pcd_filtered.cluster_dbscan(
        eps=eps,
        min_points=min_points
    ))
    
    if len(labels) == 0 or np.all(labels == -1):
        return np.asarray(pcd_filtered.points)
    
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return np.asarray(pcd_filtered.points)
    
    largest_cluster_label = unique_labels[np.argmax(counts)]
    mask = labels == largest_cluster_label
    
    return np.asarray(pcd_filtered.points)[mask]


###############################################################
# Instance matching functions (using align_instances.py functions)
###############################################################
def extract_instances_from_ply(ply_path, three_channel_id=False):
    """
    Extract instances point clouds from a PLY file.
    Uses extract_instances from align_instances.py
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    instances = extract_instances(pcd, three_channel_id=three_channel_id)
    # Remove background (id=0) if present
    if 0 in instances:
        del instances[0]
    return instances


def match_instances_with_names(instances_B, instances_A, names_B, names_A, keypoints_B, keypoints_A, 
                               target_instance_ids, source_instance_ids, dist_threshold=0.15, min_overlap_ratio=0.1,
                               keypoint_distance_threshold=1.0):
    """
    Match instances based on their names and overlap score.
    Adapted from align_instances.py to accept instance_ids as parameters.
    
    Args:
        keypoint_distance_threshold: Maximum distance between keypoints after transformation (meters)
    """
    match_dict = {}
    for i in range(len(keypoints_B)):
        id_B = target_instance_ids[i]
        if id_B not in instances_B:
            continue
        pts_B = instances_B[id_B]
        name_B = names_B.get(id_B)
        if name_B is None:
            continue
            
        # Filter A instances to same object class
        candidates_A = {id_A: pts_A for id_A, pts_A in instances_A.items() if names_A.get(id_A) == name_B}
        if not candidates_A:
            continue
        
        best_match_ratio = 0
        best_match = None
        
        # Only iterate through candidates with the same name
        for id_A, pts_A in candidates_A.items():
            # Find the index of this candidate in keypoints_A
            try:
                j = source_instance_ids.index(id_A)
                center_A = keypoints_A[j]
                center_B = keypoints_B[i]
            except ValueError:
                continue

            dist = np.linalg.norm(center_A - center_B)
            if dist > keypoint_distance_threshold:  # Check distance threshold after transformation
                continue

            match_ratio, miou, _ = compute_overlap_score(pts_B, pts_A, dist_threshold)
            # Debug: log distance for accepted matches
            if match_ratio > best_match_ratio and (match_ratio > min(0.8, min_overlap_ratio*2) or miou > min_overlap_ratio):
                if dist > keypoint_distance_threshold:
                    print(f"  WARNING: Match {id_B} -> {id_A} has distance {dist:.3f}m > threshold {keypoint_distance_threshold:.3f}m but was accepted!")
            if match_ratio > best_match_ratio and (match_ratio > min(0.8, min_overlap_ratio*2) or miou > min_overlap_ratio):
                best_match_ratio = match_ratio
                best_match = id_A
            elif dist < 0.2 and match_ratio > min_overlap_ratio:
                # If centers are very close (< 0.2m), still require minimum overlap to avoid false matches
                if match_ratio > best_match_ratio:
                    best_match_ratio = match_ratio
                    best_match = id_A

        if best_match is not None:
            match_dict[id_B] = best_match

    return match_dict


def match_instances_with_bert_embeddings(instances_B, instances_A, bert_embeddings_B, bert_embeddings_A, 
                                         keypoints_B, keypoints_A, target_instance_ids, source_instance_ids,
                                         dist_threshold=0.2, bert_name_similarity_threshold=0.4, min_overlap_ratio=0.1,
                                         keypoint_distance_threshold=1.0):
    """
    Match instances based on their names and overlap score using BERT embeddings.
    Adapted from align_instances.py to accept instance_ids as parameters.
    
    Args:
        keypoint_distance_threshold: Maximum distance between keypoints after transformation (meters)
    """
    # Precompute all BERT similarities
    print("Precomputing BERT similarities...")
    bert_similarities = compute_all_bert_similarities(bert_embeddings_A, bert_embeddings_B)
    print(f"Computed {len(bert_similarities)} BERT similarities")

    match_dict = {}
    for i in range(len(keypoints_B)):
        id_B = target_instance_ids[i]
        if id_B not in instances_B:
            continue
        pts_B = instances_B[id_B]
        best_score = 0
        best_match = None
        
        for j in range(len(keypoints_A)):
            id_A = source_instance_ids[j]
            if id_A not in instances_A:
                continue
            pts_A = instances_A[id_A]

            center_A = keypoints_A[j]
            center_B = keypoints_B[i]
            dist = np.linalg.norm(center_A - center_B)
            if dist > keypoint_distance_threshold:  # Check distance threshold after transformation
                continue

            bert_sim = bert_similarities.get((str(id_A), str(id_B)), 0.0)
            # Debug: log distance for accepted matches
            if dist > keypoint_distance_threshold:
                print(f"  WARNING: Match candidate {id_B} -> {id_A} has distance {dist:.3f}m > threshold {keypoint_distance_threshold:.3f}m but passed distance check!")

            if bert_sim < bert_name_similarity_threshold:
                continue
            
            match_ratio, miou, avg_dist = compute_overlap_score(pts_B, pts_A, dist_threshold)
            dist_score = 0
            if avg_dist > 1e-3:
                # Map the avg_dist to a score between 0 and 1. Dist = 0 should have the highest score
                dist_score = np.exp(-2*avg_dist)

            score = match_ratio * 0.3 + bert_sim * 0.1 + miou * 0.3 + dist_score * 0.3
                
            if score > best_score and (match_ratio > min(0.8, min_overlap_ratio*2) or miou > min_overlap_ratio):
                best_score = score
                best_match = id_A
                
            # Early termination: if we found a good match, stop searching
            if best_score > 0.9:  # High confidence threshold
                break
                
        if best_match is not None:
            match_dict[id_B] = best_match

    return match_dict


def visualize_correspondence_vectors(subfolder, scene_00_path, keypoints_target, keypoints_source, 
                                     target_instance_ids, source_instance_ids, match_dict):
    """
    Visualize correspondence vectors and both maps.
    
    Args:
        subfolder: Path to scene_0x folder
        scene_00_path: Path to scene_00 folder
        keypoints_target: List of scene_0x keypoints
        keypoints_source: List of transformed scene_00 keypoints
        target_instance_ids: List of scene_0x instance IDs
        source_instance_ids: List of scene_00 instance IDs
        match_dict: Dictionary mapping target_id -> source_id
    """
    print(f"  Creating visualization...")
    print(f"    Subfolder: {subfolder}")
    print(f"    Scene_00 path: {scene_00_path}")
    print(f"    Matches: {len(match_dict)}")
    
    # Load PLY files
    ply_target_path = os.path.join(subfolder, "instance_cloud_cleaned.ply")
    if not os.path.exists(ply_target_path):
        ply_target_path = os.path.join(subfolder, "instance_cloud.ply")
    
    ply_source_path = os.path.join(scene_00_path, "instance_cloud_cleaned.ply")
    if not os.path.exists(ply_source_path):
        ply_source_path = os.path.join(scene_00_path, "instance_cloud.ply")
    
    print(f"    Target PLY: {ply_target_path} (exists: {os.path.exists(ply_target_path)})")
    print(f"    Source PLY: {ply_source_path} (exists: {os.path.exists(ply_source_path)})")
    
    if not os.path.exists(ply_target_path) or not os.path.exists(ply_source_path):
        print(f"  Error: PLY files not found, skipping visualization")
        print(f"    Target exists: {os.path.exists(ply_target_path)}")
        print(f"    Source exists: {os.path.exists(ply_source_path)}")
        return
    
    # Load point clouds
    pcd_target = o3d.io.read_point_cloud(ply_target_path)
    pcd_source = o3d.io.read_point_cloud(ply_source_path)
    
    # Color point clouds differently
    pcd_target.paint_uniform_color([1.0, 0.0, 0.0])  # Red for scene_0x
    pcd_source.paint_uniform_color([0.0, 1.0, 0.0])  # Green for scene_00
    
    # Create visualization geometries
    geometries = [pcd_target, pcd_source]
    
    # Add keypoint spheres
    keypoint_spheres = []
    for kpt in keypoints_target:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(kpt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        keypoint_spheres.append(sphere)
    
    for kpt in keypoints_source:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(kpt)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        keypoint_spheres.append(sphere)
    
    geometries.extend(keypoint_spheres)
    
    # Add correspondence vectors as lines
    target_instance_ids2seq_dict = {id_: i for i, id_ in enumerate(target_instance_ids)}
    source_instance_ids2seq_dict = {id_: i for i, id_ in enumerate(source_instance_ids)}
    
    line_set_points = []
    line_set_lines = []
    line_colors = []
    line_idx = 0
    
    # Calculate distances for color coding
    distances = []
    for id_B, id_A in match_dict.items():
        if id_B not in target_instance_ids2seq_dict or id_A not in source_instance_ids2seq_dict:
            continue
        kpt_B = keypoints_target[target_instance_ids2seq_dict[id_B]]
        kpt_A = keypoints_source[source_instance_ids2seq_dict[id_A]]
        dist = np.linalg.norm(kpt_A - kpt_B)
        distances.append(dist)
    
    max_dist = max(distances) if distances else 1.0
    min_dist = min(distances) if distances else 0.0
    
    for id_B, id_A in match_dict.items():
        if id_B not in target_instance_ids2seq_dict or id_A not in source_instance_ids2seq_dict:
            continue
        kpt_B = keypoints_target[target_instance_ids2seq_dict[id_B]]
        kpt_A = keypoints_source[source_instance_ids2seq_dict[id_A]]
        dist = np.linalg.norm(kpt_A - kpt_B)
        
        line_set_points.append(kpt_A)
        line_set_points.append(kpt_B)
        line_set_lines.append([line_idx * 2, line_idx * 2 + 1])
        
        # Color code by distance: green (close) to red (far)
        # Normalize distance to [0, 1] range
        if max_dist > min_dist:
            normalized_dist = (dist - min_dist) / (max_dist - min_dist)
        else:
            normalized_dist = 0.0
        
        # Green (close) -> Yellow -> Red (far)
        if normalized_dist < 0.5:
            # Green to Yellow
            r = normalized_dist * 2.0
            g = 1.0
            b = 0.0
        else:
            # Yellow to Red
            r = 1.0
            g = 1.0 - (normalized_dist - 0.5) * 2.0
            b = 0.0
        
        line_colors.append([r, g, b])
        line_idx += 1
    
    if len(line_set_points) > 0:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(line_set_points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(line_set_lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
        geometries.append(line_set)
        print(f"    Added {len(line_set_lines)} correspondence lines (color: Green=close, Red=far)")
        if distances:
            print(f"    Distance range: {min_dist:.3f}m - {max_dist:.3f}m")
    
    # Visualize
    print(f"  Showing visualization:")
    print(f"    Red point cloud/spheres = scene_0x")
    print(f"    Green point cloud/spheres = scene_00 (transformed)")
    print(f"    Colored lines = correspondences (Green=close distance, Yellow=medium, Red=far)")
    print(f"    Total geometries: {len(geometries)}")
    print(f"    Close the visualization window to continue...")
    try:
        # Use draw_geometries which blocks until window is closed
        o3d.visualization.draw_geometries(
            geometries, 
            window_name=f"Correspondence Visualization: {os.path.basename(subfolder)}",
            width=1200,
            height=800,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=True
        )
        print(f"  Visualization window closed, continuing...")
    except Exception as e:
        print(f"  Error: Failed to show visualization: {e}")
        import traceback
        traceback.print_exc()


def update_instance_correspondence_csv(subfolder, root_dir, openset=False, 
                                       visualize=False,
                                       keypoint_distance_threshold=1.0):
    """
    Update matched_instance_correspondence_to_00.csv for sceneXXXX_0x folders.
    
    Args:
        subfolder: Path to the subfolder (e.g., scene0001_01)
        root_dir: Root directory containing all scene folders
        openset: Whether this is an openset scene (if True, uses BERT embeddings and three_channel_id)
        visualize: Whether to show visualization
        keypoint_distance_threshold: Maximum distance between matched keypoints after transformation (meters, default: 1.0)
    """
    print(f"  update_instance_correspondence_csv called with visualize={visualize}")
    # Openset automatically uses BERT embeddings and three_channel_id
    use_bert = openset
    three_channel_id = openset
    subfolder_name = os.path.basename(subfolder)
    
    # Check if subfolder matches pattern sceneXXXX_0x where x > 0
    match = re.match(r'^(scene\d+)_0([1-9]\d*)$', subfolder_name)
    if not match:
        print(f"  Subfolder {subfolder_name} does not match pattern sceneXXXX_0x (x > 0), skipping CSV update")
        return False
    
    scene_base = match.group(1)  # e.g., scene0001
    scene_00_name = f"{scene_base}_00"
    scene_00_path = os.path.join(root_dir, scene_00_name)
    
    if not os.path.exists(scene_00_path):
        print(f"  Warning: Corresponding scene_00 folder not found: {scene_00_path}")
        return False
    
    csv_path = os.path.join(subfolder, "matched_instance_correspondence_to_00.csv")
    csv_bk_path = os.path.join(subfolder, "matched_instance_correspondence_to_00.csv.old")
    
    # Backup existing CSV if it exists
    if os.path.exists(csv_path):
        try:
            shutil.copy2(csv_path, csv_bk_path)
            print(f"  Backed up CSV to: {csv_bk_path}")
        except Exception as e:
            print(f"  Warning: Failed to backup CSV: {e}")
    
    # Load topology maps
    topomap_target_path = os.path.join(subfolder, "topology_map.json")
    topomap_source_path = os.path.join(scene_00_path, "topology_map.json")
    
    if not os.path.exists(topomap_target_path) or not os.path.exists(topomap_source_path):
        print(f"  Warning: Topology maps not found, skipping CSV update")
        return False
    
    print(f"  Loading topology maps...")
    with open(topomap_target_path, "r") as f:
        topomap_target = TopologyMap()
        topomap_target.read_from_json(f.read())
    
    with open(topomap_source_path, "r") as f:
        topomap_source = TopologyMap()
        topomap_source.read_from_json(f.read())
    
    # Extract keypoints from topology maps
    keypoints_target = []
    keypoints_source = []
    target_instance_ids = []
    source_instance_ids = []
    
    for node_id, node in topomap_target.object_nodes.nodes.items():
        try:
            instance_id = int(node_id)
            keypoints_target.append(np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32))
            target_instance_ids.append(instance_id)
        except ValueError:
            continue
    
    for node_id, node in topomap_source.object_nodes.nodes.items():
        try:
            instance_id = int(node_id)
            keypoints_source.append(np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32))
            source_instance_ids.append(instance_id)
        except ValueError:
            continue
    
    # Load transformation matrix if it exists (transforms scene_00 to scene_0x)
    # Apply transformation to scene_00 keypoints to align with scene_0x coordinate system
    transformation_path = os.path.join(subfolder, "transformation.npy")
    transformation = None
    keypoints_source_original = [kpt.copy() for kpt in keypoints_source]  # Keep original for debugging
    if os.path.exists(transformation_path):
        transformation = np.load(transformation_path)
        print(f"  Loaded transformation matrix from: {transformation_path}")
        # Transform scene_00 keypoints to align with scene_0x
        keypoints_source_array = np.array(keypoints_source)
        keypoints_source_homogeneous = np.hstack([keypoints_source_array, np.ones((keypoints_source_array.shape[0], 1))])
        keypoints_source_transformed = (transformation @ keypoints_source_homogeneous.T).T
        keypoints_source_transformed = keypoints_source_transformed[:, :3]
        # Convert back to list of arrays
        keypoints_source = [keypoints_source_transformed[i] for i in range(len(keypoints_source_transformed))]
        print(f"  Applied transformation to scene_00 keypoints ({len(keypoints_source)} keypoints)")
        # Debug: show transformation effect on first few keypoints
        if len(keypoints_source) > 0:
            sample_idx = min(2, len(keypoints_source) - 1)
            print(f"    Sample transformation: {keypoints_source_original[sample_idx]} -> {keypoints_source[sample_idx]}")
    else:
        print(f"  WARNING: No transformation.npy found, using keypoints directly from topology maps")
        print(f"    This may cause incorrect distance calculations if scenes are not aligned!")
    
    # Load PLY files and extract instances
    # Require both cleaned PLY files to exist for consistency
    ply_target_path = os.path.join(subfolder, "instance_cloud_cleaned.ply")
    ply_source_path = os.path.join(scene_00_path, "instance_cloud_cleaned.ply")
    
    if not os.path.exists(ply_target_path):
        print(f"  Warning: Cleaned PLY file not found for target: {ply_target_path}")
        print(f"  Skipping CSV update (both cleaned PLY files required)")
        return False
    
    if not os.path.exists(ply_source_path):
        print(f"  Warning: Cleaned PLY file not found for source (scene_00): {ply_source_path}")
        print(f"  Skipping CSV update (both cleaned PLY files required)")
        return False
    
    print(f"  Extracting instances from PLY files...")
    instances_target = extract_instances_from_ply(ply_target_path, three_channel_id=three_channel_id)
    instances_source = extract_instances_from_ply(ply_source_path, three_channel_id=three_channel_id)
    
    # Load instance labels or BERT embeddings
    bert_embeddings_target = {}
    bert_embeddings_source = {}
    names_target = {}
    names_source = {}
    
    if use_bert:
        bert_target_path = os.path.join(subfolder, "instance_bert_embeddings.json")
        bert_source_path = os.path.join(scene_00_path, "instance_bert_embeddings.json")
        bert_embeddings_target = load_instance_bert_embeddings(bert_target_path)
        bert_embeddings_source = load_instance_bert_embeddings(bert_source_path)
        
        # Convert string keys to int if needed
        if bert_embeddings_target and len(bert_embeddings_target) > 0:
            if isinstance(list(bert_embeddings_target.keys())[0], str):
                bert_embeddings_target = {int(k): v for k, v in bert_embeddings_target.items()}
        if bert_embeddings_source and len(bert_embeddings_source) > 0:
            if isinstance(list(bert_embeddings_source.keys())[0], str):
                bert_embeddings_source = {int(k): v for k, v in bert_embeddings_source.items()}
        
        if not bert_embeddings_target or not bert_embeddings_source:
            print(f"  Warning: BERT embeddings not found, falling back to name matching")
            use_bert = False
    
    # Always load names (needed for fallback or regular mode)
    names_target_path = os.path.join(subfolder, "instance_name_map.csv")
    names_source_path = os.path.join(scene_00_path, "instance_name_map.csv")
    names_target_raw = load_instance_labels(names_target_path)
    names_source_raw = load_instance_labels(names_source_path)
    # Convert keys to int if needed
    names_target = {int(k): v for k, v in names_target_raw.items()} if names_target_raw else {}
    names_source = {int(k): v for k, v in names_source_raw.items()} if names_source_raw else {}
    
    # Match instances
    # Note: keypoints_source are already transformed to scene_0x coordinate system
    # The distance threshold is applied during matching to filter out matches where keypoints are too far apart
    print(f"  Matching instances (keypoint distance threshold: {keypoint_distance_threshold:.2f}m)...")
    print(f"    Note: Distance check is applied after transformation to scene_0x coordinate system")
    if openset and use_bert:
        # Openset: use BERT embeddings with lower threshold
        match_dict = match_instances_with_bert_embeddings(
            instances_target, instances_source, bert_embeddings_target, bert_embeddings_source,
            keypoints_target, keypoints_source, target_instance_ids, source_instance_ids,
            dist_threshold=0.2, bert_name_similarity_threshold=0.4, min_overlap_ratio=0.1,
            keypoint_distance_threshold=keypoint_distance_threshold
        )
    else:
        # Regular or openset fallback: use name matching
        match_dict = match_instances_with_names(
            instances_target, instances_source, names_target, names_source,
            keypoints_target, keypoints_source, target_instance_ids, source_instance_ids,
            dist_threshold=0.15, min_overlap_ratio=0.1,
            keypoint_distance_threshold=keypoint_distance_threshold
        )
    
    print(f"  Matched {len(match_dict)} instances (after keypoint distance filtering)")
    
    # Verify that all matches satisfy the distance threshold
    print(f"  Verifying keypoint distances for all matches...")
    target_instance_ids2seq_dict = {id_: i for i, id_ in enumerate(target_instance_ids)}
    source_instance_ids2seq_dict = {id_: i for i, id_ in enumerate(source_instance_ids)}
    violations = []
    distances = []
    for id_B, id_A in match_dict.items():
        if id_B not in target_instance_ids2seq_dict or id_A not in source_instance_ids2seq_dict:
            continue
        keypoint_B = keypoints_target[target_instance_ids2seq_dict[id_B]]
        keypoint_A = keypoints_source[source_instance_ids2seq_dict[id_A]]
        keypoint_dist = np.linalg.norm(keypoint_A - keypoint_B)
        distances.append((id_B, id_A, keypoint_dist))
        if keypoint_dist > keypoint_distance_threshold:
            violations.append((id_B, id_A, keypoint_dist))
    
    # Show distance statistics
    if distances:
        dist_values = [d[2] for d in distances]
        print(f"  Distance statistics: min={min(dist_values):.3f}m, max={max(dist_values):.3f}m, mean={np.mean(dist_values):.3f}m, median={np.median(dist_values):.3f}m")
        # Show matches with largest distances
        distances_sorted = sorted(distances, key=lambda x: x[2], reverse=True)
        print(f"  Top 10 matches with largest distances:")
        for id_B, id_A, dist in distances_sorted[:10]:
            keypoint_B = keypoints_target[target_instance_ids2seq_dict[id_B]]
            keypoint_A = keypoints_source[source_instance_ids2seq_dict[id_A]]
            print(f"    Instance {id_B} -> {id_A}: distance = {dist:.3f}m")
            print(f"      scene_0x keypoint: {keypoint_B}")
            print(f"      scene_00 keypoint (transformed): {keypoint_A}")
        
        # Warn about matches close to threshold
        near_threshold = [d for d in distances if d[2] > keypoint_distance_threshold * 0.8]
        if near_threshold:
            print(f"  Warning: {len(near_threshold)} matches are within 80% of threshold ({keypoint_distance_threshold * 0.8:.3f}m)")
    
    if violations:
        print(f"  WARNING: Found {len(violations)} matches that violate distance threshold ({keypoint_distance_threshold:.2f}m):")
        for id_B, id_A, dist in violations:
            print(f"    Instance {id_B} -> {id_A}: distance = {dist:.3f}m")
        # Remove violating matches
        print(f"  Removing {len(violations)} violating matches...")
        for id_B, id_A, _ in violations:
            if id_B in match_dict:
                del match_dict[id_B]
        print(f"  After removing violations: {len(match_dict)} matches remain")
    else:
        print(f"  All {len(match_dict)} matches satisfy distance threshold (≤ {keypoint_distance_threshold:.2f}m)")
    
    # Visualize if requested
    if visualize:
        print(f"  Visualization requested: visualize={visualize}, matches={len(match_dict)}")
        if len(match_dict) > 0:
            visualize_correspondence_vectors(
                subfolder, scene_00_path,
                keypoints_target, keypoints_source,
                target_instance_ids, source_instance_ids,
                match_dict
            )
        else:
            print(f"  Warning: No matches found, skipping visualization (use --visualize with matches to see visualization)")
    
    # Save CSV
    try:
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "instance_id_in_00"])
            for id_B, id_A in match_dict.items():
                writer.writerow([id_B, id_A])
        print(f"  Saved updated CSV → {csv_path}")
        return True
    except Exception as e:
        print(f"  Error: Failed to save CSV: {e}")
        return False


###############################################################
# Update topology map with filtered data
###############################################################
def update_topology_map_with_filtered_data(topology_map, updated_info, removed_node_ids):
    """
    Update topology map nodes with filtered positions and bboxes.
    Remove nodes that were completely filtered out.
    
    Args:
        topology_map: TopologyMap object to update
        updated_info: Dictionary mapping node_id to updated info (position, bbox)
        removed_node_ids: Set of node IDs to remove
    """
    # Remove nodes that were completely filtered out
    for node_id in removed_node_ids:
        if node_id in topology_map.object_nodes.nodes:
            topology_map.object_nodes.remove_node(node_id)
            print(f"  Removed node {node_id} (all points filtered out)")
    
    # Update remaining nodes with filtered data
    for node_id, info in updated_info.items():
        if node_id not in topology_map.object_nodes.nodes:
            continue
        
        node = topology_map.object_nodes.nodes[node_id]
        
        # Update position
        node.position = info['position']
        
        # Update bbox if available and node has a shape
        if info['bbox'] is not None and node.shape is not None:
            # Check if shape is OrientedBox or Cylinder
            shape_type = None
            if hasattr(node.shape, 'type'):
                shape_type = node.shape.type()
            elif hasattr(node.shape, '__class__'):
                shape_type = node.shape.__class__.__name__
            
            if 'OrientedBox' in str(shape_type):
                # Update OrientedBox dimensions
                node.shape.height = info['bbox']['height']
                node.shape.width = info['bbox']['width']
                node.shape.length = info['bbox']['length']
            elif 'Cylinder' in str(shape_type):
                # For cylinder, update height and set radius based on width/length average
                node.shape.height = info['bbox']['height']
                avg_radius = (info['bbox']['width'] + info['bbox']['length']) / 4.0
                node.shape.radius = avg_radius
    
    # Remove edges that reference removed nodes
    for hypothesis_id, hypothesis in list(topology_map.edge_hypotheses.items()):
        edges_to_remove = []
        for edge_id, edge in hypothesis.edges.items():
            if edge.source_id in removed_node_ids or edge.target_id in removed_node_ids:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            # Extract source and target IDs from edge_id (format: "source_id-target_id")
            parts = edge_id.split('-')
            if len(parts) == 2:
                hypothesis.remove_edge_by_ids(parts[0], parts[1])


###############################################################
# Main processing script
###############################################################
def process_instance_clouds(root_dir, nb_neighbors=20, std_ratio=2.0, eps=0.05, min_points=10, openset=False,
                            visualize=False, keypoint_distance_threshold=1.0):
    """
    - Find all first-level subfolders containing instance_cloud.ply
    - Read each PLY
    - For each unique RGB color, filter its points
    - Merge all cleaned points → save to instance_cloud_cleaned.ply (overwrite if exists)
    - Load topology_map.json, backup as topology_map.json.bk
    - Update topology map: remove filtered-out nodes, update positions and bboxes
    - Save updated topology_map.json
    """
    ply_paths = []
    
    # 1. Search only first-level subfolders
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if os.path.isdir(sub):
            ply_path = os.path.join(sub, "instance_cloud.ply")
            if os.path.exists(ply_path):
                ply_paths.append(ply_path)

    if len(ply_paths) == 0:
        print("No instance_cloud.ply found.")
        return
    
    # 2. Process each instance_cloud.ply
    for ply_path in ply_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {ply_path}")
        print(f"{'='*60}")
        
        subfolder = os.path.dirname(ply_path)
        topology_map_path = os.path.join(subfolder, "topology_map.json")
        topology_map_bk_path = os.path.join(subfolder, "topology_map.json.bk")
        out_ply_path = ply_path.replace("instance_cloud.ply", "instance_cloud_cleaned.ply")

        # Load topology map if it exists
        topology_map = None
        if os.path.exists(topology_map_path):
            print(f"Loading topology map from: {topology_map_path}")
            try:
                with open(topology_map_path, 'r') as f:
                    topology_map = TopologyMap()
                    topology_map.read_from_json(f.read())
                print(f"Loaded topology map with {len(topology_map.object_nodes.nodes)} object nodes")
            except Exception as e:
                print(f"Warning: Failed to load topology map: {e}")
                topology_map = None
        else:
            print(f"Warning: Topology map not found at {topology_map_path}, skipping topology map update")

        # Backup topology map if it exists
        if topology_map is not None:
            try:
                shutil.copy2(topology_map_path, topology_map_bk_path)
                print(f"Backed up topology map to: {topology_map_bk_path}")
            except Exception as e:
                print(f"Warning: Failed to backup topology map: {e}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if len(pts) == 0:
            print(f"Warning: Empty point cloud in {ply_path}, skipping")
            continue

        # Convert to 0–255 for discrete instance colors
        if colors.max() <= 1.0:
            rgb_int = (colors * 255).astype(np.uint8)
        else:
            rgb_int = colors.astype(np.uint8)

        # Extract instance IDs from RGB (R channel for single-channel encoding, or R=G=B)
        # Check if R=G=B (single-channel encoding) or use R channel as instance_id
        instance_ids = rgb_int[:, 0].astype(int)

        # Get unique instance IDs
        unique_instance_ids = np.unique(instance_ids)
        unique_instance_ids = unique_instance_ids[unique_instance_ids != 0]  # Exclude background

        print(f"Found {len(unique_instance_ids)} unique instance IDs in PLY")

        # Create mapping from instance_id to node_id
        # Also identify nodes with name "unknown" to remove them
        instance_to_node = {}
        unknown_node_ids = set()  # Node IDs with name "unknown" to remove
        if topology_map is not None:
            for node_id, node in topology_map.object_nodes.nodes.items():
                try:
                    instance_id = int(node_id)
                    # Skip nodes with name "unknown"
                    if node.name == "unknown":
                        unknown_node_ids.add(node_id)
                        print(f"  Instance {instance_id} (node {node_id}): name is 'unknown', will be removed")
                        continue
                    instance_to_node[instance_id] = node_id
                except ValueError:
                    # Node ID is not an integer, skip
                    pass

        merged_points = []
        merged_colors = []
        updated_info = {}  # Maps node_id to updated position and bbox
        removed_node_ids = set(unknown_node_ids)  # Node IDs to remove (start with unknown nodes)

        # Process each instance
        for instance_id in unique_instance_ids:
            # Skip instances that correspond to "unknown" nodes
            # Check if this instance_id has a corresponding node, and if that node is "unknown"
            if instance_id in instance_to_node:
                node_id = instance_to_node[instance_id]
                if node_id in unknown_node_ids:
                    print(f"  Instance {instance_id} (node {node_id}): skipping (name is 'unknown')")
                    continue
            else:
                # Instance ID not in topology map - check if it might be an "unknown" node
                # by checking if there's a node with this ID that has name "unknown"
                if topology_map is not None:
                    node_id_str = str(instance_id)
                    if node_id_str in topology_map.object_nodes.nodes:
                        node = topology_map.object_nodes.nodes[node_id_str]
                        if node.name == "unknown":
                            unknown_node_ids.add(node_id_str)
                            removed_node_ids.add(node_id_str)
                            print(f"  Instance {instance_id} (node {node_id_str}): skipping (name is 'unknown', not in instance_to_node)")
                            continue
                    else:
                        # Instance ID in PLY but not in topology map - skip it
                        print(f"  Instance {instance_id}: skipping (not in topology map)")
                        continue
                else:
                    # No topology map available - skip instances not in instance_to_node
                    print(f"  Instance {instance_id}: skipping (not in topology map)")
                    continue
            
            mask = instance_ids == instance_id
            pts_c = pts[mask]

            if len(pts_c) < min_points:
                # Too few points, mark for removal if node exists
                if instance_id in instance_to_node:
                    node_id = instance_to_node[instance_id]
                    removed_node_ids.add(node_id)
                    print(f"  Instance {instance_id} (node {node_id}): too few points ({len(pts_c)}), will be removed")
                continue

            # Filter these points
            filtered = filter_point_cloud_outliers(pts_c, nb_neighbors, std_ratio, eps, min_points)

            if len(filtered) == 0:
                # All points filtered out, mark for removal if node exists
                if instance_id in instance_to_node:
                    node_id = instance_to_node[instance_id]
                    removed_node_ids.add(node_id)
                    print(f"  Instance {instance_id} (node {node_id}): all points filtered out, will be removed")
                continue

            # Add filtered points back with the same color
            merged_points.append(filtered)
            # Get original color for this instance
            original_color = rgb_int[mask][0]
            merged_colors.append(np.tile(original_color / 255.0, (filtered.shape[0], 1)))

            # Update topology map if node exists
            if instance_id in instance_to_node:
                node_id = instance_to_node[instance_id]
                node = topology_map.object_nodes.nodes[node_id]
                
                # Recalculate center by averaging filtered points
                new_center = np.mean(filtered, axis=0).astype(np.float32)
                
                # Recalculate bbox from filtered points (axis-aligned bounding box)
                min_bounds = np.min(filtered, axis=0)
                max_bounds = np.max(filtered, axis=0)
                bbox_size = max_bounds - min_bounds
                
                # Update node info
                updated_info[node_id] = {
                    'position': new_center,
                    'bbox': {
                        'height': bbox_size[2],  # z-axis (vertical)
                        'width': bbox_size[0],   # x-axis
                        'length': bbox_size[1]   # y-axis
                    }
                }
                
                old_center = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
                distance = np.linalg.norm(new_center - old_center)
                
                if distance > 0.01:  # Only print if change is significant
                    print(f"  Instance {instance_id} (node {node_id}): center updated from {old_center} to {new_center} (distance: {distance:.4f}m)")
                    print(f"    Bbox size: height={bbox_size[2]:.3f}, width={bbox_size[0]:.3f}, length={bbox_size[1]:.3f}")

        # 3. Merge all points and save cleaned PLY
        if len(merged_points) == 0:
            print("Warning: No points after filtering. Creating empty cleaned PLY.")
            merged_points = np.empty((0, 3))
            merged_colors = np.empty((0, 3))
        else:
            merged_points = np.concatenate(merged_points, axis=0)
            merged_colors = np.concatenate(merged_colors, axis=0)

        # Save cleaned PLY (overwrite if exists)
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(merged_points)
        out_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud(out_ply_path, out_pcd)
        print(f"Saved cleaned cloud → {out_ply_path} ({len(merged_points)} points)")

        # 4. Update topology map
        if topology_map is not None:
            print(f"\nUpdating topology map...")
            print(f"  Nodes to update: {len(updated_info)}")
            print(f"  Nodes to remove: {len(removed_node_ids)}")
            
            update_topology_map_with_filtered_data(topology_map, updated_info, removed_node_ids)
            
            # Save updated topology map
            try:
                with open(topology_map_path, 'w') as f:
                    f.write(topology_map.write_to_json())
                print(f"Saved updated topology map → {topology_map_path}")
                print(f"  Remaining object nodes: {len(topology_map.object_nodes.nodes)}")
            except Exception as e:
                print(f"Error: Failed to save updated topology map: {e}")
        
        # 5. Update instance correspondence CSV if this is a sceneXXXX_0x folder
        print(f"\nChecking for instance correspondence CSV update...")
        if update_instance_correspondence_csv(subfolder, root_dir, openset=openset,
                                              visualize=visualize,
                                              keypoint_distance_threshold=keypoint_distance_threshold):
            print(f"  Successfully updated instance correspondence CSV")
        else:
            print(f"  Skipped CSV update (not a sceneXXXX_0x folder or missing files)")


###############################################################
# Run
###############################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter instance_cloud.ply files and update topology maps")
    parser.add_argument("root_dir", type=str, help="Top-level directory containing subfolders")
    parser.add_argument("--nb_neighbors", type=int, default=20,
                       help="Number of neighbors for statistical outlier removal")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                       help="Standard deviation ratio threshold for outlier removal")
    parser.add_argument("--eps", type=float, default=0.05,
                       help="DBSCAN clustering distance threshold")
    parser.add_argument("--min_points", type=int, default=10,
                       help="Minimum points per cluster")
    parser.add_argument("--openset", action="store_true", default=False,
                       help="Whether this is an openset scene (automatically uses BERT embeddings and three_channel_id)")
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Visualize correspondence vectors and both maps")
    parser.add_argument("--keypoint_distance_threshold", type=float, default=1.0,
                       help="Maximum distance between matched keypoints after transformation (meters, default: 1.0)")

    args = parser.parse_args()
    process_instance_clouds(
        args.root_dir,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        eps=args.eps,
        min_points=args.min_points,
        openset=args.openset,
        visualize=args.visualize,
        keypoint_distance_threshold=args.keypoint_distance_threshold
    )
