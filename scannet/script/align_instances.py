import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import csv
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
from sentence_transformers import SentenceTransformer

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, "..", "..", "script", "utils"))
from result_visualization import visualize_inference_results_points

sys.path.append(os.path.join(file_path, "..", "..", "script", "include"))
from topology_map import TopologyMap


def resolve_ori_scan_mesh_path(ori_scan_dir, scene_name):
    """
    Mesh used for ICP when aligning in original scan coordinates.
    Tries ScanNet-style {scene}_vh_clean_2.ply, then openset
    instance_cloud_with_background.ply in the same scene folder.
    """
    scene_dir = os.path.join(ori_scan_dir, scene_name)
    candidates = (
        os.path.join(scene_dir, scene_name + "_vh_clean_2.ply"),
        os.path.join(scene_dir, "instance_cloud_with_background.ply"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    tried = ", ".join(os.path.basename(p) for p in candidates)
    raise FileNotFoundError(
        f"No mesh for scene {scene_name!r} under {scene_dir!r} (tried: {tried})"
    )


def align_point_clouds_with_icp(pcd_source, pcd_target, voxel_size=0.05, visualize=True, save_aligned_ply=True, save_ply_path=None):
    """
    Aligns source point cloud to target using RANSAC + ICP.
    
    Parameters:
        source_path (str): Path to source .ply file (e.g., B.ply).
        target_path (str): Path to target .ply file (e.g., A.ply).
        voxel_size (float): Downsampling voxel size.
        visualize (bool): Whether to visualize the result.

    Returns:
        np.ndarray: 4x4 transformation matrix aligning source to target.
    """
    def preprocess(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return pcd_down, fpfh

    # Preprocess and extract features
    source_down, fpfh_source = preprocess(pcd_source, voxel_size)
    target_down, fpfh_target = preprocess(pcd_target, voxel_size)

    point_num_source = len(source_down.points)
    point_num_target = len(target_down.points)
    min_point_num = min(point_num_source, point_num_target)

    # Show the point clouds before ransac
    if visualize:
        o3d.visualization.draw_geometries([source_down, target_down], window_name="Point Clouds Before RANSAC")

    # RANSAC global alignment
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        fpfh_source, fpfh_target,
        mutual_filter=False,
        max_correspondence_distance=voxel_size * 5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, min_point_num)
    )

    print("RANSAC initial alignment:")
    print(result_ransac.transformation)
    print(f"RANSAC inlier ratio: {result_ransac.inlier_rmse}")
    print(f"RANSAC final fitness: {result_ransac.fitness}")

    # SHow the ransac result
    if visualize:
        ransac_source_down = source_down.transform(result_ransac.transformation)
        o3d.visualization.draw_geometries([ransac_source_down, target_down], window_name="RANSAC Initial Alignment")

    # ICP refinement
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target,
        max_correspondence_distance=voxel_size * 1.5,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    final_transformation = result_icp.transformation
    print("ICP refined transformation:")
    print(final_transformation)

    # Transform and visualize
    if visualize or (save_aligned_ply and save_ply_path is not None):
        pcd_source_transformed = pcd_source.transform(final_transformation)
        # Copy and color
        pcd_target_viz = o3d.geometry.PointCloud()
        pcd_target_viz.points = pcd_target.points
        pcd_target_viz.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0]] * len(pcd_target.points))  # Red

        pcd_source_viz = o3d.geometry.PointCloud()
        pcd_source_viz.points = pcd_source_transformed.points
        pcd_source_viz.colors = o3d.utility.Vector3dVector(
            [[0, 1, 0]] * len(pcd_source_transformed.points))  # Green

        # Combine point clouds
        combined_pcd = pcd_target_viz + pcd_source_viz

        if visualize:
            o3d.visualization.draw_geometries([combined_pcd], window_name="Final Alignment (Red = Target, Green = Source Aligned)")
        
        if save_aligned_ply and save_ply_path is not None:
            o3d.io.write_point_cloud(save_ply_path, combined_pcd)
            print(f"Saved aligned point cloud to {save_ply_path}")

    return final_transformation


def load_instance_labels(csv_path):
    """
    Load instance labels from a CSV file.
    """
    df = pd.read_csv(csv_path)
    instance_names = {row['instance_id']: row['name'] for _, row in df.iterrows()}
    return instance_names


def load_instance_bert_embeddings(json_path):
    """
    Load instance BERT embeddings from a JSON file.
    """
    with open(json_path, 'r') as f:
        instance_bert_embeddings = json.load(f)
    return instance_bert_embeddings

def extract_instances(pcd, three_channel_id=False):
    """
    Extract instances point clouds from a point cloud.
    """
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    if not three_channel_id:
        instance_ids = (colors[:, 0] * 255).astype(int)  # R channel ∈ [0,1]
    else:
        instance_ids = (colors[:, 0] * 255 + colors[:, 1] * 255 * 255 + colors[:, 2] * 255 * 255 * 255).astype(int)

    instances = {}
    for id_ in np.unique(instance_ids):
        mask = instance_ids == id_
        instances[id_] = points[mask]
    return instances

def compute_overlap_score(pts1, pts2, dist_threshold=0.03):
    """
    Compute the overlap score between two point clouds of two instances.
    """
    if len(pts1) < 3 or len(pts2) < 3:
        return 0.0, 0.0, 1e6  # Return three values: match_ratio, miou, avg_dist
    tree1 = cKDTree(pts1)
    dists, _ = tree1.query(pts2, k=1)
    match_num1 = np.sum(dists < dist_threshold)
    match_ratio1 = match_num1 / len(pts2)

    tree2 = cKDTree(pts2)
    dists, _ = tree2.query(pts1, k=1)
    match_num2 = np.sum(dists < dist_threshold)
    match_ratio2 = match_num2 / len(pts1)
    
    match_ratio = max(match_ratio1, match_ratio2) # Use the larger one
    miou = match_num1 / (len(pts1) + len(pts2) - match_num1)
    avg_dist = np.mean(dists) # Average distance from pts2 to pts1
    
    return match_ratio, miou, avg_dist


def bert_name_similarity(embeddings1, embeddings2):
    """
    Compute the cosine similarity between two names using BERT.
    """
    if embeddings1 is None or embeddings2 is None:
        return 0.0
    
    # Reshape 1D arrays to 2D for cosine_similarity
    embeddings1_2d = np.array(embeddings1).reshape(1, -1)
    embeddings2_2d = np.array(embeddings2).reshape(1, -1)
    
    return 1 - cosine_similarity(embeddings1_2d, embeddings2_2d)[0, 0]

def compute_all_bert_similarities(bert_embeddings_A, bert_embeddings_B):
    """
    Precompute all BERT similarities between embeddings A and B in a batch.
    Returns a dictionary mapping (id_A, id_B) to similarity score.
    """
    similarities = {}
    
    # Convert embeddings to numpy arrays and create lists for vectorized computation
    embeddings_A_list = []
    id_A_list = []
    embeddings_B_list = []
    id_B_list = []
    
    for id_A, emb in bert_embeddings_A.items():
        if emb is not None:
            embeddings_A_list.append(np.array(emb))
            id_A_list.append(str(id_A))
    
    for id_B, emb in bert_embeddings_B.items():
        if emb is not None:
            embeddings_B_list.append(np.array(emb))
            id_B_list.append(str(id_B))
    
    if not embeddings_A_list or not embeddings_B_list:
        return similarities
    
    # Stack all embeddings into matrices
    embeddings_A_matrix = np.vstack(embeddings_A_list)  # Shape: (n_A, embedding_dim)
    embeddings_B_matrix = np.vstack(embeddings_B_list)  # Shape: (n_B, embedding_dim)
    
    # Compute all similarities in one operation
    # cosine_similarity returns matrix of shape (n_A, n_B)
    similarity_matrix = cosine_similarity(embeddings_A_matrix, embeddings_B_matrix)
    
    for i, id_A in enumerate(id_A_list):
        for j, id_B in enumerate(id_B_list):
            similarities[(id_A, id_B)] = similarity_matrix[i, j]
    
    return similarities

def match_instances_with_bert_embeddings(instances_B, instances_A, bert_embeddings_B, bert_embeddings_A, keypoints_B, keypoints_A, dist_threshold=0.2, bert_name_similarity_threshold=0.4, min_overlap_ratio=0.1):
    """
    Match instances based on their names and overlap score. Low bert_name_similarity_threshold is used to have more candidates to match.
    """
    # Precompute all BERT similarities
    print("Precomputing BERT similarities...")
    bert_similarities = compute_all_bert_similarities(bert_embeddings_A, bert_embeddings_B)
    print(f"Computed {len(bert_similarities)} BERT similarities")

    match_dict = {}
    for i in range(len(keypoints_B)):
        id_B = target_instance_ids[i]
        pts_B = instances_B[id_B]
        best_score = 0
        best_match = None
        
        for j in range(len(keypoints_A)):
            id_A = source_instance_ids[j]
            pts_A = instances_A[id_A]

            center_A = keypoints_A[j]
            center_B = keypoints_B[i]
            dist = np.linalg.norm(center_A - center_B)
            if dist > 1.5: # 1.5 meters is the threshold for the instance centers to be considered as the same instance
                continue

            bert_sim = bert_similarities.get((str(id_A), str(id_B)), 0.0)

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
                
        print(f"Best match for instance {id_B}: {best_match}, score: {best_score}")
        if best_match is not None:
            match_dict[id_B] = best_match

    return match_dict


def match_instances_with_names(instances_B, instances_A, names_B, names_A, keypoints_B, keypoints_A, dist_threshold=0.15, min_overlap_ratio=0.1):
    """
    Match instances based on their names and overlap score.
    """

    match_dict = {}
    for i in range(len(keypoints_B)):
        id_B = target_instance_ids[i]
        pts_B = instances_B[id_B]
        name_B = names_B.get(id_B)
        if name_B is None:
            # print(f"Skipping instance {id_B} - no label found")
            continue
            
        # Filter A instances to same object class
        candidates_A = {id_A: pts_A for id_A, pts_A in instances_A.items() if names_A.get(id_A) == name_B}
        if not candidates_A:
            # print(f"No candidates found for {name_B} (instance {id_B})")
            continue
        
        print(f"Found {len(candidates_A)} candidates for {name_B} (instance {id_B}): {list(candidates_A.keys())}")

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
            if dist > 1.5: # 1.5 meters is the threshold for the instance centers to be considered as the same instance
                continue

            match_ratio, miou, _ = compute_overlap_score(pts_B, pts_A, dist_threshold)
            if match_ratio > best_match_ratio and (match_ratio > min(0.8, min_overlap_ratio*2) or miou > min_overlap_ratio):
                best_match_ratio = match_ratio
                best_match = id_A
            elif dist < 0.5:
                # Also consider a best match based on the distance between the instance centers
                best_match_ratio = match_ratio
                best_match = id_A

        if best_match is not None:
            match_dict[id_B] = best_match
            print(f"Instance B: {id_B}, label: {names_B.get(id_B)} is aligned with Instance A: {best_match}, label: {names_A.get(best_match)}, score: {best_match_ratio}")

    return match_dict


def filter_matched_instances(match_dict, keypoints_B, keypoints_A_transformed):
    """
    Filter the matched instances based on the keypoints transformation. The keypoints of the matched instances should be close to the keypoints of the target instances after the transformation.
    Here the thresholds are more strict to avoid the mis-matching.
    """

    target_instance_ids2seq_dict = {}
    for i, id_ in enumerate(target_instance_ids):
        target_instance_ids2seq_dict[id_] = i
    source_instance_ids2seq_dict = {}
    for i, id_ in enumerate(source_instance_ids):
        source_instance_ids2seq_dict[id_] = i

    filtered_match_dict = {}

    for id_B, id_A in match_dict.items():
        keypoint_B = keypoints_B[target_instance_ids2seq_dict[id_B]]
        keypoint_A = keypoints_A_transformed[source_instance_ids2seq_dict[id_A]]
        if np.linalg.norm(keypoint_A - keypoint_B) < 1.0:
            filtered_match_dict[id_B] = id_A
        else:
            print(f"Instance B: {id_B}, label: {instance_labels_target.get(id_B)} is not aligned with Instance A: {id_A}, label: {instance_labels_source.get(id_A)}")
    
    return filtered_match_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans/gpt4/openset_scans/scene0027_00")
    parser.add_argument("--target_dir", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans/gpt4/openset_scans/scene0027_01")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--bias_meter", type=float, default=10.0, help="Bias the map ply for visualization")
    parser.add_argument("--no_save_aligned_ply", action="store_true")
    parser.add_argument("--ori_pt_transform", action="store_true")
    parser.add_argument(
        "--ori_scan_dir",
        type=str,
        default=None,
        help="Folder containing per-scene subdirs with mesh PLYs. "
        "Default: parent of --source_dir (drops the scene folder name).",
    )
    parser.add_argument("--use_bert_embeddings", action="store_true")
    parser.add_argument("--recalculate_bert_embeddings", action="store_true")
    parser.add_argument("--three_channel_id", action="store_true")
    parser.add_argument("--skip_transform_if_exists", type=bool, default=True)
    args = parser.parse_args()

    print(f"Source dir: {args.source_dir}")
    print(f"Target dir: {args.target_dir}")
    print(f"Visualize: {args.visualize}")
    print(f"No save ply: {args.no_save_aligned_ply}")
    print(f"Use BERT embeddings: {args.use_bert_embeddings}")
    print(f"Three channel id: {args.three_channel_id}")

    source_ply = o3d.io.read_point_cloud(os.path.join(args.source_dir, "instance_cloud.ply"))
    target_ply = o3d.io.read_point_cloud(os.path.join(args.target_dir, "instance_cloud.ply"))

    if args.skip_transform_if_exists:
        if os.path.exists(os.path.join(args.target_dir, "transformation.npy")):
            transformation = np.load(os.path.join(args.target_dir, "transformation.npy"))
            print(f"Transformation matrix loaded from {os.path.join(args.target_dir, 'transformation.npy')}")
        else:
            args.skip_transform_if_exists = False

    if args.ori_pt_transform and not args.skip_transform_if_exists:
        if args.ori_scan_dir is None:
            ori_scan_dir = os.path.dirname(os.path.abspath(args.source_dir))
        else:
            ori_scan_dir = args.ori_scan_dir
        source_scene_name = os.path.basename(args.source_dir)
        target_scene_name = os.path.basename(args.target_dir)
        source_ply_path = resolve_ori_scan_mesh_path(ori_scan_dir, source_scene_name)
        target_ply_path = resolve_ori_scan_mesh_path(ori_scan_dir, target_scene_name)
        source_ply_for_transform = o3d.io.read_point_cloud(source_ply_path)
        target_ply_for_transform = o3d.io.read_point_cloud(target_ply_path)
        print(f"Source ply for transform: {source_ply_path}")
        print(f"Target ply for transform: {target_ply_path}")
    else:
        source_ply_for_transform = source_ply
        target_ply_for_transform = target_ply

    instance_labels_source = load_instance_labels(os.path.join(args.source_dir, "instance_name_map.csv"))
    instance_labels_target = load_instance_labels(os.path.join(args.target_dir, "instance_name_map.csv"))

    print(f"Instance labels source: {instance_labels_source}")
    print(f"Instance labels target: {instance_labels_target}")
    
    # Debug: Check for None labels
    none_labels_source = [k for k, v in instance_labels_source.items() if v is None]
    none_labels_target = [k for k, v in instance_labels_target.items() if v is None]
    if none_labels_source:
        print(f"WARNING: Source instances with None labels: {none_labels_source}")
    if none_labels_target:
        print(f"WARNING: Target instances with None labels: {none_labels_target}")
    
    # Debug: Show sample of labels
    print(f"Sample source labels: {dict(list(instance_labels_source.items())[:5])}")
    print(f"Sample target labels: {dict(list(instance_labels_target.items())[:5])}")

    if args.recalculate_bert_embeddings:
        name_list_source = list(instance_labels_source.values())
        name_list_target = list(instance_labels_target.values())
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_source = model.encode(name_list_source)
        embeddings_target = model.encode(name_list_target)
        instance_label_bert_embeddings_source = {k: v for k, v in zip(instance_labels_source.keys(), embeddings_source)}
        instance_label_bert_embeddings_target = {k: v for k, v in zip(instance_labels_target.keys(), embeddings_target)}

    else:
        instance_label_bert_embeddings_source = load_instance_bert_embeddings(os.path.join(args.source_dir, "instance_bert_embeddings.json"))
        instance_label_bert_embeddings_target = load_instance_bert_embeddings(os.path.join(args.target_dir, "instance_bert_embeddings.json"))
  
    print(f"Instance label BERT embeddings source size: {len(instance_label_bert_embeddings_source)}")
    print(f"Instance label BERT embeddings target size: {len(instance_label_bert_embeddings_target)}")


    # Get the transformation matrix from the source point cloud to the target point cloud
    if not args.skip_transform_if_exists:
        transformation = align_point_clouds_with_icp(source_ply_for_transform, target_ply_for_transform, voxel_size=0.05, visualize=args.visualize, save_aligned_ply=not args.no_save_aligned_ply, save_ply_path=os.path.join(args.target_dir, "aligned_cloud_with_scan_00.ply"))
    
    # Transform the source point cloud
    source_ply.transform(transformation)

    # Show the source and target point clouds
    if args.visualize:
        o3d.visualization.draw_geometries([source_ply, target_ply], window_name="Source and Target Point Clouds")

    # Extract instances
    instances_source = extract_instances(source_ply, args.three_channel_id)
    instances_target = extract_instances(target_ply, args.three_channel_id)

    # Get the centers of the instances
    topomaps_available = False
    center_dict_source = {}
    center_dict_target = {}
    topomap_source_path = os.path.join(args.source_dir, "topology_map.json")
    topomap_target_path = os.path.join(args.target_dir, "topology_map.json")
    if os.path.exists(topomap_source_path) and os.path.exists(topomap_target_path):        
        # Load source topology map
        with open(topomap_source_path, "r") as f:
            topomap_source = TopologyMap()
            topomap_source.read_from_json(f.read())
        
        # Load target topology map
        with open(topomap_target_path, "r") as f:
            topomap_target = TopologyMap()
            topomap_target.read_from_json(f.read())
        
        topomaps_available = True

        # Extract object node positions from source topology map
        for node_id, node in topomap_source.object_nodes.nodes.items():
            center_dict_source[node_id] = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
        
        # Extract object node positions from target topology map
        for node_id, node in topomap_target.object_nodes.nodes.items():
            center_dict_target[node_id] = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
        
        print(f"Extracted {len(center_dict_source)} object node positions from source topology map")
        print(f"Extracted {len(center_dict_target)} object node positions from target topology map")
    else:
        print("Topology maps not found, skipping topology map processing")

    # Get the keypoints of the instances
    keypoints_target = []
    keypoints_source = []
    target_instance_ids = []
    source_instance_ids = []

    for ins_a_id, ins_a_pts in instances_source.items():
        if str(ins_a_id) in center_dict_source:
            center_a = center_dict_source[str(ins_a_id)]
            # print(f"Center of instance {ins_a_id} in source topology map: {center_a}")
        else:
            center_a = np.mean(ins_a_pts, axis=0)
            # print("Using the mean of the instance points as the center")
        keypoints_source.append(center_a)
        source_instance_ids.append(ins_a_id)

    # Transform the keypoints_source by the inverse transformation matrix to match the original ply coordinate
    keypoints_source_transformed = np.array(keypoints_source)
    keypoints_source_homogeneous = np.hstack([keypoints_source_transformed, np.ones((keypoints_source_transformed.shape[0], 1))])
    keypoints_source_transformed = (transformation @ keypoints_source_homogeneous.T).T
    keypoints_source_transformed = keypoints_source_transformed[:, :3]

    for ins_b_id, ins_b_pts in instances_target.items():
        if str(ins_b_id) in center_dict_target:
            center_b = center_dict_target[str(ins_b_id)]
            print(f"Center of instance {ins_b_id} in target topology map: {center_b}")
        else:
            center_b = np.mean(ins_b_pts, axis=0)
            print("Using the mean of the instance points as the center")
        keypoints_target.append(center_b)
        target_instance_ids.append(ins_b_id)

    # Match instances with names
    if args.use_bert_embeddings:
        match_dict = match_instances_with_bert_embeddings(instances_target, instances_source, instance_label_bert_embeddings_target, instance_label_bert_embeddings_source, keypoints_target, keypoints_source_transformed)
    else:
        match_dict = match_instances_with_names(instances_target, instances_source, instance_labels_target, instance_labels_source, keypoints_target, keypoints_source_transformed)
    print(f"Match dict: {match_dict}")
    print(f"Matched node number: {len(match_dict)}")
    print(f"target_instance_number: {len(target_instance_ids)}")
    print(f"source_instance_number: {len(source_instance_ids)}")
    print(f"target_instance_ids: {target_instance_ids}")
    print(f"source_instance_ids: {source_instance_ids}")

    match_dict = filter_matched_instances(match_dict, keypoints_target, keypoints_source_transformed)
    print(f"Filtered match dict length: {len(match_dict)}")

    for id_B, id_A in match_dict.items():
        print(f"Instance B: {id_B}, label: {instance_labels_target.get(id_B)} is aligned with Instance A: {id_A}, label: {instance_labels_source.get(id_A)}")

    # Save the matched instance correspondence to a CSV file
    with open(os.path.join(args.target_dir, "matched_instance_correspondence_to_00.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "instance_id_in_00"])
        for id_B, id_A in match_dict.items():
            writer.writerow([id_B, id_A])

    # Save the transformation matrix to a numpy file and a txt file (without [])
    if not args.skip_transform_if_exists:
        np.save(os.path.join(args.target_dir, "transformation.npy"), transformation)
        inv_transformation = np.linalg.inv(transformation)
        with open(os.path.join(args.target_dir, "inv_transformation.txt"), "w") as f:
            f.write(str(np.linalg.inv(transformation)).replace("[", "").replace("]", ""))

    if args.visualize:
        # visualize_inference_results_points(results, map_ply_path, frame_ply_path, frame_ply_pose_path
        # Construct the results dictionary. Consider the target map as a frame result.
        predicted_matches0 = []
            
        for ins_b_id, _ in instances_target.items():
            found_match = False
            for item_b, item_a in match_dict.items():
                if item_b == ins_b_id:
                    # Find the index of item_a in map_node_ids
                    map_node_seq_idx = -1
                    for i, map_node_id in enumerate(source_instance_ids):
                        if map_node_id == item_a:
                            map_node_seq_idx = i
                            break
                    predicted_matches0.append(map_node_seq_idx)
                    found_match = True
                    break
            if not found_match:
                predicted_matches0.append(-1)

        # print(f"keypoints_target: {len(keypoints_target)}")
        # print(f"map_node_ids: {source_instance_ids}")
        # print(f"keypoints_source: {len(keypoints_source)}")
        # print(f"frame_instance_ids: {target_instance_ids}")
        # print(f"predicted_matches0: {predicted_matches0}")

        # print(f"keypoints_source: {keypoints_source}")

        results = {
            "data": {
                "keypoints0": np.array(keypoints_target).astype(float),
                "keypoints1": np.array(keypoints_source).astype(float),
                "frame_instance_ids": np.array(target_instance_ids).astype(int),
                "map_node_ids": np.array(source_instance_ids).astype(int)
            },
            "predicted_matches0": np.array(predicted_matches0).astype(int)
        }

        visualize_inference_results_points(results, os.path.join(args.source_dir, "instance_cloud.ply"), os.path.join(args.target_dir, "instance_cloud.ply"), os.path.join(args.target_dir, "inv_transformation.txt"), bias_meter=args.bias_meter)

    print("Done")