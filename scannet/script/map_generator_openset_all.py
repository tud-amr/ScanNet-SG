import os
import sys
import numpy as np
import tqdm
from fix_name_csv import fix_csv_format
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import open3d as o3d

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, ".."))
root_dir = os.path.abspath(os.path.join(file_path, "..", ".."))
script_dir = os.path.join(root_dir, "script")
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, "include"))

from topology_map import TopologyMap
from utils.filtering_utils import filter_point_cloud_outliers


def _rotation_matrix_to_quaternion(rotation_matrix):
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    r = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(r)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return quat.astype(np.float32)


def _fit_rotated_obb(points):
    """
    Fit a rotated OBB to points and return center, size and orientation.
    Falls back to axis-aligned bbox with identity orientation if OBB fit fails.
    """
    if len(points) == 0:
        return None

    try:
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(points)
        try:
            obb = instance_pcd.get_oriented_bounding_box(robust=True)
        except TypeError:
            obb = instance_pcd.get_oriented_bounding_box()

        center = np.asarray(obb.center, dtype=np.float32)
        extent = np.asarray(obb.extent, dtype=np.float32)
        extent = np.maximum(extent, 1e-4)  # keep dimensions positive
        quat = _rotation_matrix_to_quaternion(np.asarray(obb.R, dtype=np.float32))
    except Exception:
        min_bounds = np.min(points, axis=0).astype(np.float32)
        max_bounds = np.max(points, axis=0).astype(np.float32)
        center = ((min_bounds + max_bounds) * 0.5).astype(np.float32)
        extent = np.maximum(max_bounds - min_bounds, 1e-4).astype(np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    return {
        "position": center,
        "bbox": {
            # Keep the same convention as scannet/src/generate_json.cpp:
            # OrientedBox(length=x, width=y, height=z)
            "height": float(extent[2]),  # z-axis in local OBB frame
            "width": float(extent[1]),   # y-axis in local OBB frame
            "length": float(extent[0]),  # x-axis in local OBB frame
        },
        "orientation": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def _build_instance_to_node_map(topology_map):
    """Build mapping from integer instance_id to node_id."""
    instance_to_node = {}
    for node_id, node in topology_map.object_nodes.nodes.items():
        candidates = [node_id]
        if hasattr(node, "id"):
            candidates.append(node.id)
        for candidate in candidates:
            try:
                instance_id = int(candidate)
                if instance_id not in instance_to_node:
                    instance_to_node[instance_id] = node_id
            except (ValueError, TypeError):
                continue
    return instance_to_node


def _update_topology_map_with_filtered_data(topology_map, updated_info):
    """Update topology map nodes with filtered centers, bbox sizes and orientation."""
    for node_id, info in updated_info.items():
        if node_id not in topology_map.object_nodes.nodes:
            continue

        node = topology_map.object_nodes.nodes[node_id]
        node.position = info["position"]

        if (
            "orientation" in info
            and node.shape is not None
            and hasattr(node.shape, "orientation")
            and node.shape.orientation is not None
        ):
            node.shape.orientation.x = info["orientation"]["x"]
            node.shape.orientation.y = info["orientation"]["y"]
            node.shape.orientation.z = info["orientation"]["z"]
            node.shape.orientation.w = info["orientation"]["w"]

        if info["bbox"] is None or node.shape is None:
            continue

        shape_type = None
        if hasattr(node.shape, "type"):
            shape_type = node.shape.type()
        elif hasattr(node.shape, "__class__"):
            shape_type = node.shape.__class__.__name__

        if "OrientedBox" in str(shape_type):
            node.shape.height = info["bbox"]["height"]
            node.shape.width = info["bbox"]["width"]
            node.shape.length = info["bbox"]["length"]
        elif "Cylinder" in str(shape_type):
            node.shape.height = info["bbox"]["height"]
            node.shape.radius = (info["bbox"]["width"] + info["bbox"]["length"]) / 4.0


def post_filter_scene_map(
    processed_scene_folder,
    nb_neighbors=20,
    std_ratio=2.0,
    eps=0.05,
    min_points=10,
    filtered_ply_name="instance_cloud_filtered.ply",
    filtered_topology_name="topology_map_filtered.json",
    overwrite=False,
):
    """
    Post-filter map point cloud instance-by-instance and update topology map.
    Recomputes center and bbox from filtered points, then saves results.
    """
    map_ply_path = os.path.join(processed_scene_folder, "instance_cloud.ply")
    topology_map_path = os.path.join(processed_scene_folder, "topology_map.json")

    if not os.path.exists(map_ply_path):
        print(f"Post-filter skipped: map file not found at {map_ply_path}")
        return

    if not os.path.exists(topology_map_path):
        print(f"Post-filter skipped: topology map not found at {topology_map_path}")
        return

    with open(topology_map_path, "r") as f:
        topology_map = TopologyMap()
        topology_map.read_from_json(f.read())

    pcd = o3d.io.read_point_cloud(map_ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        print(f"Post-filter skipped: empty map cloud at {map_ply_path}")
        return

    instance_ids = (colors[:, 0] * 255).astype(int)
    unique_instance_ids = np.unique(instance_ids)
    instance_to_node = _build_instance_to_node_map(topology_map)

    merged_points = []
    merged_colors = []
    updated_info = {}
    updated_count = 0

    for instance_id in unique_instance_ids:
        mask = instance_ids == instance_id
        instance_points = points[mask]
        instance_colors = colors[mask]

        if len(instance_points) == 0:
            continue

        if instance_id == 0:
            # Keep background untouched.
            filtered_points = instance_points
        else:
            filtered_points = filter_point_cloud_outliers(
                instance_points,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
                eps=eps,
                min_points=min_points,
            )
            # If filtering removed all points, keep original points for stability.
            if len(filtered_points) == 0:
                filtered_points = instance_points

        merged_points.append(filtered_points)
        merged_colors.append(np.tile(instance_colors[0], (len(filtered_points), 1)))

        if instance_id in instance_to_node and instance_id != 0 and len(filtered_points) > 0:
            node_id = instance_to_node[instance_id]
            fit_result = _fit_rotated_obb(filtered_points)
            if fit_result is not None:
                updated_info[node_id] = fit_result
                updated_count += 1

    if len(merged_points) == 0:
        print(f"Post-filter skipped: no points remained after filtering in {processed_scene_folder}")
        return

    out_points = np.concatenate(merged_points, axis=0)
    out_colors = np.concatenate(merged_colors, axis=0)

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(out_points)
    out_pcd.colors = o3d.utility.Vector3dVector(out_colors)

    _update_topology_map_with_filtered_data(topology_map, updated_info)

    if overwrite:
        out_ply_path = map_ply_path
        out_topology_path = topology_map_path
    else:
        out_ply_path = os.path.join(processed_scene_folder, filtered_ply_name)
        out_topology_path = os.path.join(processed_scene_folder, filtered_topology_name)

    o3d.io.write_point_cloud(out_ply_path, out_pcd)
    with open(out_topology_path, "w") as f:
        f.write(topology_map.write_to_json())

    print(
        f"Post-filter complete for {os.path.basename(processed_scene_folder)}: "
        f"{len(points)} -> {len(out_points)} points, updated {updated_count} node centers/bboxes."
    )
    print(f"  Saved filtered map: {out_ply_path}")
    print(f"  Saved filtered topology map: {out_topology_path}")


def process_scene(scene_folder, processed_dataset_dir, raw_images_parent_dir, 
                 exec_path, edge_distance_threshold, skip_existing, max_depth, subsample_factor,
                 post_filter, filter_nb_neighbors, filter_std_ratio, filter_eps, filter_min_points,
                 filtered_ply_name, filtered_topology_name, overwrite_filtered_outputs,
                 post_filter_only):
    """Process a single scene folder"""
    processed_scene_folder = os.path.join(processed_dataset_dir, scene_folder)
    
    raw_images_parent_dir_with_quotes = '"' + raw_images_parent_dir + '"'
    processed_dataset_dir_with_quotes = '"' + processed_dataset_dir + '"'

    print(f"Processing scene: {scene_folder}")

    # Check if the processed scene folder exists
    if not os.path.exists(processed_scene_folder):
        print(f"Processed scene folder does not exist for {scene_folder}")
        print(f"Please generate the instance segmentation first. Skipping...")
        return

    if not post_filter_only:
        # Run the openset_ply_map.cpp to generate the map
        if not os.path.exists(os.path.join(processed_scene_folder, "instance_cloud.ply")) or not skip_existing:
            original_cwd = os.getcwd()
            os.chdir(exec_path)
            print(f"running openset_ply_map for {scene_folder}")
            os.system("./openset_ply_map {} {} {} {} {} {}".format(
                scene_folder, 0, processed_dataset_dir_with_quotes, raw_images_parent_dir_with_quotes, 
                max_depth, subsample_factor))
            os.chdir(original_cwd)
        else:
            print(f"Instance cloud ply file already exists for {scene_folder}")

        ## Generate Topology Map for a scene
        original_cwd = os.getcwd()
        os.chdir(exec_path)
        print(f"running generate_json for {scene_folder}")
        ply_file = os.path.join(processed_scene_folder, "instance_cloud.ply")
        ply_file_with_quotes = '"' + ply_file + '"'
        os.system("./generate_json {} {} {} {}".format(ply_file_with_quotes, 0, 1, edge_distance_threshold))
        os.chdir(original_cwd)

        ## Fix the name of the instances in the instance_name_map.csv
        instance_name_map_path = os.path.join(processed_scene_folder, "instance_name_map.csv")
        if os.path.exists(instance_name_map_path):
            fix_csv_format(instance_name_map_path)
    else:
        print(f"Post-filter only mode for {scene_folder}: skipping map/topology generation")

    if post_filter:
        post_filter_scene_map(
            processed_scene_folder,
            nb_neighbors=filter_nb_neighbors,
            std_ratio=filter_std_ratio,
            eps=filter_eps,
            min_points=filter_min_points,
            filtered_ply_name=filtered_ply_name,
            filtered_topology_name=filtered_topology_name,
            overwrite=overwrite_filtered_outputs,
        )
    
    print(f"Completed processing scene: {scene_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw_images_parent_dir', type=str, default='/media/cc/My Passport/dataset/scannet/images/scans')
    parser.add_argument('--processed_dataset_dir', type=str, default='/media/cc/Expansion/scannet/processed/openset_scans/ram/openset_scans')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--processing_threads', type=int, default=6)
    parser.add_argument('--edge_distance_threshold', type=float, default=2.0)
    parser.add_argument('--exec_path', type=str, default='/home/cc/chg_ws/ros_ws/topomap_ws/devel/lib/semantic_topo_map')
    parser.add_argument('--max_depth', type=float, default=0.0, 
                        help='Maximum depth threshold in meters (0 = no filtering, default: 0.0)')
    parser.add_argument('--subsample_factor', type=int, default=1,
                        help='Subsample factor for depth image rows/cols (1 = read all, 2 = read every other row/col, etc., default: 1)')
    
    parser.add_argument('--post_filter', action='store_true',
                        help='Post-filter each generated map and recalculate bbox/center in topology map')
    parser.add_argument('--post_filter_only', action='store_true',
                        help='Run only post-filtering on existing instance_cloud.ply/topology_map.json')
    
    parser.add_argument('--filter_nb_neighbors', type=int, default=20,
                        help='Number of neighbors for statistical outlier removal in post-filtering')
    parser.add_argument('--filter_std_ratio', type=float, default=2.0,
                        help='Standard deviation ratio for outlier removal in post-filtering')
    parser.add_argument('--filter_eps', type=float, default=0.05,
                        help='DBSCAN eps distance threshold for post-filtering')
    parser.add_argument('--filter_min_points', type=int, default=10,
                        help='Minimum points per cluster in post-filtering')
    parser.add_argument('--filtered_ply_name', type=str, default='instance_cloud_filtered.ply',
                        help='Output filename for filtered point cloud when not overwriting')
    parser.add_argument('--filtered_topology_name', type=str, default='topology_map_filtered.json',
                        help='Output filename for filtered topology map when not overwriting')
    parser.add_argument('--overwrite_filtered_outputs', action='store_true',
                        help='Overwrite instance_cloud.ply and topology_map.json with filtered outputs')

    args = parser.parse_args()

    raw_images_parent_dir = args.raw_images_parent_dir
    processed_dataset_dir = args.processed_dataset_dir
    skip_existing = args.skip_existing
    exec_path = args.exec_path
    edge_distance_threshold = args.edge_distance_threshold
    max_depth = args.max_depth
    subsample_factor = args.subsample_factor
    post_filter = args.post_filter
    filter_nb_neighbors = args.filter_nb_neighbors
    filter_std_ratio = args.filter_std_ratio
    filter_eps = args.filter_eps
    filter_min_points = args.filter_min_points
    filtered_ply_name = args.filtered_ply_name
    filtered_topology_name = args.filtered_topology_name
    overwrite_filtered_outputs = args.overwrite_filtered_outputs
    post_filter_only = args.post_filter_only

    if post_filter_only and not post_filter:
        print("Warning: --post_filter_only implies --post_filter. Enabling post-filter.")
        post_filter = True

    # Find the folder names that contain the word "scene" in the scans directory
    scene_folders = [f for f in os.listdir(processed_dataset_dir) if "scene" in f]
    print("Found {} scene folders".format(len(scene_folders)))

    # Process scenes using multiple threads
    with ThreadPoolExecutor(max_workers=args.processing_threads) as executor:
        # Submit all tasks directly
        future_to_scene = {
            executor.submit(
                process_scene, 
                scene_folder, 
                processed_dataset_dir, 
                raw_images_parent_dir, 
                exec_path, 
                edge_distance_threshold, 
                skip_existing,
                max_depth,
                subsample_factor,
                post_filter,
                filter_nb_neighbors,
                filter_std_ratio,
                filter_eps,
                filter_min_points,
                filtered_ply_name,
                filtered_topology_name,
                overwrite_filtered_outputs,
                post_filter_only,
            ): scene_folder
            for scene_folder in scene_folders
        }

        # Process completed tasks with progress bar
        for future in tqdm.tqdm(as_completed(future_to_scene), total=len(scene_folders), desc="Processing scenes"):
            scene_folder = future_to_scene[future]
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as exc:
                print(f"Scene {scene_folder} generated an exception: {exc}")

    print("All scenes processed!")