import os
import sys
import json
import re
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))
sys.path.append(file_path)  # Add current directory (script/utils/) to path
from include.topology_map import TopologyMap
from filtering_utils import filter_point_cloud_outliers


def generate_instance_colors(start_id=0, end_id=255, use_colormap=True):
    """
    Generate muted, pastel-like colors for each instance that match a softer visual style.
    Uses lower saturation and consistent brightness for better visual harmony.
    
    Args:
        start_id: Starting instance ID
        end_id: Ending instance ID
        use_colormap: If True, use optimized color generation for better distinction
    
    Returns:
        numpy array of shape (end_id - start_id + 1, 3) with RGB colors in [0, 255]
    """
    num_colors = end_id - start_id + 1
    
    if use_colormap and num_colors > 1:
        # Use HSV color space with muted/pastel colors
        # Generate colors with varying hue (0-360), lower saturation (0.4-0.6), and consistent value (0.7-0.85)
        instance_colors = np.zeros((num_colors, 3), dtype=np.uint8)
        
        # Generate distinct hues by using golden angle approximation for optimal spacing
        golden_angle = 0.618033988749895  # (sqrt(5)-1)/2, provides optimal color spacing
        
        for i in range(num_colors):
            # Use golden angle spacing for hue to maximize perceptual distance
            hue = (i * golden_angle * 360) % 360
            
            # Use lower saturation for muted/pastel colors (0.4-0.6 range)
            # This creates softer, more harmonious colors similar to the image style
            sat_pattern = 0.4 + 0.2 * (i % 3) / 2.0  # Cycle through 0.4, 0.5, 0.6
            
            # Use consistent, medium-high value for good visibility (0.7-0.85)
            val_pattern = 0.7 + 0.15 * ((i // 3) % 3) / 2.0  # Cycle through 0.7, 0.775, 0.85
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360.0, sat_pattern, val_pattern)
            instance_colors[i] = (np.array(rgb) * 255).astype(np.uint8)
        
        # Ensure background (id=0) gets a neutral color if start_id is 0
        if start_id == 0:
            instance_colors[0] = np.array([128, 128, 128], dtype=np.uint8)  # Gray for background
        
    else:
        # Fallback: Use HSV with muted colors
        instance_colors = np.zeros((num_colors, 3), dtype=np.uint8)
        golden_angle = 0.618033988749895
        
        for i in range(num_colors):
            hue = (i * golden_angle * 360) % 360
            # Lower saturation (0.5) and consistent value (0.8) for muted appearance
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.5, 0.8)
            instance_colors[i] = (np.array(rgb) * 255).astype(np.uint8)
        
        if start_id == 0:
            instance_colors[0] = np.array([128, 128, 128], dtype=np.uint8)
    
    return instance_colors

def visualize_inference_results_keypoints(results):
    """
    Visualize the inference results by showing the matched keypoints.
    
    Args:
        results: Inference results dictionary
    """
    
    print("\nVisualizing inference results...")

    
    # Extract keypoints from results
    frame_keypoints = results['data']['keypoints0']
    map_keypoints = results['data']['keypoints1']
    predicted_matches0 = results['predicted_matches0']
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Frame keypoints
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(frame_keypoints[:, 0], frame_keypoints[:, 1], frame_keypoints[:, 2], 
               c='blue', s=50, label='Frame Keypoints')
    ax1.set_title('Frame Keypoints')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Plot 2: Map keypoints
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(map_keypoints[:, 0], map_keypoints[:, 1], map_keypoints[:, 2], 
               c='red', s=50, label='Map Keypoints')
    ax2.set_title('Map Keypoints')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Plot 3: Matches
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot all keypoints
    ax3.scatter(frame_keypoints[:, 0], frame_keypoints[:, 1], frame_keypoints[:, 2], 
               c='blue', s=30, alpha=0.6, label='Frame Keypoints')
    ax3.scatter(map_keypoints[:, 0], map_keypoints[:, 1], map_keypoints[:, 2], 
               c='red', s=30, alpha=0.6, label='Map Keypoints')
    
    # Draw lines for valid matches
    valid_match_indices = np.where(predicted_matches0 != -1)[0]
    for frame_idx in valid_match_indices:
        map_idx = predicted_matches0[frame_idx]
        if map_idx < len(map_keypoints):
            frame_pos = frame_keypoints[frame_idx]
            map_pos = map_keypoints[map_idx]
            ax3.plot([frame_pos[0], map_pos[0]], 
                    [frame_pos[1], map_pos[1]], 
                    [frame_pos[2], map_pos[2]], 
                    'g-', alpha=0.7, linewidth=1)
    
    ax3.set_title(f'Predicted Matches ({len(valid_match_indices)} valid)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Visualization complete. Found {len(valid_match_indices)} valid matches.")



def visualize_inference_results_points(results, map_ply_path, frame_ply_path, frame_ply_pose_path, bias_meter=5.0, angle_bias=None, align_matrix=None, instance_colors=None, match_success_list=None,
                                       gt_alignment_dict=None, filter_frame_outliers=False, filter_nb_neighbors=20, filter_std_ratio=2.0, filter_eps=0.05, filter_min_points=10, visualize_filtered_points=False,
                                       map_background_ply_path=None, point_size=1.0, keypoint_radius=0.15, match_line_radius=0.03, 
                                       show_match_confidence=False, downsample_ratio=1.0, frame_alpha=0.8, map_alpha=0.8,
                                       default_front=None, default_lookat=None, default_up=None, default_zoom=None, show_view_angles=True, show_coordinate_frame=True, show_edges=False):
    """
    Visualize the inference results using Open3D. 
    Show the ply file of the map and one or more frames in one window and bias each frame ply in the direction from map center (Cm) to frame center (Cf).
    The bias distance is bias_meter meters along the direction vector from Cm to Cf.
    Connect the center of the points belonging to the same instance in the frame ply to the center of the points belonging to the same instance in the map ply.
    If align_matrix is provided, apply it to the frame point cloud and frame keypoints after pose transformation.
    
    Args:
        results: Inference results dictionary (single) or list of dictionaries (multiple frames)
        map_ply_path: Path to the map ply file
        frame_ply_path: Path to the frame ply file (single) or list of paths (multiple frames)
        frame_ply_pose_path: Path to the frame ply pose file (single) or list of paths (multiple frames)
        bias_meter: Bias distance in meters along the direction from map center to frame center (default: 5.0)
                   Can be a single value (applied to all frames) or a list (one per frame)
        angle_bias: Optional rotation angle(s) in degrees to rotate the bias direction around z-axis (default: None)
                   Can be a single value (applied to all frames) or a list (one per frame)
                   Positive angles rotate counterclockwise when viewed from above
        align_matrix: Optional (4,4) numpy array (single) or list of arrays to further transform frame clouds and keypoints
        instance_colors: Optional pre-generated colors for tracking IDs (shared across all frames)
        match_success_list: Optional list (single) or list of lists (multiple frames) indicating match success
        gt_alignment_dict: Optional ground truth alignment dictionary mapping frame_instance_id to map_instance_id (single) or list of dicts (multiple frames)
                          Used to visualize false negatives (ground truth matches not predicted by model)
        filter_frame_outliers: Whether filtering was used for center recalculation (for reference)
        filter_nb_neighbors: Number of neighbors for statistical outlier removal
        filter_std_ratio: Standard deviation ratio threshold for outlier removal
        filter_eps: DBSCAN clustering distance threshold
        filter_min_points: Minimum points per cluster
        visualize_filtered_points: Whether to use filtered points in visualization (only applies if True)
        point_size: Point size multiplier for rendering (default: 1.0)
        keypoint_radius: Radius of keypoint spheres (default: 0.15)
        match_line_radius: Radius of match connection cylinders (default: 0.03)
        show_match_confidence: If True, color match lines by confidence score (default: False)
        downsample_ratio: Downsample point clouds for better performance (1.0 = no downsampling, default: 1.0)
        frame_alpha: Transparency for frame point cloud (0.0-1.0, default: 0.8)
        map_alpha: Transparency for map point cloud (0.0-1.0, default: 0.8)
        default_front: Default front vector for camera [x, y, z] (default: None, uses Open3D default)
        default_lookat: Default lookat point [x, y, z] (default: None, uses scene center)
        default_up: Default up vector [x, y, z] (default: None, uses [0, 0, 1])
        default_zoom: Default zoom level (default: None, uses Open3D default)
        show_view_angles: Whether to print current view angles to console (default: True)
        show_coordinate_frame: Whether to show the coordinate frame (xyz axes) (default: True)
        show_edges: Whether to show edges connecting nodes within 2 meters with light blue lines (default: False)
    """
    print("\nVisualizing inference results with point clouds...")
    
    # Convert single values to lists for uniform processing
    if not isinstance(results, list):
        results = [results]
    if not isinstance(frame_ply_path, list):
        frame_ply_path = [frame_ply_path]
    if not isinstance(frame_ply_pose_path, list):
        frame_ply_pose_path = [frame_ply_pose_path]
    if not isinstance(bias_meter, list):
        bias_meter = [bias_meter] * len(results)
    if angle_bias is not None and not isinstance(angle_bias, list):
        angle_bias = [angle_bias] * len(results)
    if align_matrix is not None and not isinstance(align_matrix, list):
        align_matrix = [align_matrix]
    if match_success_list is not None and not isinstance(match_success_list[0] if len(match_success_list) > 0 else None, list):
        match_success_list = [match_success_list]
    if gt_alignment_dict is not None and not isinstance(gt_alignment_dict, list):
        gt_alignment_dict = [gt_alignment_dict]
    
    # Validate that all lists have the same length
    num_frames = len(results)
    if len(frame_ply_path) != num_frames:
        raise ValueError(f"Mismatch: {len(frame_ply_path)} frame_ply_paths but {num_frames} results")
    if len(frame_ply_pose_path) != num_frames:
        raise ValueError(f"Mismatch: {len(frame_ply_pose_path)} frame_ply_pose_paths but {num_frames} results")
    if len(bias_meter) != num_frames:
        raise ValueError(f"Mismatch: {len(bias_meter)} bias_meter values but {num_frames} results")
    if angle_bias is not None and len(angle_bias) != num_frames:
        raise ValueError(f"Mismatch: {len(angle_bias)} angle_bias values but {num_frames} results")
    if align_matrix is not None and len(align_matrix) != num_frames:
        raise ValueError(f"Mismatch: {len(align_matrix)} align_matrices but {num_frames} results")
    if match_success_list is not None and len(match_success_list) != num_frames:
        raise ValueError(f"Mismatch: {len(match_success_list)} match_success_lists but {num_frames} results")
    if gt_alignment_dict is not None and len(gt_alignment_dict) != num_frames:
        raise ValueError(f"Mismatch: {len(gt_alignment_dict)} gt_alignment_dicts but {num_frames} results")
    
    print(f"Processing {num_frames} frame(s)...")
    
    # Load map point cloud (shared across all frames)
    if not os.path.exists(map_ply_path):
        print(f"Warning: Map ply file not found: {map_ply_path}")
        return
    
    map_cloud = o3d.io.read_point_cloud(map_ply_path)

    # Downsample map point cloud if requested (for better performance)
    if downsample_ratio < 1.0:
        print(f"Downsampling map point cloud with ratio {downsample_ratio}...")
        if len(map_cloud.points) > 0:
            map_bounds = map_cloud.get_axis_aligned_bounding_box()
            map_extent = map_bounds.get_extent()
            map_voxel_size = np.max(map_extent) * downsample_ratio
            map_cloud = map_cloud.voxel_down_sample(voxel_size=map_voxel_size)
        print(f"After downsampling: map={len(map_cloud.points)} points")
    
    print(f"Loaded map cloud with {len(map_cloud.points)} points")
    
    # Process map background if provided
    map_background_cloud = None
    if map_background_ply_path is not None:
        map_background_cloud = o3d.io.read_point_cloud(map_background_ply_path)
        map_background_colors = np.asarray(map_background_cloud.colors)
        map_background_points = np.asarray(map_background_cloud.points)
        map_background_colors_255 = (map_background_colors * 255).astype(int)
        mask = (map_background_colors_255[:, 0] == 152) & (map_background_colors_255[:, 1] == 223) & (map_background_colors_255[:, 2] == 138)
        map_background_cloud.points = o3d.utility.Vector3dVector(map_background_points[mask])
        num_points = len(map_background_cloud.points)
        map_background_cloud.colors = o3d.utility.Vector3dVector(np.tile([128, 128, 128], (num_points, 1)) / 255.0)
        print(f"Found {len(map_background_cloud.points)} background points")
    
    # Calculate map center once (shared across all frames)
    map_cloud_points = np.asarray(map_cloud.points)
    Cm = np.mean(map_cloud_points, axis=0)  # Map center
    print(f"Map center (Cm): {Cm}")
    
    # Collect all tracking IDs from map for consistent coloring
    map_colors = np.asarray(map_cloud.colors)
    map_tracking_ids = (map_colors[:, 0] * 255).astype(int)
    unique_map_ids = np.unique(map_tracking_ids)
    
    # Create color map for tracking IDs (shared across all frames)
    if instance_colors is None:
        instance_colors = generate_instance_colors(0, 255, use_colormap=True)
    
    gray_color = np.array([0.4, 0.4, 0.4])
    new_map_colors = instance_colors[map_tracking_ids] / 255.0
    new_map_colors[map_tracking_ids == 0] = gray_color
    if map_alpha < 1.0:
        new_map_colors = new_map_colors * map_alpha + (1 - map_alpha) * 0.5
    map_cloud.colors = o3d.utility.Vector3dVector(new_map_colors)
    if len(map_cloud.points) > 0:
        map_cloud.estimate_normals()
    
    # Extract map keypoints (shared across all frames - assuming all results use same map)
    map_keypoints = results[0]['data']['keypoints1'] if len(results) > 0 else None
    
    # Process each frame
    all_frame_clouds = []
    all_frame_keypoints = []
    all_lines = []
    all_keypoint_spheres = []
    all_unique_ids = set(unique_map_ids)
    
    for frame_idx in range(num_frames):
        print(f"\n--- Processing Frame {frame_idx + 1}/{num_frames} ---")
        result = results[frame_idx]
        frame_path = frame_ply_path[frame_idx]
        pose_path = frame_ply_pose_path[frame_idx]
        frame_bias = bias_meter[frame_idx]
        frame_angle_bias = angle_bias[frame_idx] if angle_bias is not None else None
        frame_align_matrix = align_matrix[frame_idx] if align_matrix is not None else None
        frame_match_success = match_success_list[frame_idx] if match_success_list is not None else None
        frame_gt_alignment = gt_alignment_dict[frame_idx] if gt_alignment_dict is not None else None
        
        # Validate frame files (skip if None - map-only visualization)
        if frame_path is None or pose_path is None:
            print(f"Skipping frame {frame_idx + 1} (map-only mode)")
            continue
        if not os.path.exists(frame_path):
            print(f"Warning: Frame ply file not found: {frame_path}, skipping...")
            continue
        if not os.path.exists(pose_path):
            print(f"Warning: Frame pose file not found: {pose_path}, skipping...")
            continue
        
        # Load frame cloud
        frame_cloud = o3d.io.read_point_cloud(frame_path)
        
        # Downsample frame if requested
        if downsample_ratio < 1.0 and len(frame_cloud.points) > 0:
            frame_bounds = frame_cloud.get_axis_aligned_bounding_box()
            frame_extent = frame_bounds.get_extent()
            frame_voxel_size = np.max(frame_extent) * downsample_ratio
            frame_cloud = frame_cloud.voxel_down_sample(voxel_size=frame_voxel_size)
        
        # Load and apply transformation
        transform_matrix = np.loadtxt(pose_path)
        frame_cloud.transform(transform_matrix)
        
        if frame_align_matrix is not None:
            inv_align_matrix = np.linalg.inv(frame_align_matrix)
            frame_cloud.transform(inv_align_matrix)
        
        print(f"Loaded frame cloud with {len(frame_cloud.points)} points")
        
        # Filter frame point cloud outliers if requested
        if visualize_filtered_points:
            print("Filtering frame point cloud outliers...")
            frame_pts = np.asarray(frame_cloud.points)
            frame_colors_orig = np.asarray(frame_cloud.colors)
            frame_tracking_ids_orig = (frame_colors_orig[:, 0] * 255).astype(int)
            
            instance_points = {}
            for iid in np.unique(frame_tracking_ids_orig):
                mask = frame_tracking_ids_orig == iid
                instance_points[iid] = {
                    'points': frame_pts[mask],
                    'colors': frame_colors_orig[mask],
                    'indices': np.where(mask)[0]
                }
            
            filtered_frame_pts = []
            filtered_frame_colors = []
            
            for iid, instance_data in instance_points.items():
                if iid == 0:
                    filtered_frame_pts.append(instance_data['points'])
                    filtered_frame_colors.append(instance_data['colors'])
                else:
                    if len(instance_data['points']) >= filter_min_points:
                        filtered_pts = filter_point_cloud_outliers(
                            instance_data['points'],
                            nb_neighbors=filter_nb_neighbors,
                            std_ratio=filter_std_ratio,
                            eps=filter_eps,
                            min_points=filter_min_points
                        )
                        if len(filtered_pts) > 0:
                            filtered_frame_pts.append(filtered_pts)
                            filtered_colors = np.zeros((len(filtered_pts), 3))
                            filtered_colors[:, 0] = iid / 255.0
                            filtered_colors[:, 1] = instance_data['colors'][0, 1]
                            filtered_colors[:, 2] = instance_data['colors'][0, 2]
                            filtered_frame_colors.append(filtered_colors)
                        else:
                            continue
                    else:
                        filtered_frame_pts.append(instance_data['points'])
                        filtered_frame_colors.append(instance_data['colors'])
            
            if len(filtered_frame_pts) > 0:
                filtered_frame_pts = np.vstack(filtered_frame_pts)
                filtered_frame_colors = np.vstack(filtered_frame_colors)
                frame_cloud.points = o3d.utility.Vector3dVector(filtered_frame_pts)
                frame_cloud.colors = o3d.utility.Vector3dVector(filtered_frame_colors)
                print(f"Filtered frame cloud: {len(frame_cloud.points)} points")
        
        # Calculate frame center and bias
        frame_cloud_points = np.asarray(frame_cloud.points)
        Cf = np.mean(frame_cloud_points, axis=0)
        direction = Cf - Cm
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-6:
            direction_unit = direction / direction_norm
            direction_unit[2] = 0.0 # Keep z no bias
            
            # Normalize x-y plane direction
            xy_norm = np.linalg.norm(direction_unit[:2])
            if xy_norm > 1e-6:
                direction_unit[:2] = direction_unit[:2] / xy_norm
            else:
                direction_unit[:2] = np.array([1.0, 0.0])  # Default to x-axis
            
            # Apply angle bias rotation around z-axis if provided
            if frame_angle_bias is not None:
                angle_rad = np.radians(frame_angle_bias)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # Rotate the x-y components around z-axis
                x_rotated = direction_unit[0] * cos_angle - direction_unit[1] * sin_angle
                y_rotated = direction_unit[0] * sin_angle + direction_unit[1] * cos_angle
                
                direction_unit[0] = x_rotated
                direction_unit[1] = y_rotated
                
                print(f"Frame {frame_idx + 1}: Rotated bias direction by {frame_angle_bias:.1f}° around z-axis")
            
            bias_vector = direction_unit * frame_bias
        else:
            print(f"Warning: Frame {frame_idx + 1} centers too close, using x-axis bias")
            default_direction = np.array([1.0, 0.0, 0.0])
            
            # Apply angle bias to default direction if provided
            if frame_angle_bias is not None:
                angle_rad = np.radians(frame_angle_bias)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                default_direction[0] = cos_angle
                default_direction[1] = sin_angle
                print(f"Frame {frame_idx + 1}: Applied {frame_angle_bias:.1f}° angle bias to default x-axis direction")
            
            bias_vector = default_direction * frame_bias
        
        print(f"Frame {frame_idx + 1} center (Cf): {Cf}")
        print(f"Bias vector: {bias_vector}")
        
        # Apply bias to frame cloud
        frame_cloud_points += bias_vector
        frame_cloud.points = o3d.utility.Vector3dVector(frame_cloud_points)
        
        # Color frame cloud
        frame_colors = np.asarray(frame_cloud.colors)
        frame_tracking_ids = (frame_colors[:, 0] * 255).astype(int)
        all_unique_ids.update(np.unique(frame_tracking_ids))
        
        new_frame_colors = instance_colors[frame_tracking_ids] / 255.0
        new_frame_colors[frame_tracking_ids == 0] = gray_color
        if frame_alpha < 1.0:
            new_frame_colors = new_frame_colors * frame_alpha + (1 - frame_alpha) * 0.5
        frame_cloud.colors = o3d.utility.Vector3dVector(new_frame_colors)
        if len(frame_cloud.points) > 0:
            frame_cloud.estimate_normals()
        
        all_frame_clouds.append(frame_cloud)
        
        # Process keypoints and matches for this frame
        frame_keypoints = result['data']['keypoints0']
        predicted_matches0 = result['predicted_matches0']
        frame_instance_ids = result['data']['frame_instance_ids']
        map_node_ids = result['data']['map_node_ids']
        
        # Transform frame keypoints
        frame_keypoints_homogeneous = np.hstack([frame_keypoints, np.ones((frame_keypoints.shape[0], 1))])
        transformed_frame_keypoints_homogeneous = (transform_matrix @ frame_keypoints_homogeneous.T).T
        if frame_align_matrix is not None:
            transformed_frame_keypoints_homogeneous = (inv_align_matrix @ transformed_frame_keypoints_homogeneous.T).T
        transformed_frame_keypoints = transformed_frame_keypoints_homogeneous[:, :3]
        transformed_frame_keypoints += bias_vector
        
        all_frame_keypoints.append(transformed_frame_keypoints)
        
        # Create match lines
        valid_match_indices = np.where(predicted_matches0 != -1)[0]
        if frame_match_success is not None:
            frame_match_success = np.array(frame_match_success)
            valid_match_success_list = frame_match_success[valid_match_indices]
        else:
            valid_match_success_list = [True] * len(valid_match_indices)
        
        print(f"Creating {len(valid_match_indices)} match connections for frame {frame_idx + 1}...")
        
        seq = 0
        for match_idx in valid_match_indices:
            map_idx = predicted_matches0[match_idx]
            if map_idx < len(map_keypoints):
                frame_pos = transformed_frame_keypoints[match_idx]
                map_pos = map_keypoints[map_idx]
                
                direction = map_pos - frame_pos
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=match_line_radius, height=length)
                    
                    if valid_match_success_list[seq]:
                        if show_match_confidence and 'matching_scores0' in result:
                            score = result['matching_scores0'][match_idx]
                            normalized_score = np.clip(score, 0.0, 1.0)
                            color = np.array([normalized_score, 1.0, 0.0])
                            cylinder.paint_uniform_color(color)
                        else:
                            cylinder.paint_uniform_color([0.0, 1.0, 0.0])
                    else:
                        cylinder.paint_uniform_color([1.0, 0.0, 0.0])
                    
                    center = (frame_pos + map_pos) / 2
                    cylinder.translate(center)
                    
                    z_axis = np.array([0, 0, 1])
                    if not np.allclose(direction, z_axis):
                        rotation_axis = np.cross(z_axis, direction)
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        cos_angle = np.dot(z_axis, direction)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        R = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                        cylinder.rotate(R, center=center)
                    
                    all_lines.append(cylinder)
                seq += 1
        
        # Create false negative lines (ground truth matches not predicted by model)
        if frame_gt_alignment is not None:
            false_negative_count = 0
            # Create mapping from frame_instance_id to frame keypoint index
            frame_instance_id_to_idx = {instance_id: idx for idx, instance_id in enumerate(frame_instance_ids)}
            # Create mapping from map_instance_id to map keypoint index
            map_instance_id_to_idx = {instance_id: idx for idx, instance_id in enumerate(map_node_ids)}
            
            # Find false negatives: ground truth matches that weren't predicted
            for frame_instance_id, gt_map_instance_id in frame_gt_alignment.items():
                # Check if this frame instance has a predicted match
                if frame_instance_id in frame_instance_id_to_idx:
                    frame_kp_idx = frame_instance_id_to_idx[frame_instance_id]
                    predicted_match_idx = predicted_matches0[frame_kp_idx]
                    
                    # Check if predicted match exists and matches ground truth
                    is_predicted = predicted_match_idx != -1
                    if is_predicted:
                        predicted_map_instance_id = map_node_ids[predicted_match_idx]
                        matches_gt = (predicted_map_instance_id == gt_map_instance_id)
                    else:
                        matches_gt = False
                    
                    # If not predicted or predicted incorrectly, check if GT match exists in map
                    if not matches_gt and gt_map_instance_id in map_instance_id_to_idx:
                        gt_map_kp_idx = map_instance_id_to_idx[gt_map_instance_id]
                        
                        # Draw false negative line (yellow/orange color)
                        frame_pos = transformed_frame_keypoints[frame_kp_idx]
                        map_pos = map_keypoints[gt_map_kp_idx]
                        
                        direction = map_pos - frame_pos
                        length = np.linalg.norm(direction)
                        if length > 0:
                            direction = direction / length
                            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=match_line_radius * 0.8, height=length)
                            
                            # Use yellow/orange color for false negatives (dashed appearance via thinner radius)
                            cylinder.paint_uniform_color([1.0, 0.65, 0.0])  # Orange color
                            
                            center = (frame_pos + map_pos) / 2
                            cylinder.translate(center)
                            
                            z_axis = np.array([0, 0, 1])
                            if not np.allclose(direction, z_axis):
                                rotation_axis = np.cross(z_axis, direction)
                                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                cos_angle = np.dot(z_axis, direction)
                                angle = np.arccos(np.clip(cos_angle, -1, 1))
                                R = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                                cylinder.rotate(R, center=center)
                            
                            all_lines.append(cylinder)
                            false_negative_count += 1
            
            if false_negative_count > 0:
                print(f"Created {false_negative_count} false negative connections (ground truth matches not predicted) for frame {frame_idx + 1}...")
        
        # Create keypoint spheres for this frame
        matched_frame_indices = set(valid_match_indices)
        for i, pos in enumerate(transformed_frame_keypoints):
            is_matched = i in matched_frame_indices
            radius = keypoint_radius * (1 if is_matched else 0.7)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pos)
            if is_matched:
                sphere.paint_uniform_color([1.0, 0.2, 0.2])
            else:
                sphere.paint_uniform_color([0.3, 0.3, 0.8])
            all_keypoint_spheres.append(sphere)
    
    # Create map keypoint spheres (only once, shared across all frames)
    if map_keypoints is not None:
        all_matched_map_indices = set()
        for result in results:
            predicted_matches0 = result['predicted_matches0']
            valid_match_indices = np.where(predicted_matches0 != -1)[0]
            all_matched_map_indices.update(predicted_matches0[valid_match_indices])
        
        for i, pos in enumerate(map_keypoints):
            is_matched = i in all_matched_map_indices
            radius = keypoint_radius * (0.975 if is_matched else 0.7)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pos)
            if is_matched:
                sphere.paint_uniform_color([1.0, 0.2, 0.2])
            else:
                sphere.paint_uniform_color([0.3, 0.3, 0.8])
            all_keypoint_spheres.append(sphere)
    
    # Create edge visualizations if requested
    all_edge_lines = []
    if show_edges:
        print("\nCreating edge visualizations...")
        edge_distance_threshold = 2.0  # Filter edges within 2 meters
        edge_line_radius = 0.02  # Radius for edge cylinders (same as default match_line_radius)
        
        # Create map edges (edges1) - shared across all frames
        if map_keypoints is not None and len(results) > 0:
            if 'edges1' in results[0]['data']:
                edges1 = results[0]['data']['edges1']
                # Convert to numpy if it's a tensor
                if hasattr(edges1, 'cpu'):
                    edges1 = edges1.cpu().numpy()
                elif hasattr(edges1, 'numpy'):
                    edges1 = edges1.numpy()
                edges1 = np.asarray(edges1)
                
                if edges1.shape[0] == 2 and edges1.shape[1] > 0:
                    map_edges_created = 0
                    map_edges_skipped = 0
                    
                    for edge_idx in range(edges1.shape[1]):
                        source_idx = int(edges1[0, edge_idx])
                        target_idx = int(edges1[1, edge_idx])
                        
                        if source_idx < len(map_keypoints) and target_idx < len(map_keypoints):
                            source_pos = map_keypoints[source_idx]
                            target_pos = map_keypoints[target_idx]
                            
                            # Calculate distance between nodes
                            distance = np.linalg.norm(target_pos - source_pos)
                            
                            # Filter by distance threshold
                            if distance <= edge_distance_threshold:
                                # Create a cylinder between source and target positions (light blue color)
                                direction = target_pos - source_pos
                                length = np.linalg.norm(direction)
                                if length > 0:
                                    direction = direction / length
                                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=edge_line_radius, height=length)
                                    cylinder.paint_uniform_color([0.5, 0.8, 1.0])  # Light blue color for map edges
                                    
                                    center = (source_pos + target_pos) / 2
                                    cylinder.translate(center)
                                    
                                    z_axis = np.array([0, 0, 1])
                                    if not np.allclose(direction, z_axis):
                                        rotation_axis = np.cross(z_axis, direction)
                                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                        cos_angle = np.dot(z_axis, direction)
                                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                                        R = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                                        cylinder.rotate(R, center=center)
                                    
                                    all_edge_lines.append(cylinder)
                                    map_edges_created += 1
                            else:
                                map_edges_skipped += 1
                    
                    print(f"  Map edges: {map_edges_created} created, {map_edges_skipped} skipped (threshold: {edge_distance_threshold:.3f}m)")
        
        # Create frame edges (edges0) - for each frame
        for frame_idx in range(num_frames):
            result = results[frame_idx]
            frame_path = frame_ply_path[frame_idx]
            pose_path = frame_ply_pose_path[frame_idx]
            frame_bias = bias_meter[frame_idx]
            frame_angle_bias = angle_bias[frame_idx] if angle_bias is not None else None
            frame_align_matrix = align_matrix[frame_idx] if align_matrix is not None else None
            
            # Skip if frame files don't exist
            if frame_path is None or pose_path is None or not os.path.exists(frame_path) or not os.path.exists(pose_path):
                continue
            
            if 'edges0' in result['data'] and frame_idx < len(all_frame_keypoints):
                edges0 = result['data']['edges0']
                transformed_frame_keypoints = all_frame_keypoints[frame_idx]
                
                # Convert to numpy if it's a tensor
                if hasattr(edges0, 'cpu'):
                    edges0 = edges0.cpu().numpy()
                elif hasattr(edges0, 'numpy'):
                    edges0 = edges0.numpy()
                edges0 = np.asarray(edges0)
                
                if edges0.shape[0] == 2 and edges0.shape[1] > 0:
                    frame_edges_created = 0
                    frame_edges_skipped = 0
                    
                    for edge_idx in range(edges0.shape[1]):
                        source_idx = int(edges0[0, edge_idx])
                        target_idx = int(edges0[1, edge_idx])
                        
                        if source_idx < len(transformed_frame_keypoints) and target_idx < len(transformed_frame_keypoints):
                            source_pos = transformed_frame_keypoints[source_idx]
                            target_pos = transformed_frame_keypoints[target_idx]
                            
                            # Calculate distance between nodes
                            distance = np.linalg.norm(target_pos - source_pos)
                            
                            # Filter by distance threshold
                            if distance <= edge_distance_threshold:
                                # Create a cylinder between source and target positions (light blue color)
                                direction = target_pos - source_pos
                                length = np.linalg.norm(direction)
                                if length > 0:
                                    direction = direction / length
                                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=edge_line_radius, height=length)
                                    cylinder.paint_uniform_color([0.5, 0.8, 1.0])  # Light blue color for frame edges
                                    
                                    center = (source_pos + target_pos) / 2
                                    cylinder.translate(center)
                                    
                                    z_axis = np.array([0, 0, 1])
                                    if not np.allclose(direction, z_axis):
                                        rotation_axis = np.cross(z_axis, direction)
                                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                        cos_angle = np.dot(z_axis, direction)
                                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                                        R = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                                        cylinder.rotate(R, center=center)
                                    
                                    all_edge_lines.append(cylinder)
                                    frame_edges_created += 1
                            else:
                                frame_edges_skipped += 1
                    
                    print(f"  Frame {frame_idx + 1} edges: {frame_edges_created} created, {frame_edges_skipped} skipped (threshold: {edge_distance_threshold:.3f}m)")
        
        print(f"Total edge lines created: {len(all_edge_lines)}")
    
    # Combine all geometries
    geometries = [map_cloud] + all_frame_clouds + all_lines + all_keypoint_spheres
    if map_background_cloud is not None:
        geometries.append(map_background_cloud)
    if show_edges and len(all_edge_lines) > 0:
        geometries.extend(all_edge_lines)
    if show_edges and len(all_edge_lines) > 0:
        geometries.extend(all_edge_lines)
    
    # Create coordinate frame (optional)
    if show_coordinate_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        geometries.append(coord_frame)
    
    # Create tracking_id_colors dictionary
    tracking_id_colors = {i: instance_colors[i] / 255.0 for i in all_unique_ids}
    tracking_id_colors[0] = gray_color
    
    print("\nColor assignment complete")
    print(f"Total unique tracking IDs: {len(all_unique_ids)}")
    
    # Visualize
    print(f"\nVisualizing {len(geometries)} geometries ({num_frames} frame(s))...")
    print("Legend:")
    print("  Colored point clouds: Map and frames (colored by tracking ID)")
    print("  Large red spheres: Matched keypoints")
    print("  Small blue spheres: Unmatched keypoints")
    if show_edges:
        print("  Light blue lines: Edges connecting nodes within 2 meters")
    if show_match_confidence:
        print("  Match lines: Green (high confidence) to Yellow (low confidence)")
    else:
        print("  Green cylinders: Successful predicted matches")
        print("  Red cylinders: Failed predicted matches")
    if gt_alignment_dict is not None:
        print("  Orange cylinders: False negatives (ground truth matches not predicted by model)")
    if show_coordinate_frame:
        print("  Coordinate frame: Reference (red=X, green=Y, blue=Z)")
    print(f"  Color mapping: {len(tracking_id_colors)} unique tracking IDs")
    
    # Create visualizer with key callback support
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Inference Results - {num_frames} Frame(s) and Map", 
                      width=1400, height=900, visible=True)
    
    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set up render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size * 1.5
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    # render_option.background_color = np.array([1.0, 0.97, 0.95])  # Lighter orange/peach background
    render_option.mesh_show_back_face = True
    
    # Set up view control with default view angles
    view_control = vis.get_view_control()
    
    # Calculate scene center if lookat not provided
    lookat_point = default_lookat
    if lookat_point is None:
        # Use the center of all geometries
        all_points = []
        for geom in geometries:
            if isinstance(geom, o3d.geometry.PointCloud):
                all_points.extend(np.asarray(geom.points))
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                all_points.extend(np.asarray(geom.vertices))
        if len(all_points) > 0:
            all_points = np.array(all_points)
            lookat_point = np.mean(all_points, axis=0)
        else:
            lookat_point = np.array([0.0, 0.0, 0.0])
    else:
        lookat_point = np.array(lookat_point)
    
    # Set default view parameters
    # Note: Order matters! Set lookat first, then front, then up
    view_control.set_lookat(lookat_point)
    
    if default_front is not None:
        front_vec = np.array(default_front, dtype=np.float64)
        # Normalize front vector
        front_norm = np.linalg.norm(front_vec)
        if front_norm > 1e-8:
            front_vec = front_vec / front_norm
        view_control.set_front(front_vec)
    
    if default_up is not None:
        up_vec = np.array(default_up, dtype=np.float64)
        # Normalize up vector
        up_norm = np.linalg.norm(up_vec)
        if up_norm > 1e-8:
            up_vec = up_vec / up_norm
        view_control.set_up(up_vec)
    
    if default_zoom is not None:
        view_control.set_zoom(default_zoom)
    
    # Update the view to apply changes
    vis.poll_events()
    vis.update_renderer()
    
    # Helper function to extract view parameters from camera parameters
    def extract_view_params(view_control):
        """Extract view parameters from ViewControl using camera parameters."""
        try:
            # Get camera parameters
            params = view_control.convert_to_pinhole_camera_parameters()
            extrinsic = params.extrinsic
            
            # Extract camera position (eye) from extrinsic matrix
            # Extrinsic is [R | t] where R is 3x3 rotation and t is 3x1 translation
            # Camera position in world: -R^T * t
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            eye = -R.T @ t
            
            # In Open3D's coordinate system:
            # - The rotation matrix R transforms from world to camera coordinates
            # - Camera's forward direction is -Z in camera frame
            # - Camera's up direction is +Y in camera frame
            
            # Extract front vector: direction camera is looking (negative Z axis in camera frame)
            # In world coordinates, this is -R^T * [0, 0, 1] = -R^T[:, 2]
            front = -R.T[:, 2]
            
            # Extract up vector: camera's up direction (Y axis in camera frame)
            # In world coordinates, this is R^T * [0, 1, 0] = R^T[:, 1]
            up = R.T[:, 1]
            
            # Normalize vectors
            front = front / (np.linalg.norm(front) + 1e-8)
            up = up / (np.linalg.norm(up) + 1e-8)
            
            # Ensure up vector is orthogonal to front (Gram-Schmidt)
            # Remove component of up that's parallel to front
            up = up - np.dot(up, front) * front
            up = up / (np.linalg.norm(up) + 1e-8)
            
            # Check if up vector points in expected direction (typically +Z or +Y)
            # If most of the up vector is in negative Z or Y, it might cause upside-down view
            # We'll check the Z component (assuming Z-up coordinate system)
            # If up[2] < 0, suggest negating it
            up_warning = ""
            if up[2] < -0.1:  # Significant negative Z component
                up_warning = " (up vector has negative Z - may cause upside-down view)"
            
            # Calculate lookat point: eye + front * some_distance
            # We'll use a reasonable default distance (e.g., 5 units) or calculate from scene
            # Actually, we can't get exact lookat distance from extrinsic alone
            # So we'll approximate: lookat = eye + front (normalized) * 1.0
            # But better: use the actual lookat if we can get it
            # For now, we'll calculate an approximate lookat
            lookat_approx = eye + front * 5.0  # Default distance
            
            # For zoom, we can't directly get it, but we can estimate from field of view
            fov = view_control.get_field_of_view()
            
            return front, eye, up, lookat_approx, fov
        except Exception as e:
            # Fallback: return None values if extraction fails
            print(f"Warning: Could not extract view parameters: {e}")
            return None, None, None, None, None
    
    # Print current view angles if requested
    if show_view_angles:
        # Store the values we set for display
        print("\n" + "="*60)
        print("View Parameters Set:")
        print("="*60)
        if default_front is not None:
            front_arr = np.array(default_front)
            print(f"Front vector: [{front_arr[0]:.4f}, {front_arr[1]:.4f}, {front_arr[2]:.4f}]")
        else:
            print("Front vector: [default]")
        lookat_arr = np.array(lookat_point)
        print(f"Lookat point: [{lookat_arr[0]:.4f}, {lookat_arr[1]:.4f}, {lookat_arr[2]:.4f}]")
        if default_up is not None:
            up_arr = np.array(default_up)
            print(f"Up vector:    [{up_arr[0]:.4f}, {up_arr[1]:.4f}, {up_arr[2]:.4f}]")
        else:
            print("Up vector:    [default: 0, 0, 1]")
        if default_zoom is not None:
            print(f"Zoom level:   {default_zoom:.4f}")
        else:
            print("Zoom level:   [default]")
        print("="*60)
        print("Tip: Adjust view in the window, then press 'P' to print current camera parameters")
        print("="*60 + "\n")
    
    # Add key callback to print view angles when 'P' is pressed
    def print_view_angles(vis):
        view_control = vis.get_view_control()
        result = extract_view_params(view_control)
        
        if result[0] is not None:
            front, eye, up, lookat_approx, fov = result
            
            print("\n" + "="*60)
            print("Current Camera Parameters (pressed 'P'):")
            print("="*60)
            print(f"Camera position (eye): [{eye[0]:.4f}, {eye[1]:.4f}, {eye[2]:.4f}]")
            print(f"Front vector:         [{front[0]:.4f}, {front[1]:.4f}, {front[2]:.4f}]")
            print(f"Up vector:            [{up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}]")
            print(f"Lookat (approx):      [{lookat_approx[0]:.4f}, {lookat_approx[1]:.4f}, {lookat_approx[2]:.4f}]")
            print(f"Field of view:        {fov:.4f}")
            
            # Check if up vector might cause upside-down view
            # Check both Y and Z components (depending on coordinate system)
            warning_printed = False
            if up[2] < -0.1:  # Negative Z (assuming Z-up)
                print("\n⚠️  WARNING: Up vector has negative Z component - view might appear upside down!")
                print("   Try negating the up vector if the restored view is inverted.")
                warning_printed = True
            elif up[1] < -0.1:  # Negative Y (assuming Y-up)
                print("\n⚠️  WARNING: Up vector has negative Y component - view might appear upside down!")
                print("   Try negating the up vector if the restored view is inverted.")
                warning_printed = True
            
            print("\nTo set this view, use:")
            print(f"  --view_front {front[0]:.4f} {front[1]:.4f} {front[2]:.4f} \\")
            print(f"  --view_up {up[0]:.4f} {up[1]:.4f} {up[2]:.4f} \\")
            print(f"  --view_lookat {lookat_approx[0]:.4f} {lookat_approx[1]:.4f} {lookat_approx[2]:.4f}")
            
            # Also provide alternative with negated up if there's a warning
            if warning_printed:
                up_negated = -up
                print("\nAlternative (if view is upside down, try negating up):")
                print(f"  --view_front {front[0]:.4f} {front[1]:.4f} {front[2]:.4f} \\")
                print(f"  --view_up {up_negated[0]:.4f} {up_negated[1]:.4f} {up_negated[2]:.4f} \\")
                print(f"  --view_lookat {lookat_approx[0]:.4f} {lookat_approx[1]:.4f} {lookat_approx[2]:.4f}")
            
            print("="*60 + "\n")
        else:
            print("\nCould not extract view parameters. Try adjusting the view and press 'P' again.\n")
        return False
    
    vis.register_key_callback(ord('P'), print_view_angles)
    
    # Run visualization
    vis.run()
    vis.destroy_window()
    
    total_matches = sum(len(np.where(r['predicted_matches0'] != -1)[0]) for r in results)
    print(f"Visualization complete. Total matches across all frames: {total_matches}")
    
    return tracking_id_colors


def visualize_map_with_nodes(map_ply_path, topology_map_path=None, topology_map=None, bias_meter=0.0, instance_colors=None, node_radius=0.1, show_bboxes=True, show_edges=True, hypothesis_id="default_hypothesis"):
    """
    Visualize the map PLY file with object node positions from topology map highlighted.
    
    Args:
        map_ply_path: Path to the map ply file
        topology_map_path: Optional path to topology map JSON file
        topology_map: Optional pre-loaded TopologyMap object
        bias_meter: Bias for map cloud in x-axis (default: 5.0)
        instance_colors: Optional pre-generated colors for tracking IDs
        node_radius: Radius of spheres representing nodes (default: 0.1)
        show_bboxes: Whether to show bounding boxes for object nodes (default: True)
        show_edges: Whether to show edges between nodes (default: True)
        hypothesis_id: ID of the hypothesis to visualize edges for (default: "default_hypothesis")
    
    Returns:
        tracking_id_colors: Dictionary mapping tracking IDs to colors
    """
    print("\nVisualizing map with object node positions...")
    
    # Load topology map if not provided
    if topology_map is None:
        if topology_map_path is None:
            print("Error: Either topology_map or topology_map_path must be provided")
            return None
        
        if not os.path.exists(topology_map_path):
            print(f"Warning: Topology map file not found: {topology_map_path}")
            return None
        
        print(f"Loading topology map from: {topology_map_path}")
        with open(topology_map_path, "r") as f:
            topology_map = TopologyMap()
            topology_map.read_from_json(f.read())
    
    # Extract object node positions and shapes
    object_node_positions = []
    object_node_info = []
    bbox_geometries = []
    
    for node_id, node in topology_map.object_nodes.nodes.items():
        # Convert Eigen::Vector3f to numpy array
        position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
        object_node_positions.append(position)
        object_node_info.append({
            'id': node_id,
            'name': node.name,
            'position': position,
            'shape': node.shape
        })
        
        # Create bounding box if shape exists and show_bboxes is True
        if show_bboxes and node.shape is not None:
            try:
                bbox = create_bbox_from_shape(node.shape, position, bias_meter)
                if bbox is not None:
                    bbox_geometries.append(bbox)
                    print(f"  Created bbox for {node_id} with dimensions: {get_shape_dimensions(node.shape)}")
            except Exception as e:
                print(f"  Warning: Could not create bbox for {node_id}: {e}")
    
    print(f"Found {len(object_node_positions)} object nodes in topology map")
    if show_bboxes:
        print(f"Created {len(bbox_geometries)} bounding boxes")
    
    # Create edge geometries if requested
    edge_geometries = []
    if show_edges:
        print("Creating edge visualizations...")
        
        # Create a map from node ID to position for edge lookup
        id_to_node_position_map = {}
        for node_id, node in topology_map.object_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            id_to_node_position_map[node_id] = position
        
        # Add free space nodes to the position map
        for node_id, node in topology_map.free_space_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            id_to_node_position_map[node_id] = position
        
        # Visualize edge hypotheses with the given hypothesis id
        if hypothesis_id in topology_map.edge_hypotheses:
            hypothesis = topology_map.edge_hypotheses[hypothesis_id]
            print(f"Visualizing edges for hypothesis: {hypothesis_id}")
            
            for edge_id, edge in hypothesis.edges.items():
                source_id = edge.source_id
                target_id = edge.target_id
                
                if source_id in id_to_node_position_map and target_id in id_to_node_position_map:
                    source_position = id_to_node_position_map[source_id]
                    target_position = id_to_node_position_map[target_id]
                    
                    # Create a line between source and target positions (green color)
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([source_position, target_position])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.paint_uniform_color([0.0, 1.0, 0.0])  # Green color for hypothesis edges
                    
                    edge_geometries.append(line)
                    print(f"  Created edge: {source_id} -> {target_id}")
                else:
                    print(f"  Warning: Could not find positions for edge {edge_id}: {source_id} -> {target_id}")
        else:
            print(f"Warning: Hypothesis '{hypothesis_id}' not found in edge_hypotheses")
            print(f"Available hypotheses: {list(topology_map.edge_hypotheses.keys())}")
    
    if len(object_node_positions) == 0:
        print("Warning: No object nodes found in topology map")
        return None
    
    # Convert to numpy array
    object_node_positions = np.array(object_node_positions)
    
    # Load point clouds
    if not os.path.exists(map_ply_path):
        print(f"Warning: Map ply file not found: {map_ply_path}")
        return None
    
    # Load map point cloud
    map_cloud = o3d.io.read_point_cloud(map_ply_path)
    print(f"Loaded map cloud with {len(map_cloud.points)} points")
    
    # Bias the map cloud by bias_meter meters in the x-axis
    map_cloud_points = np.asarray(map_cloud.points)
    map_cloud_points[:, 0] += bias_meter  # Add bias_meter meters to x-coordinate
    map_cloud.points = o3d.utility.Vector3dVector(map_cloud_points)
    
    # Assign random colors based on tracking IDs from R channel
    print("Assigning random colors based on tracking IDs...")
    
    # Get tracking IDs from R channel of point cloud
    map_colors = np.asarray(map_cloud.colors)
    
    # Extract tracking IDs from R channel (assuming R channel contains tracking IDs)
    map_tracking_ids = (map_colors[:, 0] * 255).astype(int)
    
    # Get unique tracking IDs
    unique_map_ids = np.unique(map_tracking_ids)
    print(f"Found {len(unique_map_ids)} unique tracking IDs in map")
    
    # Create a color map for tracking IDs
    if instance_colors is None:
        instance_colors = generate_instance_colors(0, 255)

    new_map_colors = instance_colors[map_tracking_ids] / 255.0
    tracking_id_colors = {i: instance_colors[i] / 255.0 for i in unique_map_ids}

    map_cloud.colors = o3d.utility.Vector3dVector(new_map_colors)
    
    print("Color assignment complete")
    
    # Print some tracking ID to color mappings for reference
    print("\nTracking ID to Color Mapping (first 10):")
    for i, (tracking_id, color) in enumerate(list(tracking_id_colors.items())[:10]):
        print(f"  Tracking ID {tracking_id}: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
    if len(tracking_id_colors) > 10:
        print(f"  ... and {len(tracking_id_colors) - 10} more tracking IDs")
    
    # Bias the node positions by bias_meter meters in x
    biased_node_positions = object_node_positions.copy()
    biased_node_positions[:, 0] += bias_meter
    
    # Create node spheres for visualization
    node_spheres = []
    print(f"Creating {len(biased_node_positions)} object node spheres...")
    
    for i, (pos, info) in enumerate(zip(biased_node_positions, object_node_info)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=node_radius)
        sphere.translate(pos)
        # Use a distinct color for nodes (yellow) to differentiate from tracking ID colors
        sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow for object nodes
        node_spheres.append(sphere)
        
        # Print node information
        print(f"  Object Node {info['id']} ('{info['name']}'): Position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # Create free space node spheres (blue color like C++)
    free_space_spheres = []
    if hasattr(topology_map, 'free_space_nodes') and topology_map.free_space_nodes:
        print(f"Creating {len(topology_map.free_space_nodes.nodes)} free space node spheres...")
        
        for node_id, node in topology_map.free_space_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger radius for free space nodes
            sphere.translate(position)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for free space nodes
            free_space_spheres.append(sphere)
            
            print(f"  Free Space Node {node_id}: Position ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    
    # Combine all geometries
    geometries = [map_cloud] + node_spheres + free_space_spheres
    if show_bboxes and bbox_geometries:
        geometries.extend(bbox_geometries)
    if show_edges and edge_geometries:
        geometries.extend(edge_geometries)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)
    
    # Visualize
    print(f"Visualizing {len(geometries)} geometries...")
    print("Legend:")
    print("  Random colors: Map point cloud (colored by tracking ID)")
    print("  Yellow spheres: Object node positions from topology map")
    print("  Blue spheres: Free space node positions")
    if show_bboxes and bbox_geometries:
        print("  Cyan wireframes: Object bounding boxes")
    if show_edges and edge_geometries:
        print("  Green lines: Hypothesis edges between nodes")
    print("  Coordinate frame: Reference")
    print(f"  Color mapping: {len(tracking_id_colors)} unique tracking IDs")
    print(f"  Map bias: {bias_meter} meters in x-axis")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Map Visualization with Object Node Positions",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_wireframe=False,  # We handle wireframe conversion separately
        mesh_show_back_face=True
    )
    
    print(f"Visualization complete. Displayed {len(biased_node_positions)} object nodes.")
    if hasattr(topology_map, 'free_space_nodes') and topology_map.free_space_nodes:
        print(f"Displayed {len(topology_map.free_space_nodes.nodes)} free space nodes.")
    if show_bboxes and bbox_geometries:
        print(f"Displayed {len(bbox_geometries)} bounding boxes.")
    if show_edges and edge_geometries:
        print(f"Displayed {len(edge_geometries)} edges.")
    
    return tracking_id_colors
    """
    Visualize the map PLY file with object node positions from topology map highlighted.
    
    Args:
        map_ply_path: Path to the map ply file
        topology_map_path: Optional path to topology map JSON file
        topology_map: Optional pre-loaded TopologyMap object
        bias_meter: Bias for map cloud in x-axis (default: 5.0)
        instance_colors: Optional pre-generated colors for tracking IDs
        node_radius: Radius of spheres representing nodes (default: 0.1)
        show_bboxes: Whether to show bounding boxes for object nodes (default: True)
        show_edges: Whether to show edges between nodes (default: True)
        hypothesis_id: ID of the hypothesis to visualize edges for (default: "default_hypothesis")
    
    Returns:
        tracking_id_colors: Dictionary mapping tracking IDs to colors
    """
    print("\nVisualizing map with object node positions...")
    
    # Load topology map if not provided
    if topology_map is None:
        if topology_map_path is None:
            print("Error: Either topology_map or topology_map_path must be provided")
            return None
        
        if not os.path.exists(topology_map_path):
            print(f"Warning: Topology map file not found: {topology_map_path}")
            return None
        
        print(f"Loading topology map from: {topology_map_path}")
        with open(topology_map_path, "r") as f:
            topology_map = TopologyMap()
            topology_map.read_from_json(f.read())
    
    # Extract object node positions and shapes
    object_node_positions = []
    object_node_info = []
    bbox_geometries = []
    
    for node_id, node in topology_map.object_nodes.nodes.items():
        # Convert Eigen::Vector3f to numpy array
        position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
        object_node_positions.append(position)
        object_node_info.append({
            'id': node_id,
            'name': node.name,
            'position': position,
            'shape': node.shape
        })
        
        # Create bounding box if shape exists and show_bboxes is True
        if show_bboxes and node.shape is not None:
            try:
                bbox = create_bbox_from_shape(node.shape, position, bias_meter)
                if bbox is not None:
                    bbox_geometries.append(bbox)
                    print(f"  Created bbox for {node_id} with dimensions: {get_shape_dimensions(node.shape)}")
            except Exception as e:
                print(f"  Warning: Could not create bbox for {node_id}: {e}")
    
    print(f"Found {len(object_node_positions)} object nodes in topology map")
    if show_bboxes:
        print(f"Created {len(bbox_geometries)} bounding boxes")
    
    # Create edge geometries if requested
    edge_geometries = []
    if show_edges:
        print("Creating edge visualizations...")
        
        # Create a map from node ID to position for edge lookup
        id_to_node_position_map = {}
        for node_id, node in topology_map.object_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            id_to_node_position_map[node_id] = position
        
        # Add free space nodes to the position map
        for node_id, node in topology_map.free_space_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            id_to_node_position_map[node_id] = position
        
        # Visualize edge hypotheses with the given hypothesis id
        if hypothesis_id in topology_map.edge_hypotheses:
            hypothesis = topology_map.edge_hypotheses[hypothesis_id]
            print(f"Visualizing edges for hypothesis: {hypothesis_id}")
            
            for edge_id, edge in hypothesis.edges.items():
                source_id = edge.source_id
                target_id = edge.target_id
                
                if source_id in id_to_node_position_map and target_id in id_to_node_position_map:
                    source_position = id_to_node_position_map[source_id]
                    target_position = id_to_node_position_map[target_id]
                    
                    # Create a line between source and target positions (green color)
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([source_position, target_position])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.paint_uniform_color([0.0, 1.0, 0.0])  # Green color for hypothesis edges
                    
                    edge_geometries.append(line)
                    print(f"  Created edge: {source_id} -> {target_id}")
                else:
                    print(f"  Warning: Could not find positions for edge {edge_id}: {source_id} -> {target_id}")
        else:
            print(f"Warning: Hypothesis '{hypothesis_id}' not found in edge_hypotheses")
            print(f"Available hypotheses: {list(topology_map.edge_hypotheses.keys())}")
    
    if len(object_node_positions) == 0:
        print("Warning: No object nodes found in topology map")
        return None
    
    # Convert to numpy array
    object_node_positions = np.array(object_node_positions)
    
    # Load point clouds
    if not os.path.exists(map_ply_path):
        print(f"Warning: Map ply file not found: {map_ply_path}")
        return None
    
    # Load map point cloud
    map_cloud = o3d.io.read_point_cloud(map_ply_path)
    print(f"Loaded map cloud with {len(map_cloud.points)} points")
    
    # Bias the map cloud by bias_meter meters in the x-axis
    map_cloud_points = np.asarray(map_cloud.points)
    map_cloud_points[:, 0] += bias_meter  # Add bias_meter meters to x-coordinate
    map_cloud.points = o3d.utility.Vector3dVector(map_cloud_points)
    
    # Assign random colors based on tracking IDs from R channel
    print("Assigning random colors based on tracking IDs...")
    
    # Get tracking IDs from R channel of point cloud
    map_colors = np.asarray(map_cloud.colors)
    
    # Extract tracking IDs from R channel (assuming R channel contains tracking IDs)
    map_tracking_ids = (map_colors[:, 0] * 255).astype(int)
    
    # Get unique tracking IDs
    unique_map_ids = np.unique(map_tracking_ids)
    print(f"Found {len(unique_map_ids)} unique tracking IDs in map")
    
    # Create a color map for tracking IDs
    if instance_colors is None:
        instance_colors = generate_instance_colors(0, 255)

    new_map_colors = instance_colors[map_tracking_ids] / 255.0
    tracking_id_colors = {i: instance_colors[i] / 255.0 for i in unique_map_ids}

    map_cloud.colors = o3d.utility.Vector3dVector(new_map_colors)
    
    print("Color assignment complete")
    
    # Print some tracking ID to color mappings for reference
    print("\nTracking ID to Color Mapping (first 10):")
    for i, (tracking_id, color) in enumerate(list(tracking_id_colors.items())[:10]):
        print(f"  Tracking ID {tracking_id}: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
    if len(tracking_id_colors) > 10:
        print(f"  ... and {len(tracking_id_colors) - 10} more tracking IDs")
    
    # Bias the node positions by bias_meter meters in x
    biased_node_positions = object_node_positions.copy()
    biased_node_positions[:, 0] += bias_meter
    
    # Create node spheres for visualization
    node_spheres = []
    print(f"Creating {len(biased_node_positions)} object node spheres...")
    
    for i, (pos, info) in enumerate(zip(biased_node_positions, object_node_info)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=node_radius)
        sphere.translate(pos)
        # Use a distinct color for nodes (yellow) to differentiate from tracking ID colors
        sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow for object nodes
        node_spheres.append(sphere)
        
        # Print node information
        print(f"  Object Node {info['id']} ('{info['name']}'): Position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # Create free space node spheres (blue color like C++)
    free_space_spheres = []
    if hasattr(topology_map, 'free_space_nodes') and topology_map.free_space_nodes:
        print(f"Creating {len(topology_map.free_space_nodes.nodes)} free space node spheres...")
        
        for node_id, node in topology_map.free_space_nodes.nodes.items():
            position = np.array([node.position[0], node.position[1], node.position[2]], dtype=np.float32)
            # Apply bias to position
            position[0] += bias_meter
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger radius for free space nodes
            sphere.translate(position)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for free space nodes
            free_space_spheres.append(sphere)
            
            print(f"  Free Space Node {node_id}: Position ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    
    # Combine all geometries
    geometries = [map_cloud] + node_spheres + free_space_spheres
    if show_bboxes and bbox_geometries:
        geometries.extend(bbox_geometries)
    if show_edges and edge_geometries:
        geometries.extend(edge_geometries)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)
    
    # Visualize
    print(f"Visualizing {len(geometries)} geometries...")
    print("Legend:")
    print("  Random colors: Map point cloud (colored by tracking ID)")
    print("  Yellow spheres: Object node positions from topology map")
    print("  Blue spheres: Free space node positions")
    if show_bboxes and bbox_geometries:
        print("  Cyan wireframes: Object bounding boxes")
    if show_edges and edge_geometries:
        print("  Green lines: Hypothesis edges between nodes")
    print("  Coordinate frame: Reference")
    print(f"  Color mapping: {len(tracking_id_colors)} unique tracking IDs")
    print(f"  Map bias: {bias_meter} meters in x-axis")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Map Visualization with Object Node Positions",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_wireframe=False,  # We handle wireframe conversion separately
        mesh_show_back_face=True
    )
    
    print(f"Visualization complete. Displayed {len(biased_node_positions)} object nodes.")
    if hasattr(topology_map, 'free_space_nodes') and topology_map.free_space_nodes:
        print(f"Displayed {len(topology_map.free_space_nodes.nodes)} free space nodes.")
    if show_bboxes and bbox_geometries:
        print(f"Displayed {len(bbox_geometries)} bounding boxes.")
    if show_edges and edge_geometries:
        print(f"Displayed {len(edge_geometries)} edges.")
    
    return tracking_id_colors


def create_wireframe_mesh(mesh, color=[0.0, 1.0, 1.0], line_width=2.0):
    """
    Convert a mesh to wireframe representation.
    
    Args:
        mesh: Open3D mesh object
        color: RGB color for the wireframe
        line_width: Width of the wireframe lines
    
    Returns:
        o3d.geometry.LineSet: Wireframe representation
    """
    try:
        # Extract vertices and faces
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Create line set from triangle edges
        lines = []
        for triangle in triangles:
            # Add edges of the triangle
            lines.append([triangle[0], triangle[1]])
            lines.append([triangle[1], triangle[2]])
            lines.append([triangle[2], triangle[0]])
        
        # Remove duplicate lines
        lines = np.array(lines)
        unique_lines = []
        for line in lines:
            # Sort line indices to ensure consistent ordering
            sorted_line = sorted(line)
            if sorted_line not in unique_lines:
                unique_lines.append(sorted_line)
        
        # Create LineSet
        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(vertices)
        wireframe.lines = o3d.utility.Vector2iVector(unique_lines)
        wireframe.paint_uniform_color(color)
        
        return wireframe
        
    except Exception as e:
        print(f"  Error creating wireframe: {e}")
        return None


def create_bbox_from_shape(shape, position, bias_meter=0.0):
    """
    Create a bounding box mesh from a shape object.
    Follows the C++ approach: create corners in local space, then transform.
    
    Args:
        shape: Shape object (Cylinder or OrientedBox)
        position: 3D position of the shape
        bias_meter: Bias to apply to x-coordinate
    
    Returns:
        o3d.geometry.LineSet: Wireframe representation of the bounding box
    """
    try:
        # Apply bias to position
        biased_position = position.copy()
        biased_position[0] += bias_meter
        
        # Check shape type by class name or type method
        shape_type = None
        if hasattr(shape, 'type'):
            shape_type = shape.type()
        elif hasattr(shape, '__class__'):
            shape_type = shape.__class__.__name__
        
        # Handle OrientedBox
        if shape_type == 'ORIENTED_BOX' or 'OrientedBox' in str(shape_type):
            # Get dimensions
            length = shape.length
            width = shape.width
            height = shape.height
            
            # Create box size vector (following C++ convention)
            box_size = np.array([length, width, height])
            
        # Handle Cylinder
        elif shape_type == 'CYLINDER' or 'Cylinder' in str(shape_type):
            # Get dimensions
            radius = shape.radius
            height = shape.height
            
            # Turn cylinder into oriented bounding box (following C++ approach)
            box_size = np.array([radius * 2, radius * 2, height])
            
        else:
            print(f"  Warning: Unknown shape type: {shape_type} (class: {type(shape)})")
            return None
        
        # Get rotation matrix from orientation (following C++ approach)
        rotation_matrix = None
        if hasattr(shape, 'orientation'):
            q = shape.orientation
            # Convert quaternion to rotation matrix using the same method as C++
            # The C++ code calls orientation.toRotationMatrix()
            # In Python, we need to construct the rotation matrix from quaternion
            w, x, y, z = q.w, q.x, q.y, q.z
            
            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
            
            # Convert to rotation matrix (following Eigen convention)
            rotation_matrix = np.array([
                [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
            ])
        
        # Create the 8 corners of the box in local coordinates (following C++ approach)
        half_size = 0.5 * box_size
        corners_local = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]]
        ])
        
        # Transform corners to world coordinates (following C++ approach)
        corners_world = []
        if rotation_matrix is not None:
            for corner in corners_local:
                # Apply rotation first, then translation
                rotated_corner = rotation_matrix @ corner
                world_corner = rotated_corner + biased_position
                corners_world.append(world_corner)
        else:
            # No rotation, just translation
            for corner in corners_local:
                world_corner = corner + biased_position
                corners_world.append(world_corner)
        
        corners_world = np.array(corners_world)
        
        # Define the 12 edges of the box (following C++ approach)
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # Bottom face
            [4,5], [5,6], [6,7], [7,4],  # Top face
            [0,4], [1,5], [2,6], [3,7]   # Vertical edges
        ]
        
        # Create LineSet for wireframe
        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(corners_world)
        wireframe.lines = o3d.utility.Vector2iVector(edges)
        wireframe.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan color for bboxes
        
        return wireframe
        
    except Exception as e:
        print(f"  Error creating bbox: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_shape_dimensions(shape):
    """
    Get dimensions string for a shape object.
    
    Args:
        shape: Shape object
        
    Returns:
        str: Dimensions string
    """
    try:
        # Check shape type by class name or type method
        shape_type = None
        if hasattr(shape, 'type'):
            shape_type = shape.type()
        elif hasattr(shape, '__class__'):
            shape_type = shape.__class__.__name__
        
        if 'OrientedBox' in str(shape_type):
            return f"L:{shape.length:.3f} x W:{shape.width:.3f} x H:{shape.height:.3f}"
        elif 'Cylinder' in str(shape_type):
            return f"R:{shape.radius:.3f} x H:{shape.height:.3f}"
        else:
            return f"Unknown shape type: {shape_type}"
    except Exception as e:
        return f"Error: {e}"



