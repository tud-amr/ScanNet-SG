#!/usr/bin/env python3
"""
Script to read .pgm depth image and info.txt calibration file,
generate point cloud, visualize it, and print depth statistics.
"""

import numpy as np
import cv2
import open3d as o3d
import re
import argparse
import os


def parse_info_txt(info_path):
    """Parse info.txt file and extract calibration parameters."""
    params = {}
    
    with open(info_path, 'r') as f:
        content = f.read()
    
    # Extract depth shift
    match = re.search(r'm_depthShift\s*=\s*(\d+)', content)
    if match:
        params['depth_shift'] = float(match.group(1))
    else:
        raise ValueError("Could not find m_depthShift in info.txt")
    
    # Extract depth intrinsics (4x4 matrix)
    match = re.search(r'm_calibrationDepthIntrinsic\s*=\s*([\d\.\s\-]+)', content)
    if match:
        intrinsic_str = match.group(1)
        intrinsic_values = [float(x) for x in intrinsic_str.split()]
        # Convert to 4x4 matrix, then extract 3x3 upper-left part
        intrinsic_4x4 = np.array(intrinsic_values).reshape(4, 4)
        params['intrinsic'] = intrinsic_4x4[:3, :3]
    else:
        raise ValueError("Could not find m_calibrationDepthIntrinsic in info.txt")
    
    # Extract depth dimensions (optional, for validation)
    match = re.search(r'm_depthWidth\s*=\s*(\d+)', content)
    if match:
        params['depth_width'] = int(match.group(1))
    
    match = re.search(r'm_depthHeight\s*=\s*(\d+)', content)
    if match:
        params['depth_height'] = int(match.group(1))
    
    return params


def read_pgm_depth(pgm_path):
    """Read .pgm depth image."""
    # Read as 16-bit unsigned integer
    depth = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth image: {pgm_path}")
    
    # PGM files are typically 16-bit
    if depth.dtype != np.uint16:
        print(f"Warning: Expected uint16, got {depth.dtype}. Converting...")
        depth = depth.astype(np.uint16)
    
    return depth


def depth_to_pointcloud(depth, intrinsic, depth_shift):
    """Convert depth image to point cloud."""
    height, width = depth.shape
    
    # Extract intrinsic parameters
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert depth to meters
    z = depth.astype(np.float32) / depth_shift
    
    # Convert to 3D points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    
    # Reshape to (N, 3)
    points = points.reshape(-1, 3)
    
    # Filter out invalid points (zero depth, inf, nan)
    valid_mask = (z.flatten() > 0) & np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    return points, z.flatten()[valid_mask]


def print_depth_statistics(depth_values):
    """Print depth statistics excluding inf and nan."""
    # Count inf and nan
    num_inf = np.isinf(depth_values).sum()
    num_nan = np.isnan(depth_values).sum()
    
    # Filter valid values
    valid_depths = depth_values[np.isfinite(depth_values)]
    
    if len(valid_depths) > 0:
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        mean_depth = np.mean(valid_depths)
        
        print("\n" + "="*50)
        print("Depth Statistics")
        print("="*50)
        print(f"Min depth (m):     {min_depth:.4f}")
        print(f"Max depth (m):     {max_depth:.4f}")
        print(f"Mean depth (m):    {mean_depth:.4f}")
        print(f"Number of inf:     {num_inf}")
        print(f"Number of nan:     {num_nan}")
        print(f"Valid points:      {len(valid_depths)}")
        print(f"Total points:      {len(depth_values)}")
        print("="*50)
    else:
        print("Warning: No valid depth values found!")


def visualize_pointcloud(points):
    """Visualize point cloud using Open3D."""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color points by depth (z-coordinate)
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-10)
    
    # Use colormap (blue to red)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized  # Red
    colors[:, 2] = 1.0 - z_normalized  # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("\nVisualizing point cloud...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Shift + Mouse: Pan view")
    print("  - Mouse wheel: Zoom")
    print("  - Q or close window: Exit")
    
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Depth Image Point Cloud",
                                      width=1280,
                                      height=720)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Read .pgm depth image and info.txt calibration file, "
                    "generate point cloud, visualize it, and print depth statistics."
    )
    parser.add_argument(
        "depth_image",
        type=str,
        help="Path to the .pgm depth image file"
    )
    parser.add_argument(
        "info_file",
        type=str,
        help="Path to the info.txt calibration file"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization (only print statistics)"
    )
    
    args = parser.parse_args()
    
    pgm_path = args.depth_image
    info_path = args.info_file
    
    # Check if files exist
    if not os.path.exists(pgm_path):
        parser.error(f"Depth image not found: {pgm_path}")
    
    if not os.path.exists(info_path):
        parser.error(f"Info file not found: {info_path}")
    
    print(f"Reading depth image: {pgm_path}")
    print(f"Reading calibration info: {info_path}")
    
    # Parse calibration parameters
    params = parse_info_txt(info_path)
    print(f"\nCalibration parameters:")
    print(f"  Depth shift: {params['depth_shift']}")
    print(f"  Intrinsic matrix:\n{params['intrinsic']}")
    
    # Read depth image
    depth = read_pgm_depth(pgm_path)
    print(f"\nDepth image shape: {depth.shape}")
    print(f"Depth image dtype: {depth.dtype}")
    
    # Convert to point cloud
    print("\nConverting depth to point cloud...")
    points, depth_values = depth_to_pointcloud(
        depth, 
        params['intrinsic'], 
        params['depth_shift']
    )
    
    print(f"Generated {len(points)} points")
    
    # Print statistics
    print_depth_statistics(depth_values)
    
    # Visualize
    if not args.no_visualize:
        visualize_pointcloud(points)
    else:
        print("\nVisualization skipped (--no-visualize flag set)")


if __name__ == "__main__":
    main()
