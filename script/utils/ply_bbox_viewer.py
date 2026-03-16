import open3d as o3d
import numpy as np
import json
import argparse
from recolor_ply_with_id import assign_random_colors_to_instances

def draw_bbox(center, size, rotation, color=[1, 0, 0]):
    # Generate box corners in local frame
    dx, dy, dz = size
    local_corners = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2]
    ])
    
    # Rotate and translate
    R = np.array(rotation)
    transformed_corners = (R @ local_corners.T).T + np.array(center)

    # Define lines between corners
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7]   # verticals
    ]
    
    # Create LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(transformed_corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

def main(ply_path, json_path, recolor=False):
    # Load point cloud
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Successfully loaded point cloud with {len(pcd.points)} points")
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        print("Trying alternative loading methods...")
        
        # Try with explicit format specification
        try:
            pcd = o3d.io.read_point_cloud(ply_path, format='ply')
            print(f"Successfully loaded with explicit PLY format")
        except Exception as e2:
            print(f"Failed to load with explicit format: {e2}")
            return

    # Load JSON with instance info
    try:
        with open(json_path, 'r') as f:
            instances = json.load(f)
        print(f"Loaded {len(instances)} instances from JSON")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Create a list of geometries to visualize
    if recolor:
        pcd = assign_random_colors_to_instances(pcd, bkg_black=True)
    
    geometries = [pcd]
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]

    for i, inst in enumerate(instances):
        size = [inst['bbox_size']['x'], inst['bbox_size']['y'], inst['bbox_size']['z']]
        center = [inst['center']['x'], inst['center']['y'], inst['center']['z']]
        rotation = inst['rotation']
        color = colors[i % len(colors)]
        bbox = draw_bbox(center, size, rotation, color)
        geometries.append(bbox)
        

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to the instance point cloud .ply file")
    parser.add_argument("--json", type=str, required=True, help="Path to the bounding box JSON file")
    parser.add_argument("--recolor", action="store_true", help="Recolor the point cloud with random colors")
    args = parser.parse_args()
    main(args.ply, args.json, args.recolor)
