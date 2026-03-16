import os
import sys
import argparse

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))

from utils.result_visualization import visualize_map_with_nodes



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--map_ply_path", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans/gpt4/openset_scans/scene0000_00/instance_cloud.ply")
    # parser.add_argument("--topology_map_path", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans/gpt4/openset_scans/scene0000_00/topology_map.json")
    parser.add_argument("--map_ply_path", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans/scene0000_00/instance_cloud.ply")
    parser.add_argument("--topology_map_path", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans/scene0000_00/topology_map.json")
    parser.add_argument("--node_radius", type=float, default=0.1)
    parser.add_argument("--show_bboxes", action="store_true")
    parser.add_argument("--show_edges", action="store_true")
    args = parser.parse_args()

    example_map_ply_path = args.map_ply_path
    example_topology_map_path = args.topology_map_path

    # Check if example files exist before running
    if os.path.exists(example_map_ply_path) and os.path.exists(example_topology_map_path):
        try:
            tracking_colors = visualize_map_with_nodes(
                map_ply_path=example_map_ply_path,
                topology_map_path=example_topology_map_path,
                node_radius=args.node_radius,
                show_bboxes=args.show_bboxes,
                show_edges=args.show_edges,
            )
            print(f"Successfully visualized map with {len(tracking_colors) if tracking_colors else 0} tracking IDs")
        except Exception as e:
            print(f"Error during map visualization: {e}")
    else:
        print("Example files not found. Please modify the paths above to match your actual files.")
        print(f"Map PLY path: {example_map_ply_path}")
        print(f"Topology map path: {example_topology_map_path}")
    
    print("\nVisualization example complete!")
    print("Modify the file paths above to use with your actual data.")