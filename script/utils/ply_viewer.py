import open3d as o3d
import argparse
import numpy as np
import sys
import os

def read_transformation(file_path):
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"PLY file '{file_path}' is empty or invalid.")
    return pcd

def load_and_color_ply(file_path, color):
    pcd = load_ply(file_path)
    pcd.paint_uniform_color(color)
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Visualize one or two PLY files with optional transformation.")
    parser.add_argument("ply1", type=str, help="Path to the first PLY file")
    parser.add_argument("ply2", type=str, nargs="?", help="Optional second PLY file")
    parser.add_argument("--transform", type=str, help="Optional 4x4 transformation matrix (.txt or .npy) for second PLY")

    args = parser.parse_args()

    if not os.path.isfile(args.ply1):
        print(f"Error: File '{args.ply1}' does not exist.")
        sys.exit(1)

    try:
        pcd1 = load_ply(args.ply1)
    except Exception as e:
        print(e)
        sys.exit(1)

    if args.ply2 is None:
        # Only one PLY, show with original color
        o3d.visualization.draw_geometries([pcd1])
        return

    if not os.path.isfile(args.ply2):
        print(f"Error: File '{args.ply2}' does not exist.")
        sys.exit(1)

    try:
        pcd1.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        pcd2 = load_ply(args.ply2)
        pcd2.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    except Exception as e:
        print(e)
        sys.exit(1)

    if args.transform:
        if not os.path.isfile(args.transform):
            print(f"Error: Transformation file '{args.transform}' does not exist.")
            sys.exit(1)
        try:
            transformation = read_transformation(args.transform)
            if transformation.shape != (4, 4):
                raise ValueError("Transformation matrix must be 4x4.")
            pcd2.transform(transformation)
        except Exception as e:
            print(f"Error reading transformation: {e}")
            sys.exit(1)

    o3d.visualization.draw_geometries([pcd1, pcd2])

if __name__ == "__main__":
    main()
