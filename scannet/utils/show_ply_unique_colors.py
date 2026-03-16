import argparse
import numpy as np
import open3d as o3d


def read_unique_colors(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_colors():
        raise ValueError("PLY file has no color information")

    # Colors are floats in [0, 1]
    colors = np.asarray(pcd.colors)

    # Convert to uint8 [0,255] for clean uniqueness
    colors_u8 = (colors * 255).round().astype(np.uint8)

    unique_colors, counts = np.unique(colors_u8, axis=0, return_counts=True)

    return unique_colors, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Input PLY file")
    args = parser.parse_args()

    colors, counts = read_unique_colors(args.ply)

    print(f"Found {len(colors)} unique colors:\n")
    for c, count in zip(colors, counts):
        print(f"R={c[0]:3d}, G={c[1]:3d}, B={c[2]:3d} - {count:,} points")


if __name__ == "__main__":
    main()
