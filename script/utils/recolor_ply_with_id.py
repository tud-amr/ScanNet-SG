import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def assign_random_colors_to_instances(pcd: o3d.geometry.PointCloud, bkg_black=False) -> o3d.geometry.PointCloud:
    """
    Given a point cloud where r = g = b = instance_id / 255.0,
    assign a random but nice color for each instance id.
    
    Returns a new point cloud with updated colors.
    """
    # Extract instance IDs from grayscale RGB values
    colors = np.asarray(pcd.colors)
    instance_ids = (colors[:, 0] * 255).astype(np.int32)  # Assume r = g = b

    unique_ids = np.unique(instance_ids)

    # Generate visually distinct colors using a color map
    cmap = plt.get_cmap("tab20")  # up to 20 distinct colors
    base_colors = cmap(np.linspace(0, 1, len(unique_ids)))[:, :3]

    # Shuffle for randomness
    np.random.seed(42)  # fixed seed for reproducibility
    np.random.shuffle(base_colors)

    # Map instance id to color
    id_to_color = {id_: base_colors[i % len(base_colors)] for i, id_ in enumerate(unique_ids)}
    if bkg_black:
        id_to_color[0] = [0, 0, 0]
    # Apply new colors
    new_colors = np.array([id_to_color[id_] for id_ in instance_ids])
    
    # Create a copy with updated colors
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = pcd.points
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return new_pcd
