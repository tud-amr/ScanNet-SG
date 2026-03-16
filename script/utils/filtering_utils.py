import numpy as np
import open3d as o3d

def filter_matches_ransac_numpy(frame_positions, map_positions, predicted_match_dict,
                                distance_threshold=0.3,
                                ransac_n=3,
                                max_iters=1000):
    """
    NumPy-only RANSAC for filtering incorrect frame→map node matches.
    
    Args:
        frame_positions: dict {frame_id: np.array([x,y,z])}
        map_positions:   dict {map_id:   np.array([x,y,z])}
        predicted_match_dict: {frame_id: predicted_map_id}
        distance_threshold: inlier threshold (meters)
        ransac_n: minimal set size
        max_iters: RANSAC iterations

    Returns:
        filtered_matches: dict {frame_id: map_id}
        best_T: 4×4 rigid transformation matrix (numpy)
    """

    # ------------------------------------------------------------
    # Build correspondence arrays
    # ------------------------------------------------------------
    frame_ids = []
    map_ids = []
    src_pts = []
    tgt_pts = []
    
    for fid, mid in predicted_match_dict.items():
        if fid in frame_positions and mid in map_positions:
            frame_ids.append(fid)
            map_ids.append(mid)
            src_pts.append(frame_positions[fid])
            tgt_pts.append(map_positions[mid])
    
    src_pts = np.array(src_pts, dtype=float)  # Nx3
    tgt_pts = np.array(tgt_pts, dtype=float)  # Nx3
    N = len(src_pts)

    if N < ransac_n:
        return {}, np.eye(4)

    # ------------------------------------------------------------
    # Helper: solve rigid transform using SVD (Umeyama)
    # ------------------------------------------------------------
    def estimate_transform(A, B):
        """
        Solve R, t that aligns A → B using SVD.
        A, B are Nx3.
        """
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation (det=+1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A

        # 4×4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    # ------------------------------------------------------------
    # RANSAC main loop
    # ------------------------------------------------------------
    best_inliers = []
    best_T = np.eye(4)

    for _ in range(max_iters):
        # Random minimal subset
        idx = np.random.choice(N, ransac_n, replace=False)

        A = src_pts[idx]
        B = tgt_pts[idx]

        # Estimate transform
        T = estimate_transform(A, B)

        # Apply transform to all src points
        src_transformed = (src_pts @ T[:3, :3].T) + T[:3, 3]

        # Compute distances
        dists = np.linalg.norm(src_transformed - tgt_pts, axis=1)

        # Inliers
        inliers = np.where(dists < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_T = T

    # ------------------------------------------------------------
    # Build filtered match dict
    # ------------------------------------------------------------
    filtered_matches = {
        frame_ids[i]: map_ids[i]
        for i in best_inliers
    }

    return filtered_matches, best_T



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
        nb_neighbors: number of neighbors for statistical outlier removal
        std_ratio: standard deviation ratio threshold for outlier removal
        eps: DBSCAN clustering distance threshold
        min_points: minimum points per cluster
    
    Returns:
        Filtered point cloud as numpy array (M, 3) where M <= N
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
        # No clusters found or all points are noise, return SOR filtered points
        return np.asarray(pcd_filtered.points)
    
    # Find the largest cluster (excluding noise points with label -1)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        # All points are noise, return SOR filtered points
        return np.asarray(pcd_filtered.points)
    
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # Keep only points from the largest cluster
    mask = labels == largest_cluster_label
    filtered_pts = np.asarray(pcd_filtered.points)[mask]
    
    return filtered_pts
