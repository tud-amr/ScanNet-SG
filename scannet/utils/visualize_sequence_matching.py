import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "..", "..", "script"))


def visualize_sequence_matching(data_entry, save_path=None, show_labels=False, 
                                 map_folder=None, bias_meters=10.0):
    """
    Visualize sequence matching data showing nodes from sequence and first scan with their matches.
    
    Args:
        data_entry: Dictionary containing keypoints0, keypoints1, matches0, and optional metadata
        save_path: Optional path to save the figure
        show_labels: Whether to show text labels for nodes
        map_folder: Path to the map folder containing transformation.npy files
        bias_meters: Bias offset in meters for visualization
    """
    keypoints0 = data_entry['keypoints0']  # Sequence nodes (from more_scans)
    keypoints1 = data_entry['keypoints1']  # First scan nodes (target)
    matches0 = data_entry['matches0']  # Matching indices
    
    # Get metadata if available
    num_frames = data_entry.get('num_frames', 'N/A')
    sequence_id = data_entry.get('sequence_id', 'N/A')
    scene_graph_id = data_entry.get('scene_graph_id', 'N/A')
    frame_scene = data_entry.get('frame_scene', 'N/A')
    
    # Load transformation and align sequence nodes
    keypoints0_aligned = keypoints0.copy()
    if map_folder and scene_graph_id:
        transformation_path = os.path.join(map_folder, scene_graph_id, "transformation.npy")
        if os.path.exists(transformation_path):
            # Load transformation (this transforms from first_scan to more_scans)
            transformation = np.load(transformation_path)
            # We need inverse to transform from more_scans to first_scan
            inv_transformation = np.linalg.inv(transformation)
            
            # Convert keypoints0 to homogeneous coordinates
            keypoints0_homogeneous = np.hstack([keypoints0, np.ones((keypoints0.shape[0], 1))])
            
            # Apply inverse transformation
            keypoints0_transformed = (inv_transformation @ keypoints0_homogeneous.T).T
            keypoints0_aligned = keypoints0_transformed[:, :3]
            
            print(f"Loaded transformation from {transformation_path}")
            print(f"Applied inverse transformation to align sequence nodes")
        else:
            print(f"Warning: Transformation file not found at {transformation_path}, skipping alignment")
    
    # Estimate scene scale for bias offset (assuming normalized coordinates represent ~30m scene)
    # Convert 10 meters to normalized coordinates
    scene_size_meters = 30.0  # Default assumption
    bias_normalized = bias_meters / scene_size_meters
    
    # Apply bias offset to aligned sequence nodes (in X direction for visibility)
    keypoints0_aligned_offset = keypoints0_aligned.copy()
    keypoints0_aligned_offset[:, 0] += bias_normalized
    
    # Get text embeddings if available for labels
    text_embedding0 = data_entry.get('text_embedding0', None)
    text_embedding1 = data_entry.get('text_embedding1', None)
    
    # Find valid matches
    valid_match_indices = np.where(matches0 != -1)[0]
    num_valid_matches = len(valid_match_indices)
    
    # Create visualization with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Sequence nodes (keypoints0)
    ax1 = fig.add_subplot(131, projection='3d')
    matched_mask = np.zeros(len(keypoints0), dtype=bool)
    matched_mask[valid_match_indices] = True
    
    # Plot matched nodes in green, unmatched in blue
    if np.any(matched_mask):
        ax1.scatter(keypoints0[matched_mask, 0], keypoints0[matched_mask, 1], keypoints0[matched_mask, 2],
                   c='green', s=100, alpha=0.7, label=f'Matched ({np.sum(matched_mask)})', marker='o')
    if np.any(~matched_mask):
        ax1.scatter(keypoints0[~matched_mask, 0], keypoints0[~matched_mask, 1], keypoints0[~matched_mask, 2],
                   c='blue', s=50, alpha=0.5, label=f'Unmatched ({np.sum(~matched_mask)})', marker='x')
    
    ax1.set_title(f'Sequence Nodes (from {scene_graph_id})\n{num_frames} frames, {len(keypoints0)} nodes')
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')
    ax1.set_zlabel('Z (normalized)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First scan nodes (keypoints1)
    ax2 = fig.add_subplot(132, projection='3d')
    matched_map_indices = set(matches0[valid_match_indices])
    matched_map_mask = np.array([i in matched_map_indices for i in range(len(keypoints1))])
    
    # Plot matched nodes in green, unmatched in red
    if np.any(matched_map_mask):
        ax2.scatter(keypoints1[matched_map_mask, 0], keypoints1[matched_map_mask, 1], keypoints1[matched_map_mask, 2],
                   c='green', s=100, alpha=0.7, label=f'Matched ({np.sum(matched_map_mask)})', marker='o')
    if np.any(~matched_map_mask):
        ax2.scatter(keypoints1[~matched_map_mask, 0], keypoints1[~matched_map_mask, 1], keypoints1[~matched_map_mask, 2],
                   c='red', s=50, alpha=0.5, label=f'Unmatched ({np.sum(~matched_map_mask)})', marker='x')
    
    ax2.set_title(f'First Scan Nodes (from {frame_scene})\n{len(keypoints1)} nodes')
    ax2.set_xlabel('X (normalized)')
    ax2.set_ylabel('Y (normalized)')
    ax2.set_zlabel('Z (normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined view with match lines (aligned and offset)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot aligned sequence nodes with bias offset
    ax3.scatter(keypoints0_aligned_offset[:, 0], keypoints0_aligned_offset[:, 1], keypoints0_aligned_offset[:, 2],
               c='blue', s=60, alpha=0.6, label=f'Sequence Aligned ({len(keypoints0)})', marker='o')
    
    # Plot first scan nodes (no offset - they stay in original position)
    ax3.scatter(keypoints1[:, 0], keypoints1[:, 1], keypoints1[:, 2],
               c='red', s=60, alpha=0.6, label=f'First Scan ({len(keypoints1)})', marker='^')
    
    # Draw lines for valid matches (from aligned+offset sequence to first scan)
    for seq_idx in valid_match_indices:
        map_idx = matches0[seq_idx]
        if map_idx < len(keypoints1):
            seq_pos = keypoints0_aligned_offset[seq_idx]
            map_pos = keypoints1[map_idx]
            ax3.plot([seq_pos[0], map_pos[0]], 
                    [seq_pos[1], map_pos[1]], 
                    [seq_pos[2], map_pos[2]], 
                    'g-', alpha=0.5, linewidth=1.5)
    
    ax3.set_title(f'Aligned Matches Visualization\n{num_valid_matches} valid matches, {bias_meters}m offset (Sequence ID: {sequence_id})')
    ax3.set_xlabel('X (normalized)')
    ax3.set_ylabel('Y (normalized)')
    ax3.set_zlabel('Z (normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    print(f"\nVisualization Summary:")
    print(f"  Sequence nodes: {len(keypoints0)}")
    print(f"  First scan nodes: {len(keypoints1)}")
    print(f"  Valid matches: {num_valid_matches}")
    print(f"  Match rate: {num_valid_matches/len(keypoints0)*100:.1f}%")
    print(f"  Sequence ID: {sequence_id}")
    print(f"  Number of frames: {num_frames}")


def visualize_multiple_sequences(data_list, indices=None, save_dir=None, map_folder=None, bias_meters=10.0):
    """
    Visualize multiple sequence matching entries.
    
    Args:
        data_list: List of data dictionaries
        indices: List of indices to visualize (if None, visualize all)
        save_dir: Directory to save visualizations (if None, display interactively)
        map_folder: Path to the map folder containing transformation.npy files
        bias_meters: Bias offset in meters for visualization
    """
    if indices is None:
        indices = list(range(len(data_list)))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for idx in indices:
        if idx >= len(data_list):
            print(f"Warning: Index {idx} out of range (max: {len(data_list)-1})")
            continue
        
        print(f"\nVisualizing entry {idx}/{len(data_list)-1}")
        data_entry = data_list[idx]
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"sequence_matching_{idx:04d}.png")
        
        visualize_sequence_matching(data_entry, save_path=save_path, 
                                   map_folder=map_folder, bias_meters=bias_meters)
        
        if save_dir is None:
            # Interactive mode - wait for user input
            input("Press Enter to continue to next visualization...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sequence matching data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the pickle file containing sequence matching data")
    parser.add_argument("--index", type=int, default=None,
                       help="Index of the data entry to visualize (if None, visualize all)")
    parser.add_argument("--indices", type=str, default=None,
                       help="Comma-separated list of indices to visualize (e.g., '0,5,10')")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save visualizations (if None, display interactively)")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save a single visualization (overrides save_dir)")
    parser.add_argument("--show_labels", action="store_true",
                       help="Show text labels for nodes (if available)")
    parser.add_argument("--map_folder", type=str, default="/media/cc/Expansion/scannet/processed/scans",
                       help="Path to the map folder containing transformation.npy files")
    parser.add_argument("--bias_meters", type=float, default=10.0,
                       help="Bias offset in meters for visualization")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, "rb") as f:
        data_list = pickle.load(f)
    
    print(f"Loaded {len(data_list)} data entries")
    
    # Determine which indices to visualize
    if args.index is not None:
        indices = [args.index]
    elif args.indices is not None:
        indices = [int(i.strip()) for i in args.indices.split(',')]
    else:
        indices = None  # Visualize all
    
    # Visualize
    if args.save_path and args.index is not None:
        # Single visualization with specific save path
        visualize_sequence_matching(data_list[args.index], save_path=args.save_path, 
                                   show_labels=args.show_labels,
                                   map_folder=args.map_folder, bias_meters=args.bias_meters)
    else:
        # Multiple visualizations
        visualize_multiple_sequences(data_list, indices=indices, save_dir=args.save_dir,
                                    map_folder=args.map_folder, bias_meters=args.bias_meters)

