import os
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path


def find_all_pkl_files(root_dir, exclude_dynamic=True):
    """
    Find all .pkl files in root_dir and its subdirectories.
    
    Args:
        root_dir: Root directory to search
        exclude_dynamic: If True, exclude files with 'dynamic' in their path/name (default: True)
        
    Returns:
        List of paths to .pkl files
    """
    pkl_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                # Exclude files with 'dynamic' in path if exclude_dynamic is True
                if exclude_dynamic and 'dynamic' in file_path.lower():
                    continue
                pkl_files.append(file_path)
    return sorted(pkl_files)


def has_many_to_one_matching(matches0):
    """
    Check if matches0 has many-to-one situation (two or more elements have the same value, except -1).
    
    Args:
        matches0: numpy array of matches
        
    Returns:
        True if many-to-one exists, False otherwise
    """
    # Filter out -1 values
    valid_matches = matches0[matches0 != -1]
    
    if len(valid_matches) == 0:
        return False
    
    # Count occurrences of each value
    unique, counts = np.unique(valid_matches, return_counts=True)
    
    # Check if any value appears more than once
    return np.any(counts > 1)


def analyze_pkl_data(pkl_folder, output_dir=None, exclude_dynamic=True):
    """
    Analyze pkl data files and generate statistics.
    
    Args:
        pkl_folder: Folder containing pkl files (searches recursively)
        output_dir: Output directory for CSV files (defaults to pkl_folder)
        exclude_dynamic: If True, exclude files with 'dynamic' in their path/name (default: True)
    """
    if output_dir is None:
        output_dir = pkl_folder
    
    # Find all pkl files
    pkl_files = find_all_pkl_files(pkl_folder, exclude_dynamic=exclude_dynamic)
    print(f"Found {len(pkl_files)} pkl files")
    
    # Counters for keypoint statistics
    keypoints0_counts = Counter()
    keypoints1_counts = Counter()
    many_to_one_count = 0
    total_frames = 0
    
    # Track matching rates for each frame
    matching_rates = []
    
    # Track seen scene_graph_ids for keypoints1 counting
    seen_scene_graph_ids = set()
    
    # Track total node number and many-to-one node number
    # Count total matched nodes across all frames (not unique)
    total_node_number = 0  # Total count of matched nodes in all frames
    many_to_one_node_number = 0  # Total count of nodes with many-to-one matching in all frames
    
    # Process each pkl file
    for pkl_file in pkl_files:
        print(f"Processing: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data_list = pickle.load(f)
            
            if not isinstance(data_list, list):
                print(f"Warning: {pkl_file} does not contain a list, skipping")
                continue
            
            # Process each data entry in the list
            for data in data_list:
                if not isinstance(data, dict):
                    print(f"Warning: Entry is not a dictionary, skipping")
                    continue
                
                # Count keypoints0
                if 'keypoints0' in data:
                    keypoints0 = data['keypoints0']
                    if isinstance(keypoints0, np.ndarray):
                        num_kp0 = len(keypoints0)
                    elif isinstance(keypoints0, list):
                        num_kp0 = len(keypoints0)
                    else:
                        print(f"Warning: keypoints0 has unexpected type: {type(keypoints0)}")
                        continue
                    keypoints0_counts[num_kp0] += 1
                else:
                    print(f"Warning: 'keypoints0' not found in data entry")
                    continue
                
                # Count keypoints1 only when a different scene_graph_id is provided
                if 'keypoints1' in data:
                    # Check if scene_graph_id exists and is different from previously seen ones
                    scene_graph_id = data.get('scene_graph_id', None)
                    
                    # Only count if scene_graph_id is provided and we haven't seen this one before
                    if scene_graph_id is not None:
                        if scene_graph_id not in seen_scene_graph_ids:
                            keypoints1 = data['keypoints1']
                            if isinstance(keypoints1, np.ndarray):
                                num_kp1 = len(keypoints1)
                            elif isinstance(keypoints1, list):
                                num_kp1 = len(keypoints1)
                            else:
                                print(f"Warning: keypoints1 has unexpected type: {type(keypoints1)}")
                                seen_scene_graph_ids.add(scene_graph_id)
                                continue
                            keypoints1_counts[num_kp1] += 1
                            seen_scene_graph_ids.add(scene_graph_id)
                    # If scene_graph_id is not provided, skip counting keypoints1
                else:
                    print(f"Warning: 'keypoints1' not found in data entry")
                    continue
                
                # Check for many-to-one matching and calculate matching rate
                if 'matches0' in data:
                    matches0 = data['matches0']
                    if isinstance(matches0, np.ndarray):
                        if has_many_to_one_matching(matches0):
                            many_to_one_count += 1
                        # Calculate matching rate: valid matches / total keypoints0
                        if num_kp0 > 0:
                            valid_matches = np.sum(matches0 != -1)
                            matching_rate = valid_matches / num_kp0
                            matching_rates.append(matching_rate)
                        
                        # Track total node number and many-to-one node number
                        valid_matches_array = matches0[matches0 != -1]
                        if len(valid_matches_array) > 0:
                            # Count total matched nodes (all nodes that are matched to)
                            total_node_number += len(valid_matches_array)
                            
                            # Count nodes that appear multiple times (many-to-one)
                            unique, counts = np.unique(valid_matches_array, return_counts=True)
                            many_to_one_node_indices = unique[counts > 1]
                            many_to_one_node_number += len(many_to_one_node_indices)
                    elif isinstance(matches0, list):
                        matches0 = np.array(matches0)
                        if has_many_to_one_matching(matches0):
                            many_to_one_count += 1
                        # Calculate matching rate: valid matches / total keypoints0
                        if num_kp0 > 0:
                            valid_matches = np.sum(matches0 != -1)
                            matching_rate = valid_matches / num_kp0
                            matching_rates.append(matching_rate)
                        
                        # Track total node number and many-to-one node number
                        valid_matches_array = matches0[matches0 != -1]
                        if len(valid_matches_array) > 0:
                            # Count total matched nodes (all nodes that are matched to)
                            total_node_number += len(valid_matches_array)
                            
                            # Count nodes that appear multiple times (many-to-one)
                            unique, counts = np.unique(valid_matches_array, return_counts=True)
                            many_to_one_node_indices = unique[counts > 1]
                            many_to_one_node_number += len(many_to_one_node_indices)
                    else:
                        print(f"Warning: matches0 has unexpected type: {type(matches0)}")
                else:
                    # If no matches0, matching rate is 0
                    if num_kp0 > 0:
                        matching_rates.append(0.0)
                
                total_frames += 1
                
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue
    
    print(f"\nTotal frames processed: {total_frames}")
    print(f"Frames with many-to-one matching: {many_to_one_count}")
    print(f"Total node number (matched nodes across all frames): {total_node_number}")
    print(f"Many-to-one node number: {many_to_one_node_number}")
    
    # Generate CSV files for keypoints0
    if keypoints0_counts:
        kp0_df = pd.DataFrame({
            'num_keypoints': sorted(keypoints0_counts.keys()),
            'num_frames': [keypoints0_counts[k] for k in sorted(keypoints0_counts.keys())]
        })
        kp0_csv_path = os.path.join(output_dir, 'keypoints0_statistics.csv')
        kp0_df.to_csv(kp0_csv_path, index=False)
        print(f"\nSaved keypoints0 statistics to: {kp0_csv_path}")
        print(f"Total unique keypoint counts: {len(keypoints0_counts)}")
    
    # Generate CSV files for keypoints1
    if keypoints1_counts:
        kp1_df = pd.DataFrame({
            'num_keypoints': sorted(keypoints1_counts.keys()),
            'num_frames': [keypoints1_counts[k] for k in sorted(keypoints1_counts.keys())]
        })
        kp1_csv_path = os.path.join(output_dir, 'keypoints1_statistics.csv')
        kp1_df.to_csv(kp1_csv_path, index=False)
        print(f"Saved keypoints1 statistics to: {kp1_csv_path}")
        print(f"Total unique keypoint counts: {len(keypoints1_counts)}")
    
    # Analyze matching rates: group by 0 to 1 with step 0.1
    matching_rate_counts = Counter()
    if matching_rates:
        matching_rates_array = np.array(matching_rates)
        # Create bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        # Use np.histogram to count values in each bin
        bins = np.arange(0.0, 1.1, 0.1)
        counts, _ = np.histogram(matching_rates_array, bins=bins)
        
        # Create bin labels and count dictionary
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"
            matching_rate_counts[bin_label] = int(counts[i])
        
        # Generate CSV file for matching rate statistics
        matching_rate_df = pd.DataFrame({
            'matching_rate_range': sorted(matching_rate_counts.keys(), key=lambda x: float(x.split('-')[0])),
            'num_frames': [matching_rate_counts[k] for k in sorted(matching_rate_counts.keys(), key=lambda x: float(x.split('-')[0]))]
        })
        matching_rate_csv_path = os.path.join(output_dir, 'matching_rate_statistics.csv')
        matching_rate_df.to_csv(matching_rate_csv_path, index=False)
        print(f"\nSaved matching rate statistics to: {matching_rate_csv_path}")
        print(f"Total frames with matching rate data: {len(matching_rates)}")
        if len(matching_rates) > 0:
            print(f"Average matching rate: {np.mean(matching_rates):.4f}")
            print(f"Median matching rate: {np.median(matching_rates):.4f}")
    
    return {
        'total_frames': total_frames,
        'many_to_one_count': many_to_one_count,
        'keypoints0_counts': keypoints0_counts,
        'keypoints1_counts': keypoints1_counts,
        'matching_rate_counts': matching_rate_counts,
        'matching_rates': matching_rates,
        'total_node_number': total_node_number,
        'many_to_one_node_number': many_to_one_node_number
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pkl data files and generate statistics")
    parser.add_argument("--pkl_folder", type=str, nargs='+', required=True,
                        help="Folder(s) containing pkl files (searches recursively). Can specify multiple folders.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for CSV files (defaults to first pkl_folder if single, or current dir if multiple)")
    parser.add_argument("--include_dynamic", action='store_true', default=False,
                        help="Include PKL files with 'dynamic' in their path/name (default: False, excludes dynamic files)")
    
    args = parser.parse_args()
    
    # Validate all input folders
    pkl_folders = args.pkl_folder if isinstance(args.pkl_folder, list) else [args.pkl_folder]
    for folder in pkl_folders:
        if not os.path.isdir(folder):
            print(f"Error: {folder} is not a valid directory")
            exit(1)
    
    # Determine output directory
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Created output directory: {args.output_dir}")
        output_dir = args.output_dir
    else:
        # If single folder, use it; if multiple, use current directory
        if len(pkl_folders) == 1:
            output_dir = pkl_folders[0]
        else:
            output_dir = os.getcwd()
            print(f"Multiple folders provided, using current directory for output: {output_dir}")
    
    # Process all folders and aggregate results
    all_keypoints0_counts = Counter()
    all_keypoints1_counts = Counter()
    total_many_to_one_count = 0
    total_frames = 0
    all_matching_rates = []
    total_node_number = 0
    many_to_one_node_number = 0
    
    # Determine exclude_dynamic flag (opposite of include_dynamic)
    exclude_dynamic = not args.include_dynamic
    
    print(f"Processing {len(pkl_folders)} folder(s)...")
    if exclude_dynamic:
        print("Excluding PKL files with 'dynamic' in their path/name (use --include_dynamic to include them)")
    else:
        print("Including all PKL files (including those with 'dynamic' in path/name)")
    
    for i, pkl_folder in enumerate(pkl_folders, 1):
        print(f"\n[{i}/{len(pkl_folders)}] Analyzing pkl files in: {pkl_folder}")
        # Analyze each folder (will write individual CSVs to each folder)
        results = analyze_pkl_data(pkl_folder, output_dir=None, exclude_dynamic=exclude_dynamic)
        
        # Aggregate results
        all_keypoints0_counts.update(results['keypoints0_counts'])
        all_keypoints1_counts.update(results['keypoints1_counts'])
        total_many_to_one_count += results['many_to_one_count']
        total_frames += results['total_frames']
        all_matching_rates.extend(results['matching_rates'])
        total_node_number += results['total_node_number']
        many_to_one_node_number += results['many_to_one_node_number']
    
    # Generate aggregated CSV files
    if all_keypoints0_counts:
        kp0_df = pd.DataFrame({
            'num_keypoints': sorted(all_keypoints0_counts.keys()),
            'num_frames': [all_keypoints0_counts[k] for k in sorted(all_keypoints0_counts.keys())]
        })
        kp0_csv_path = os.path.join(output_dir, 'keypoints0_statistics.csv')
        kp0_df.to_csv(kp0_csv_path, index=False)
        print(f"\nSaved aggregated keypoints0 statistics to: {kp0_csv_path}")
    
    if all_keypoints1_counts:
        kp1_df = pd.DataFrame({
            'num_keypoints': sorted(all_keypoints1_counts.keys()),
            'num_frames': [all_keypoints1_counts[k] for k in sorted(all_keypoints1_counts.keys())]
        })
        kp1_csv_path = os.path.join(output_dir, 'keypoints1_statistics.csv')
        kp1_df.to_csv(kp1_csv_path, index=False)
        print(f"Saved aggregated keypoints1 statistics to: {kp1_csv_path}")
    
    # Analyze aggregated matching rates
    matching_rate_counts = Counter()
    if all_matching_rates:
        matching_rates_array = np.array(all_matching_rates)
        bins = np.arange(0.0, 1.1, 0.1)
        counts, _ = np.histogram(matching_rates_array, bins=bins)
        
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"
            matching_rate_counts[bin_label] = int(counts[i])
        
        matching_rate_df = pd.DataFrame({
            'matching_rate_range': sorted(matching_rate_counts.keys(), key=lambda x: float(x.split('-')[0])),
            'num_frames': [matching_rate_counts[k] for k in sorted(matching_rate_counts.keys(), key=lambda x: float(x.split('-')[0]))]
        })
        matching_rate_csv_path = os.path.join(output_dir, 'matching_rate_statistics.csv')
        matching_rate_df.to_csv(matching_rate_csv_path, index=False)
        print(f"Saved aggregated matching rate statistics to: {matching_rate_csv_path}")
    
    # Create aggregated results dictionary
    results = {
        'total_frames': total_frames,
        'many_to_one_count': total_many_to_one_count,
        'keypoints0_counts': all_keypoints0_counts,
        'keypoints1_counts': all_keypoints1_counts,
        'matching_rate_counts': matching_rate_counts,
        'matching_rates': all_matching_rates,
        'total_node_number': total_node_number,
        'many_to_one_node_number': many_to_one_node_number
    }
    
    # Prepare summary text
    summary_lines = []
    summary_lines.append("="*50)
    summary_lines.append("SUMMARY")
    summary_lines.append("="*50)
    if len(pkl_folders) > 1:
        summary_lines.append(f"Processed {len(pkl_folders)} folders:")
        for folder in pkl_folders:
            summary_lines.append(f"  - {folder}")
        summary_lines.append("")
    summary_lines.append(f"Total frames: {results['total_frames']}")
    summary_lines.append(f"Frames with many-to-one matching: {results['many_to_one_count']}")
    if results['total_frames'] > 0:
        percentage = 100 * results['many_to_one_count'] / results['total_frames']
        summary_lines.append(f"Percentage with many-to-one matching: {percentage:.2f}%")
    summary_lines.append(f"Total node number (matched nodes across all frames): {results['total_node_number']}")
    summary_lines.append(f"Many-to-one node number: {results['many_to_one_node_number']}")
    if results['total_node_number'] > 0:
        percentage = 100 * results['many_to_one_node_number'] / results['total_node_number']
        summary_lines.append(f"Percentage of nodes with many-to-one matching: {percentage:.2f}%")
    
    # Add matching rate statistics to summary
    if results['matching_rate_counts']:
        summary_lines.append("\nMatching Rate Distribution (keypoints0 to keypoints1):")
        summary_lines.append("-" * 50)
        matching_rates_list = results['matching_rates']
        if len(matching_rates_list) > 0:
            summary_lines.append(f"Average matching rate: {np.mean(matching_rates_list):.4f}")
            summary_lines.append(f"Median matching rate: {np.median(matching_rates_list):.4f}")
            summary_lines.append(f"Min matching rate: {np.min(matching_rates_list):.4f}")
            summary_lines.append(f"Max matching rate: {np.max(matching_rates_list):.4f}")
        summary_lines.append("\nMatching Rate Groups:")
        for rate_range in sorted(results['matching_rate_counts'].keys(), key=lambda x: float(x.split('-')[0])):
            count = results['matching_rate_counts'][rate_range]
            percentage = 100 * count / len(matching_rates_list) if len(matching_rates_list) > 0 else 0
            summary_lines.append(f"  {rate_range}: {count} frames ({percentage:.2f}%)")
    
    summary_text = "\n".join(summary_lines)
    
    # Print summary to console
    print("\n" + summary_text)
    
    # Write summary to txt file
    summary_txt_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_txt_path, 'w') as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to: {summary_txt_path}")

