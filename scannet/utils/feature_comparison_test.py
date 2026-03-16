import os
import sys
import json
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "..", "..", "script"))

from include.topology_map import *


def findTopMatches(instance_feature: np.ndarray, topology_map: TopologyMap, top_k: int = 5):
    """
    Find the top k matches for an instance feature in the topology map based on the cosine similarity.
    """
    matches = []
    for node in topology_map.object_nodes.nodes.values():
        node_feature = node.visual_embedding
        # Skip if the node name is unknown
        if node.name == "unknown":
            continue
        # Calculate the cosine similarity between the instance feature and the node feature 
        similarity = np.dot(instance_feature, node_feature) / (np.linalg.norm(instance_feature) * np.linalg.norm(node_feature))
        # Add the match to the list of matches if the similarity is greater than 0.5
        if similarity > 0.5:
            matches.append((node.id, node.name, similarity))
    
    # Sort the matches by similarity
    matches.sort(key=lambda x: x[2], reverse=True)

    if len(matches) < top_k:
        return matches, len(matches)
    else:
        return matches[:top_k], top_k


def findTopMatchesInstanceJson(frame_instance_json_path: str, topology_map: TopologyMap, top_k: int = 5, if_print: bool = False, id_correction_dict: dict = None):
    """
    Find the top k matches for an instance feature in the topology map based on the cosine similarity.
    """
    # Load and parse the JSON
    with open(frame_instance_json_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary mapping instance_id to a dictionary with feature, name, and confidence
    frame_feature_dict = {
        entry['instance_id']: {
            'feature': np.array(entry['feature'], dtype=np.float32),
            'object_name': entry['object_name'],
            'confidence': entry['confidence']
        }
        for entry in data
    }

    results = {}
    # Compare the features of each instance in the frame with the features of the object nodes in the topology map and find the top 5 matches
    for instance_id, instance_data in frame_feature_dict.items():
        # Correct the instance id if the id correction dictionary is provided and the instance id is in the dictionary
        if id_correction_dict is not None and instance_id in id_correction_dict:
            instance_id = id_correction_dict[instance_id]

        # Get the feature of the instance
        instance_feature = instance_data['feature']
        # Get the name of the instance
        instance_name = instance_data['object_name']

        # Find the top k matches
        matches, top_k_count = findTopMatches(instance_feature, topology_map, top_k)
        if if_print:
            print(f"Top {top_k_count} matches for Instance {instance_id} name: {instance_name}")
            for match in matches:
                print(f"Match {match[0]} name: {match[1]} similarity: {match[2]}")
            print("--------------------------------")
        
        results[instance_id] = {
            'object_name': instance_name,
            'matches': matches
        }
    
    return results


def loadIdCorrectionCsv(csv_path: str):
    """
    Load the id correction csv file. This is because scene0000_01 is initially not aligned with scene0000_00, so the instance id in scene0000_01 is not the same as the instance id in scene0000_00.
    """
    df = pd.read_csv(csv_path)
    return {row['instance_id']: row['instance_id_in_00'] for _, row in df.iterrows()}


def testFeatureComparison(args):
    """
    Test the feature comparison between the instance features and the object nodes in the topology map.
    """
    ## Load the id correction csv file
    if args.id_correction_csv_for_frames is not None and os.path.exists(args.id_correction_csv_for_frames):
        id_correction_dict = loadIdCorrectionCsv(args.id_correction_csv_for_frames)
    else:
        id_correction_dict = None

    ## Load the topology map from a json file
    map_path = os.path.join(args.map_folder, "topology_map.json")
    check_top_k = args.check_top_k

    with open(map_path, "r") as f:
        topology_map = TopologyMap()
        topology_map.read_from_json(f.read())

    # Test for a single frame
    # frame_instance_json_path = "/media/cc/My Passport/dataset/scannet/processed/scans/scene0000_00/refined_instance/366.json"
    # single_frame_results = findTopMatchesInstanceJson(frame_instance_json_path, topology_map, top_k=check_top_k)

    # Test for all frames
    folder_path = args.frame_folder
    first_k_matches = [0] * check_top_k
    name_matches = [0] * check_top_k

    total_checked_instance_count = 0
    total_checked_frame_count = 0

    # Check if the first k matches correctly match the instance id
    for file in tqdm(os.listdir(folder_path), desc="Checking frames"):
        if file.endswith(".json"):
            frame_instance_json_path = os.path.join(folder_path, file)
            results = findTopMatchesInstanceJson(frame_instance_json_path, topology_map, top_k=check_top_k, id_correction_dict=id_correction_dict)

            # Check if the first k matches correctly match the instance id
            for frame_instance_id, match_results in results.items():
                id_match = False
                name_match = False
                for i in range(len(match_results['matches'])):
                    # print(match_results['matches'][i][0], frame_instance_id)
                    if int(match_results['matches'][i][0]) == int(frame_instance_id) and not id_match:
                        first_k_matches[i] += 1
                        id_match = True
                    if match_results['matches'][i][1] == match_results['object_name'] and not name_match:
                        name_matches[i] += 1
                        name_match = True
                    if id_match and name_match:
                        break

                total_checked_instance_count += 1
            total_checked_frame_count += 1
    
    first_k_matches_percentage = [match / total_checked_instance_count for match in first_k_matches]
    name_matches_percentage = [match / total_checked_instance_count for match in name_matches]
    

    if args.visualize:
        print(f"Total checked frame count: {total_checked_frame_count}")
        print(f"Total checked instance count: {total_checked_instance_count}")

        # Print the results for id match
        print("Id match results:")
        for i in range(check_top_k):
            print(f"First {i+1} matches: {first_k_matches[i]} ({first_k_matches_percentage[i] * 100}%)")

        for i in range(check_top_k):
            print(f"Success rate of first {i+1} matches: {sum(first_k_matches[:i+1]) / total_checked_instance_count * 100}%")
        
        # Print the results for name match
        print("Name match results:")
        for i in range(check_top_k):
            print(f"First {i+1} matches: {name_matches[i]} ({name_matches_percentage[i] * 100}%)")
    
    return first_k_matches_percentage, name_matches_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--map_folder", type=str, help="The folder that contains the topology map or scenes with topology maps", default="/media/cc/My Passport/dataset/scannet/processed/scans/scene0069_00")
    parser.add_argument("--map_folder", type=str, help="The folder that contains the topology map or scenes with topology maps", default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans")
    parser.add_argument("--check_top_k", type=int, default=5)
    parser.add_argument("--frame_folder", type=str, default="/media/cc/My Passport/dataset/scannet/processed/scans/scene0069_00/refined_instance")
    parser.add_argument("--id_correction_csv_for_frames", type=str, default="/media/cc/My Passport/dataset/scannet/processed/scans/scene0069_00/matched_instance_correspondence_to_00.csv")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--cross_scene_test", action="store_true")
    args = parser.parse_args()

    if args.map_folder.endswith("scans") and args.cross_scene_test:
        '''
        Testing for all scenes with cross scene test. E.g. scene0000_01 to scene0000_00
        '''
        print("testing for all scenes with cross scene test. E.g. scene0000_01 to scene0000_00")
        map_all_scene_folder = os.path.join(args.map_folder)

        avg_first_k_matches_percentage = [0] * args.check_top_k
        avg_name_matches_percentage = [0] * args.check_top_k
        total_checked_instance_count = 0
        
        # Find all the scene folders that don't end with _00
        scene_folders_to_test = [f for f in os.listdir(map_all_scene_folder) if not f.endswith("_00")]
        for scene_folder in scene_folders_to_test:
            print(f"Testing scene {scene_folder}")
            map_folder = os.path.join(map_all_scene_folder, scene_folder.split("_")[0] + "_00")
            frame_folder = os.path.join(map_all_scene_folder, scene_folder, "refined_instance")
            id_correction_csv_for_frames = os.path.join(map_all_scene_folder, scene_folder, "matched_instance_correspondence_to_00.csv")

            # Check if map_folder exists
            if not os.path.exists(map_folder):
                print(f"Warning: Map folder {map_folder} does not exist")
                continue

            # Check if refined_instance folder exists
            if not os.path.exists(frame_folder):
                print(f"Warning: Refined instance folder {frame_folder} does not exist")
                continue
            
            # Check if id_correction_csv_for_frames exists
            if not os.path.exists(id_correction_csv_for_frames):
                print(f"Warning: Id correction csv for frames {id_correction_csv_for_frames} does not exist")
                continue

            args.map_folder = map_folder
            args.frame_folder = frame_folder
            args.id_correction_csv_for_frames = id_correction_csv_for_frames

            # Test the feature comparison
            first_k_matches_percentage, name_matches_percentage = testFeatureComparison(args)

            print(f"First {args.check_top_k} matches: {first_k_matches_percentage}")
            print(f"Name matches: {name_matches_percentage}")

            total_checked_instance_count += 1
            avg_first_k_matches_percentage = [avg + first_k_matches_percentage[i] for i, avg in enumerate(avg_first_k_matches_percentage)]
            avg_name_matches_percentage = [avg + name_matches_percentage[i] for i, avg in enumerate(avg_name_matches_percentage)]

        print(f"Total checked instance count: {total_checked_instance_count}")

        if total_checked_instance_count > 0:
            avg_first_k_matches_percentage = [avg / total_checked_instance_count for avg in avg_first_k_matches_percentage]
            avg_name_matches_percentage = [avg / total_checked_instance_count for avg in avg_name_matches_percentage]
            print(f"Average first {args.check_top_k} matches: {avg_first_k_matches_percentage}")
            print(f"Average name matches: {avg_name_matches_percentage}")

    elif args.map_folder.endswith("scans") and not args.cross_scene_test:
        '''
        Testing for all scenes internally. Consider scenes ends with _00 only.
        '''
        print("testing for all scenes internally")
        map_all_scene_folder = os.path.join(args.map_folder)
        scene_folders_to_test = [f for f in os.listdir(map_all_scene_folder) if f.endswith("_00")]

        avg_first_k_matches_percentage = [0] * args.check_top_k
        avg_name_matches_percentage = [0] * args.check_top_k
        total_checked_instance_count = 0
        id_correction_csv_for_frames = None

        for scene_folder in scene_folders_to_test:
            print(f"Testing scene {scene_folder}")
            map_folder = os.path.join(map_all_scene_folder, scene_folder)
            frame_folder = os.path.join(map_folder, "refined_instance")

            # Check if refined_instance folder exists
            if not os.path.exists(frame_folder):
                print(f"Warning: Refined instance folder {frame_folder} does not exist")
                continue
            
            args.map_folder = map_folder
            args.frame_folder = frame_folder
            args.id_correction_csv_for_frames = id_correction_csv_for_frames
            first_k_matches_percentage, name_matches_percentage = testFeatureComparison(args)

            total_checked_instance_count += 1
            avg_first_k_matches_percentage = [avg + first_k_matches_percentage[i] for i, avg in enumerate(avg_first_k_matches_percentage)]
            avg_name_matches_percentage = [avg + name_matches_percentage[i] for i, avg in enumerate(avg_name_matches_percentage)]

        print(f"Total checked instance count: {total_checked_instance_count}")

        if total_checked_instance_count > 0:
            avg_first_k_matches_percentage = [avg / total_checked_instance_count for avg in avg_first_k_matches_percentage]
            avg_name_matches_percentage = [avg / total_checked_instance_count for avg in avg_name_matches_percentage]
            print(f"Average first {args.check_top_k} matches: {avg_first_k_matches_percentage}")
            print(f"Average name matches: {avg_name_matches_percentage}")
    
    else:   
        '''
        Testing for a single scene
        '''
        print("testing for a single scene")
        first_k_matches_percentage, name_matches_percentage = testFeatureComparison(args)
        print(f"First {args.check_top_k} matches: {first_k_matches_percentage}")
        print(f"Name matches: {name_matches_percentage}")
    