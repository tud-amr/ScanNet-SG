'''
This script is used to get the segmentation masks and GroundingDINO features for the openset scannet dataset.
The openset object names are from the chatgpt api and are stored in the json file.
The images are the same as the fixed-set images.
'''


import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(path)
sys.path.append(path)

import cv2
import json
import numpy as np
from grounded_sam.grounded_sam.grounded_sam_simple_demo import GroundedSam
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import tqdm
import argparse
import traceback

class InstanceSegmenter:
    def __init__(self, visualize=False, confidence_threshold=0.3):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gsam = GroundedSam()
        self.visualize = visualize
        self.confidence_threshold = confidence_threshold

    def set_folder_and_json(self, json_dir, image_dir):
        self.json_dir = Path(json_dir)
        self.image_dir = Path(image_dir)

    def process(self):
        # Get all json files in the json_dir that doesn't end with _instance.json
        json_files = sorted(self.json_dir.glob("*.json"))
        json_files = [file for file in json_files if not file.name.endswith("_instance.json")]

        print(f"Processing {len(json_files)} json files")

        #### Process each json file
        for json_file in tqdm.tqdm(json_files):
            frame_index = int(json_file.stem)
            image_filename = f"frame-{frame_index:06d}.color.jpg"
            image_path = self.image_dir / image_filename

            if not image_path.exists():
                print(f"Image not found for {json_file.name}: {image_path}")
                continue

            # Read object info
            with open(json_file, 'r') as f:
                obj_data = json.load(f)

                #name_description_list = [f"{obj['name']}: {obj['description']}" for obj in obj_data['objects']]

            name_description_dict = {}
            for obj in obj_data['objects']:
                # Skip objects with empty or None names/descriptions
                if obj.get('name') and obj.get('description'):
                    # Ensure name and description are strings
                    if isinstance(obj['name'], str) and isinstance(obj['description'], str):
                        name = obj['name'].strip()
                        description = obj['description'].strip()
                        if name and description:  # Only add if both are non-empty after stripping
                            # Clean the name to make it more suitable for GroundingDINO
                            # Remove special characters and normalize
                            clean_name = name.replace('/', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ')
                            clean_name = ' '.join(clean_name.split())  # Remove extra whitespace
                            if clean_name:  # Only add if cleaned name is not empty
                                name_description_dict[clean_name] = description
                    else:
                        print(f"Warning: Skipping object with non-string name or description in {json_file.name}")
                        print(f"name type: {type(obj['name'])}, description type: {type(obj['description'])}")
                        continue

            # Exclude objects with name "Floor, floor, Wall, wall, Ceiling, ceiling, Roof, roof"
            exclude_names = ["Floor", "floor", "Wall", "wall", "Ceiling", "ceiling", "Roof", "roof", "way", "Shadow"]
            name_description_list = [f"{name}: {description}" for name, description in name_description_dict.items() if not any(exclude_name in name for exclude_name in exclude_names)]
            
            if len(name_description_list) == 0:
                print(f"Skipping {json_file.name} because the number of objects is 0")
                continue

            if self.visualize:
                print(f"name_description_list: {name_description_list}")
                print(f"name_list: {name_list}")
                print(f"Number of valid names: {len(name_list)}")
            
            # Debug output to help identify issues
            print(f"Processing {json_file.name}: {len(name_description_list)} objects")
            if len(name_description_list) > 0:
                print(f"Sample object: {name_description_list[0]}")

            # Read image
            rgb_img = cv2.imread(str(image_path))
            if rgb_img is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Run segmentation
            try:
                # Split on the first ':' only (descriptions can contain extra colons)
                name_list = [name_description.split(":", 1)[0] for name_description in name_description_list]
                
                # Additional validation: ensure no empty names
                name_list = [name.strip() for name in name_list if name.strip()]
                
                if len(name_list) == 0:
                    print(f"Skipping {json_file.name} because no valid names found after filtering")
                    continue
                
                print(f"Processing {json_file.name} with {len(name_list)} names: {name_list}")
                
                annotated, masks, class_ids, confidences, features = self.gsam.infer(
                    rgb_img,
                    #name_description_list,
                    name_list,
                    box_threshold=0.2,
                    text_threshold=0.2,
                    nms_threshold=0.3,
                    confidence_threshold=self.confidence_threshold
                )
            except NotImplementedError:
                print("GroundedSam model is not implemented.")
                return
            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")
                print(f"name_list: {name_list}")
                print(f"name_description_list: {name_description_list}")
                print(traceback.format_exc())
                continue

            if self.visualize:
                cv2.imshow("annotated", annotated)
                cv2.waitKey(0)

            # Generate mono8 mask. Assume max instance id is 255 in one frame.
            if len(masks) > 0:
                # Filter valid instances
                image_pixel_num_limit = rgb_img.shape[0] * rgb_img.shape[1] * 0.8 # if the instance is larger than 80% of the image, it is not a valid instance
                valid_indices = [i for i, cid in enumerate(class_ids) if cid != -1 and np.sum(masks[i]) < image_pixel_num_limit]
                
                # Rank the masks by size. We paint bigger masks first.
                valid_indices = sorted(valid_indices, key=lambda x: np.sum(masks[x]), reverse=True)
                instance_img = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
                for i in range(len(valid_indices)):
                    instance_img[masks[valid_indices[i]]] = i + 1

                mask_out_path = self.json_dir / f"{frame_index}.png"
                cv2.imwrite(str(mask_out_path), instance_img)

                #### Prepare instance info
                name_list = []
                description_list = []
                for idx in valid_indices:
                    class_id = class_ids[idx]
                    name_description = name_description_list[class_id]
                    if ":" in name_description:
                        name, description = name_description.split(":", 1)
                    else:
                        name, description = name_description, ""
                    name_list.append(name.strip())
                    description_list.append(description.strip())

                # Compute BERT embeddings only for valid descriptions
                # bert_embeddings = self.bert_model.encode(description_list)
                bert_embeddings = self.bert_model.encode(name_list)

                # Build image_info only for valid instances
                image_info = []
                for i in range(len(valid_indices)):
                    idx = valid_indices[i]
                    entry = {
                        "instance_id": -1,  # -1 means not allocated yet by matching
                        "frame_instance_id": i + 1,
                        "object_name": name_list[i],
                        "object_description": description_list[i],
                        "confidence": float(confidences[idx]), # Orignal confidence list. Use idx instead if i
                        "feature": features[idx].tolist(),  # Orignal feature list. Use idx instead if i
                        "bert_embedding": bert_embeddings[i].tolist()
                    }
                    image_info.append(entry)

                json_out_path = self.json_dir / f"{frame_index}_instance.json"
                with open(json_out_path, 'w') as f:
                    json.dump(image_info, f, indent=2)

            else:
                print(f"No masks found for {json_file.name}")

def check_if_result_exists(json_folder):
    json_files = sorted(os.listdir(json_folder))
    instance_json_files = [file for file in json_files if file.endswith("_instance.json")]
    json_files = [file for file in json_files if not file.endswith("_instance.json") and file.endswith(".json")]
    print(f"number of instance json files: {len(instance_json_files)}")
    print(f"number of json files: {len(json_files)}")
    return len(instance_json_files) == len(json_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/media/cc/My Passport/dataset/scannet/images/scans")
    parser.add_argument("--json_folder", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum confidence for keeping a detection/mask from Grounded-SAM")
    args = parser.parse_args()

    segmenter = InstanceSegmenter(visualize=args.visualize, confidence_threshold=args.confidence_threshold)
    # If the json_folder is the openset_scans folder, we need to process each subfolder
    if args.json_folder.endswith("openset_scans"):
        print("**********Processing openset scannet**********")
        subfolders = [f for f in os.listdir(args.json_folder) if os.path.isdir(os.path.join(args.json_folder, f))]
        for subfolder in subfolders:
            json_folder = os.path.join(args.json_folder, subfolder, "refined_instance")
            image_folder = os.path.join(args.image_folder, subfolder)
            if args.skip_existing and check_if_result_exists(json_folder):
                print(f"Skipping {subfolder} because it already exists")
                continue
            print(f"Processing {subfolder}")
            segmenter.set_folder_and_json(json_folder, image_folder)
            segmenter.process()
    else:
        print("**********Processing fixed set scannet**********")
        # We assume the json_folder is the refined_instance folder and only process one scene
        if args.skip_existing and check_if_result_exists(args.json_folder):
            print(f"Skipping {args.json_folder} because it already exists")
            exit()
        print(f"Processing {args.json_folder}")
        segmenter.set_folder_and_json(args.json_folder, args.image_folder)
        segmenter.process()
