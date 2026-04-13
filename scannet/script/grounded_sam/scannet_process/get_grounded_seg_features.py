'''
This script is used to get the grounded seg features from the scannet dataset.
We read GT instances (mesh-based) and the corresponding labels for each frame. 
Then we use Grounded-Segment-Anything to get the RGB instance segs and features.
Finally, we match the instance segs with the GT instances based on MIOU to get intance ID.
What we get is the instance mask, ID, category, and features for each frame.
'''

# TODO: Use GroundedSam Class in grounded_sam_simple_demo.py to get the instance segs and features.
# NOTE: Bert should has already been integrated in GroundedSam Class. USe that for convenience.

import os
import sys
import zipfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(path)
sys.path.append(path)

import cv2
import json
import numpy as np
from grounded_sam.grounded_sam.grounded_sam_simple_demo import GroundedSam
from sentence_transformers import SentenceTransformer, util
from progress.bar import Bar
import pandas as pd

# Suppress warnings to make the output clean
import warnings
warnings.filterwarnings("ignore")

class InstanceLabelRefiner:
    def __init__(self, class_name_file):
        self.gsam = GroundedSam()  # instantiate once
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.class_name_df = pd.read_csv(class_name_file, sep="\t")
        self.label_id_to_category = dict(zip(self.class_name_df["id"], self.class_name_df["category"]))

    def set_folder_and_json(self, instance_folder, label_folder, rgb_folder, pixel_threshold=100):
        self.instance_folder = instance_folder
        self.label_folder = label_folder
        self.rgb_folder = rgb_folder
        self.pixel_threshold = pixel_threshold

    def read_instance_image(self, path):
        '''Instance image is a mono8 image'''
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img
    
    def read_label_image(self, path):
        '''Label image is a mono16 image'''
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    # def extract_valid_instances(self, img):
    #     ids, counts = np.unique(img, return_counts=True)
    #     return [i for i, c in zip(ids, counts) if c > self.pixel_threshold and i != 0]

    def extract_valid_instances(self, instance_img):
        exclude_labels = {"floor", "wall", "ceiling", "object"}
        unique_ids = np.unique(instance_img)
        valid_ids = []
        for i in unique_ids:
            if i == 0:
                continue
            if np.count_nonzero(instance_img == i) <= self.pixel_threshold:
                continue
            label = self.instance_id_to_category.get(i, "")
            if label in exclude_labels:
                continue
            valid_ids.append(i)
        return valid_ids

    def compute_mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def match_masks(self, instance_img, masks, labels_result, threshold=0.1):
        """
        Match the masks with the instance IDs based on the IoU and label.
        """
        matched_ids = []
        for mask, label in zip(masks, labels_result):
            best_iou = 0
            best_id = 0
            for inst_id in self.extract_valid_instances(instance_img):
                instance_mask = instance_img == inst_id
                instance_label = self.instance_id_to_category.get(inst_id, "")

                # if instance_label != label:
                #     continue

                # Compute the similarity between the instance label and the mask label
                embeddings = self.bert_model.encode([instance_label, label])
                similarity = util.cos_sim(embeddings[0], embeddings[1])
                if similarity < 0.4: # Labels should have close meaning
                    continue
                
                iou = self.compute_mask_iou(mask, instance_mask)
                if iou > best_iou and iou > threshold:
                    best_iou = iou
                    best_id = inst_id

            if best_iou > threshold:
                matched_ids.append(best_id)
            else:
                matched_ids.append(0)
                # print(f"No match found for mask {label}")
        return matched_ids
    
    def get_instance_id_to_category(self, label_img, instance_img):
        instance_ids = np.unique(instance_img)
        instance_id_to_label_id = {}

        for instance_id in instance_ids:
            if instance_id == 0:
                continue  # skip background/unlabeled
            mask = (instance_img == instance_id)
            label_vals = label_img[mask]
            
            if len(label_vals) == 0:
                continue

            # Get the most frequent label ID for this instance
            label_id = np.bincount(label_vals).argmax()
            instance_id_to_label_id[instance_id] = label_id

        instance_id_to_category = {
            inst_id: self.label_id_to_category.get(label_id, 'unknown')
            for inst_id, label_id in instance_id_to_label_id.items()
        }

        return instance_id_to_category

    def process_image_pair(self, instance_filename, output_mask_dir, output_json_dir):
        instance_path = os.path.join(self.instance_folder, instance_filename)
        label_path = os.path.join(self.label_folder, instance_filename)

        rgb_name = f"frame-{int(os.path.splitext(instance_filename)[0]):06d}.color.jpg"
        rgb_path = os.path.join(self.rgb_folder, rgb_name)

        instance_img = self.read_instance_image(instance_path)
        label_img = self.read_label_image(label_path)

        # Set instance_id_to_category for this image
        self.instance_id_to_category = self.get_instance_id_to_category(label_img, instance_img)

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")

        # Get labels for current instance IDs
        instance_ids = self.extract_valid_instances(instance_img)
        labels = [self.instance_id_to_category.get(i, "unknown") for i in instance_ids]

        labels_no_overlap = []
        for label in labels:
            if label not in labels_no_overlap:
                labels_no_overlap.append(label)
        # print(f"labels_no_overlap: {labels_no_overlap}")

        # If no labels, skip
        if len(labels_no_overlap) == 0:
            print(f"No labels found for {instance_filename}, skipping")
            return

        # Grounded SAM inference
        annotated, masks, class_ids, confidences, features = self.gsam.infer(
            rgb_img, labels_no_overlap, box_threshold=0.2, text_threshold=0.2, nms_threshold=0.8, confidence_threshold=0.3
        )
        #labels_result = [labels_no_overlap[i] for i in class_ids]
        labels_result = []
        for class_id in class_ids:
            if class_id == -1:
                labels_result.append("object")  # CHG. Change to "object" to indicate the wrong phrase.
            else:
                labels_result.append(labels_no_overlap[class_id])
        # print(f"labels_result: {labels_result}")

        # Match SAM masks to original instance IDs
        matched_ids = self.match_masks(instance_img, masks, labels_result)

        # Create refined mono8 image
        refined_mask = np.zeros(instance_img.shape, dtype=np.uint8)
        for i, mask in enumerate(masks):
            refined_mask[mask] = matched_ids[i]

        os.makedirs(output_mask_dir, exist_ok=True)
        mask_out_path = os.path.join(output_mask_dir, instance_filename)
        cv2.imwrite(mask_out_path, refined_mask)
        # print(f"Saved refined mask to {mask_out_path}")

        ### Save the annotated image
        ## This will take big storage space.
        # cv2.imwrite(os.path.join(output_mask_dir, f"annotated_{instance_filename}"), annotated)

        ### Save original masks from GroundedSAM
        # os.makedirs(output_mask_dir, exist_ok=True)
        # for i, mask in enumerate(masks):
        #     mask_out_path = os.path.join(output_mask_dir, f"original_mask_{instance_filename}_{i}_{labels_result[i]}.png")
        #     mask_to_save = (mask * 255).astype(np.uint8)
        #     cv2.imwrite(mask_out_path, mask_to_save)
        #     print(f"Saved original mask to {mask_out_path}")

        # Create and save metadata JSON
        os.makedirs(output_json_dir, exist_ok=True)
        image_info = []
        for i in range(len(masks)):
            if matched_ids[i] == 0: # Not matched
                continue
            entry = {
                "instance_id": int(matched_ids[i]),
                "object_name": labels_result[i],
                "confidence": float(confidences[i]),
                "feature": features[i].tolist()  # features is a NumPy array
            }
            image_info.append(entry)

        json_out_path = os.path.join(output_json_dir, instance_filename.replace(".png", ".json"))
        with open(json_out_path, 'w') as f:
            json.dump(image_info, f, indent=2)
        # print(f"Saved instance info to {json_out_path}")

    def natural_sort_key(self, s):
        """
        Sort strings with numbers in natural order.
        For example: ['0.png', '2.png', '11.png', '20.png'] instead of ['0.png', '11.png', '20.png', '2.png']
        """
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    def process_all(self, output_mask_dir, output_json_dir, skip_every_n_frames=1):
        #for i, fname in enumerate(sorted(os.listdir(self.instance_folder), key=self.natural_sort_key)):
        # Add progress bar
        for i, fname in enumerate(Bar('Processing').iter(sorted(os.listdir(self.instance_folder), key=self.natural_sort_key))):
            if fname.endswith(".png") and i % skip_every_n_frames == 0:
                self.process_image_pair(fname, output_mask_dir, output_json_dir)


if __name__ == "__main__":
    refiner = InstanceLabelRefiner("/media/cc/My Passport/dataset/scannet/data/scannetv2-labels.combined.tsv")
    
    # Process the whole dataset
    dataset_root = "/media/cc/My Passport/dataset/scannet"
    instance_label_subfolder = "data"
    rgb_subfolder = "images"
    output_subfolder = "processed"

    process_every_n_frames = 3

    # Get all the files end with "instance-filt.zip" under the instance_json_subfolder and its subfolders
    instance_files = []
    for root, dirs, files in os.walk(os.path.join(dataset_root, instance_label_subfolder)):
        for file in files:
            if file.endswith("instance-filt.zip"):
                instance_files.append(os.path.join(root, file))
    print(f"Found {len(instance_files)} instance-filt.zip files")

    # Unzip the instance-filt.zip files into the same folder if the folder is not already unzipped
    for instance_label_file in Bar('Processing').iter(instance_files):
        if not os.path.exists(os.path.join(os.path.dirname(instance_label_file), "instance-filt")):
            with zipfile.ZipFile(instance_label_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(instance_label_file))
        else:
            print(f"Skipping {instance_label_file} because it is already unzipped")

        # Unzip the xx-label-filt.zip if it does not exist
        if not os.path.exists(os.path.join(os.path.dirname(instance_label_file), "label-filt")):
            with zipfile.ZipFile(instance_label_file.replace("_2d-instance-filt.zip", "_2d-label-filt.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(instance_label_file))
        else:
            print(f"Skipping {instance_label_file.replace('_2d-instance-filt.zip', '_2d-label-filt.zip')} because it is already unzipped")

        
        relative_parent_folder = os.path.relpath(os.path.dirname(instance_label_file), os.path.join(dataset_root, instance_label_subfolder))
        rgb_folder = os.path.join(dataset_root, rgb_subfolder, relative_parent_folder)
        print(f"Processing {rgb_folder}")

        # get the number of jpg files in the rgb_folder
        num_jpg_files = len([f for f in os.listdir(rgb_folder) if f.endswith(".jpg")])
        num_frames_to_process = num_jpg_files // process_every_n_frames
        print(f"frames to be processed: {num_frames_to_process}")

        output_mask_dir = os.path.join(dataset_root, output_subfolder, relative_parent_folder, "refined_instance")
        output_json_dir = output_mask_dir

        # Create the output directories if they don't exist
        os.makedirs(output_mask_dir, exist_ok=True)
        # Check the number of png files in the output_mask_dir
        num_png_files = len([f for f in os.listdir(output_mask_dir) if f.endswith(".png")])
        print(f"existing mask png files: {num_png_files}")
        if num_png_files >= num_frames_to_process / 2:
            print(f"Skipping {rgb_folder} because the result already exists")
            continue

        refiner.set_folder_and_json(
            instance_folder=os.path.join(os.path.dirname(instance_label_file), "instance-filt"),
            label_folder=os.path.join(os.path.dirname(instance_label_file), "label-filt"),
            rgb_folder=rgb_folder)
        
        refiner.process_all(
            output_mask_dir=output_mask_dir,
            output_json_dir=output_json_dir,
            skip_every_n_frames=process_every_n_frames
        )
        
