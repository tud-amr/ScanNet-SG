'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import os
import torch
import json
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

unwanted_embeddings = None
        

def filter_unwanted_tags(tags_list, sentence_model, similarity_threshold=0.3):
    """
    Filter out tags that are highly similar to walls, floors, roads, and colors
    """
    # Define reference terms for unwanted categories
    global unwanted_embeddings
    if unwanted_embeddings is None:
        unwanted_categories = [
            # Walls
            'wall', 'ceiling', 'floor', 'room',
            # Roads
            'road', 'street', 'pavement', 'sidewalk', 'asphalt',
            # Colors
            'white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'brown', 'gray', 'grey', 'pink', 'color', 'colored',
        ]
        unwanted_embeddings = sentence_model.encode(unwanted_categories)
    
    tag_embeddings = sentence_model.encode(tags_list)

    # Compute similarity between each tag and unwanted categories
    filtered_tags = []
    for i, tag in enumerate(tags_list):
        max_similarity = 0
        for unwanted_embedding in unwanted_embeddings:
            # Compute cosine similarity
            similarity = np.dot(tag_embeddings[i], unwanted_embedding) / (
                np.linalg.norm(tag_embeddings[i]) * np.linalg.norm(unwanted_embedding)
            )
            # print(f"Similarity: {similarity}")
            max_similarity = max(max_similarity, similarity)
        
        # Keep tag if similarity is below threshold
        if max_similarity < similarity_threshold:
            filtered_tags.append(tag)

    # Filter out tags that have word "room"
    filtered_tags = [tag for tag in filtered_tags if 'room' not in tag]
    
    return filtered_tags


def create_objects_json(filtered_tags):
    """
    Create JSON structure with objects array where description equals name
    """
    objects = []
    for tag in filtered_tags:
        # Clean up the tag name (remove extra spaces, capitalize properly)
        clean_name = tag.strip().title()
        objects.append({
            "name": clean_name,
            "description": clean_name
        })

    return {"objects": objects}

def save_json(json_path, tags_list):
    # make the directory if not exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # save the json file
    data = create_objects_json(tags_list)
    with open(json_path, 'w') as f:
        json.dump(data, f)


def run_inference(args):
    if args.save_json:
        scene_folder_name = args.image.split('scans/')[-1]
        print(f"Scene folder name: {scene_folder_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    # Load sentence transformer model once
    print("Loading sentence transformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence transformer model loaded!")

    # check if the image is a path or a folder
    if os.path.isdir(args.image):
        # get all the images in the folder
        image_names = [f for f in os.listdir(args.image) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_seqs = [int(image_name.split('.')[0].split('-')[-1]) for image_name in image_names]

        # Sort the image_paths by the image_seqs
        image_names = [x for _, x in sorted(zip(image_seqs, image_names))]
        # print(f"Image names: {image_names}")

        for i, image_name in enumerate(tqdm(image_names)):
            # process every n images
            if i % args.process_every_n_images != 0:
                continue

            image = transform(Image.open(os.path.join(args.image, image_name))).unsqueeze(0).to(device)
            res = inference(image, model)
            # Put res[0] into a list
            tags_list = res[0].split(' | ')
            # print(f"Original tags: {tags_list}")
            
            # Filter unwanted tags
            filtered_tags = filter_unwanted_tags(tags_list, sentence_model, args.similarity_threshold)
            # print(f"Filtered tags: {filtered_tags}")

            # print(f"Image name: {image_name}")

            if args.save_json:
                image_seq = int(image_name.split('.')[0].split('-')[-1])
                # print(f"Image sequence: {image_seq}")
                save_json_path = os.path.join(args.output_json_folder, scene_folder_name, "refined_instance", f"{image_seq}.json")
                # print(f"Save json path: {save_json_path}")
                save_json(save_json_path, filtered_tags)

    else:
        image = transform(Image.open(args.image)).unsqueeze(0).to(device)
        res = inference(image, model)
        print("Image Tags: ", res[0])
        print("图像标签: ", res[1])

        tags_list = res[0].split(' | ')

        # filter out unwanted tags
        filtered_tags = filter_unwanted_tags(tags_list, sentence_model, args.similarity_threshold)
        print(f"Filtered tags: {filtered_tags}")
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tag2Text inferece for tagging and captioning')
    parser.add_argument('--image',
                        metavar='DIR',
                        help='path to dataset',
                        default='images/demo/demo1.jpg')
    parser.add_argument('--pretrained',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='/media/cc/Expansion/models/ram_plus_swin_large_14m.pth')
    parser.add_argument('--image_size',
                        default=384,
                        type=int,
                        metavar='N',
                        help='input image size (default: 448)')
    parser.add_argument('--similarity_threshold',
                        default=0.6,
                        type=float,
                        help='similarity threshold for filtering (default: 0.3)')
    parser.add_argument('--output_json_folder',
                        default='output_json',
                        help='path to output json folder')
    parser.add_argument('--save_json',
                        default=False,
                        type=bool,
                        help='save json file (default: False)')
    parser.add_argument('--process_every_n_images',
                        default=3,
                        type=int,
                        help='process every n images (default: 3)')

    args = parser.parse_args()

    run_inference(args)

    
