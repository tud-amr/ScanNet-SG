import os
import json
import csv
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="/media/cc/My Passport/dataset/scannet/processed/scans/scene0705_00")
    parser.add_argument("--scene_id", type=str, default="scene0705_00")
    args = parser.parse_args()

    scene_id = args.scene_id
    folder_path = os.path.join(args.input_folder, "refined_instance")
    output_csv = os.path.join(args.input_folder, "instance_name_map.csv")
    output_embeddings = os.path.join(args.input_folder, "instance_bert_embeddings.json")

    # Path to folder containing JSON files
    # folder_path = "/media/cc/My Passport/dataset/scannet/processed/scans/scene0000_00/refined_instance"
    # output_csv = "/media/cc/My Passport/dataset/scannet/processed/scans/scene0000_00/instance_name_map.csv"

    # Dictionary to hold object names for each instance_id
    instance_name_counts = defaultdict(list)

    # Read all JSON files from the folder
    # Show progress
    print(f"Reading JSON files from {folder_path}")
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                try:
                    data = json.load(f)
                    for entry in data:
                        instance_id = int(entry.get('instance_id'))
                        object_name = entry.get('object_name')
                        if instance_id is not None and object_name:
                            instance_name_counts[instance_id].append(object_name)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse {filename}, skipping.")

    # Apply max polling (i.e., most common object_name) for each instance_id
    final_mapping = {
        instance_id: Counter(names).most_common(1)[0][0]
        for instance_id, names in instance_name_counts.items()
    }

    # Write the result to a CSV file and 
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance_id', 'name'])
        for instance_id, name in sorted(final_mapping.items()):
            writer.writerow([instance_id, name])

    print(f"CSV written to {output_csv}")

    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased') # Use distilbert-base-uncased for faster inference and smaller output dimension
    model = BertModel.from_pretrained('distilbert-base-uncased')

    # Get the embedding for each instance name and save to a json file
    instance_embeddings = {}
    for instance_id, name in sorted(final_mapping.items()):
        # print(f"Processing instance {instance_id} with name {name}")
        inputs = tokenizer(name, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        instance_embeddings[instance_id] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

    # Save the embeddings to a json file
    with open(output_embeddings, 'w') as f:
        json.dump(instance_embeddings, f, indent=2)

    print(f"Embeddings saved to {output_embeddings}")