import argparse
from write_batch_jsonl import write_batch_jsonl_scene
from submit_batch_to_openai import submit_batch_to_openai
from check_batch_status_and_retrieve import check_batch_status_and_retrieve
from decode_batch_results import decode_batch_results
import os
import tqdm
import time
import sys
from openai import OpenAI

def read_openai_keys():
    with open("openai_keys.txt", "r") as f:
        keys = f.readlines()
        # Remove the \n
        keys = [key.strip() for key in keys]
    return keys

def set_openai_key(key):
    os.environ["OPENAI_API_KEY"] = key


def scenes_openai_inference(processed_data_folder, raw_images_folder, scene_id, start_frame, end_frame, output_folder, model_name, client):
    # Write the batch jsonl file
    name = f"{scene_id}_{start_frame}_{end_frame}"
    output_jsonl_path = "openai_submit_batch.jsonl"
    output_image_paths_path = "batch_" + name + ".txt"
    save_file_name = "batch_results_" + name + ".jsonl"

    # Skip if the save_file_name exists
    if os.path.exists(save_file_name):
        print(f"Skipping scene {scene_id} from frame {start_frame} to {end_frame} because the save_file_name exists")
        return "skipped"

    # Write the batch jsonl file
    output_jsonl_path, output_image_paths_path = write_batch_jsonl_scene(processed_data_folder, raw_images_folder, scene_id, start_frame, end_frame, output_jsonl_path, output_image_paths_path, model_name)

    # Submit the batch to OpenAI
    batch_id = submit_batch_to_openai(output_jsonl_path, client)

    # Check the batch status and retrieve the results
    batch_results_file = check_batch_status_and_retrieve(batch_id, client, save_file_name=save_file_name)
    
    # Decode the batch results. Many syntax errors are found in the batch results. Let's process it later
    # decode_batch_results(batch_results_file, output_image_paths_path, output_folder)

    # Sleep for 10 seconds
    time.sleep(10)

    return batch_results_file

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_folder", type=str, default="/media/cc/Expansion/scannet/processed/scans")
    parser.add_argument("--raw_images_folder", type=str, default="/media/cc/My Passport/dataset/scannet/images/scans")
    parser.add_argument("--start_scene_id", type=int, default="50")
    parser.add_argument("--end_scene_id", type=int, default="100")
    parser.add_argument("--process_every_n_frames", type=int, default=1000)
    parser.add_argument("--output_folder", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini") #gpt-4.1-nano gpt-5-mini
    args = parser.parse_args()

    # Find all the subfolders in the processed data folder. They are named as scene0704_01, scene0704_02, etc. Find the scene ids in the range.
    scene_ids = [f for f in os.listdir(args.processed_data_folder) if f.startswith("scene")]
    scene_ids_int = [int(scene_id.split("_")[0].split("e")[-1]) for scene_id in scene_ids]
    indices = [i for i, scene_id in enumerate(scene_ids_int) if scene_id >= args.start_scene_id and scene_id <= args.end_scene_id]
    scene_ids = [scene_ids[i] for i in indices]
    scene_ids_int = [scene_ids_int[i] for i in indices]

    # Sort the scene ids with the int value
    scene_ids_int, scene_ids = zip(*sorted(zip(scene_ids_int, scene_ids)))

    print(f"Found {len(scene_ids)} scenes")
    print(f"Scene ids: {scene_ids}")

    # Read the openai keys
    openai_keys = read_openai_keys()
    if len(openai_keys) == 0:
        print("No openai keys found. Please add your openai keys to the openai_keys.txt file.")
        exit()
    print(f"Found {len(openai_keys)} openai keys. Use the first one first.")
    
    current_key_seq = 0
    set_openai_key(openai_keys[current_key_seq])
    print(f"Set the openai key to {openai_keys[current_key_seq]}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Process each scene
    for scene_id in tqdm.tqdm(scene_ids):
        # Find all the "png" images in the processed data folder
        png_images = [f for f in os.listdir(os.path.join(args.processed_data_folder, scene_id, "refined_instance")) if f.endswith(".png")]
        print(f"Processing scene {scene_id}: Found {len(png_images)} png images")
        png_images_int = [int(png_image.split(".")[0]) for png_image in png_images]
        max_frame_id = max(png_images_int)
        # Process every args.process_every_n_frames frames
        for i in range(0, max_frame_id, args.process_every_n_frames):
            start_frame = i
            end_frame = min(i + args.process_every_n_frames - 1, max_frame_id)
            print(f"Processing scene {scene_id} from frame {start_frame} to {end_frame}")

            save_file = scenes_openai_inference(args.processed_data_folder, args.raw_images_folder, scene_id, start_frame, end_frame, args.output_folder, args.model_name, client)
            if save_file is None:
                # Usually because the available tokens are not enough. Let's try the next key
                current_key_seq += 1
                if current_key_seq >= len(openai_keys):
                    print("No more openai keys to try. Please add more openai keys to the openai_keys.txt file.")
                    exit()
                set_openai_key(openai_keys[current_key_seq])
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print(f"Current openai key: {os.getenv('OPENAI_API_KEY')}")

                # Retry the scene inference
                save_file = scenes_openai_inference(args.processed_data_folder, args.raw_images_folder, scene_id, start_frame, end_frame, args.output_folder, args.model_name, client)
                if save_file is None:
                    print(f"With the new key {openai_keys[current_key_seq]}, the error is still the same. Please check the error and fix it.")
                    exit()
