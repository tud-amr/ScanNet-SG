import json
import base64
import os
import argparse

def write_batch_jsonl(image_paths, output_path, model_name="gpt-4o-mini"):
    with open(output_path, "w") as f:
        for img_path in image_paths:
            with open(img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            request = {
                "custom_id": img_path.split("/")[-2] + "_" + img_path.split("/")[-1],  # helps track responses
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What are the objects in this image (exclude floor, wall, ceiling, etc.)? Give the name and a detailed description of each object and present it in JSON format.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                }
            }
            f.write(json.dumps(request) + "\n")


def write_batch_jsonl_scene(processed_data_folder, raw_images_folder, scene_id, start_frame, end_frame, output_jsonl_path="batch_inputs.jsonl", output_image_paths_path="request_image_paths.txt", model_name="gpt-4.1-nano"): 
    '''
    Write the batch jsonl file for a scene
    Args:
        processed_data_folder: Note: this folder is used to find the image ids to process. THe images are assumed to be in the refined_instance folder with name frameid.png. Only the name is used.
        raw_images_folder: the folder that contains the raw images
        scene_id: the id of the scene
        start_frame: the start frame id
        end_frame: the end frame
    '''
    # Find scans folder that contains the scene_id
    scans_folder_list = [f for f in os.listdir(processed_data_folder) if scene_id in f]
    if len(scans_folder_list) == 0:
        print(f"Scene {scene_id} not found in {processed_data_folder}")
        exit()
        
    # Get the image ids from the refined_instance folder
    refined_instance_folder = os.path.join(processed_data_folder, scans_folder_list[0], "refined_instance")
    image_files = [f for f in os.listdir(refined_instance_folder) if f.endswith(".png")]
    image_ids = [int(f.split(".")[0]) for f in image_files]
    image_ids.sort()

    # print(f"Image ids: {image_ids}")
    
    # Filter the image ids
    image_ids = [i for i in image_ids if i >= start_frame and i <= end_frame]
    # print(f"Filtered image ids: {image_ids}")

    # Get the image paths from the raw_images folder
    image_paths = [os.path.join(raw_images_folder, scans_folder_list[0], f"frame-{i:06d}.color.jpg") for i in image_ids]

    # print(f"Total number of images: {len(image_paths)}")

    write_batch_jsonl(image_paths, output_jsonl_path, model_name)

    # Write the request image paths to a file
    with open(output_image_paths_path, "w") as f:
        for img_path in image_paths:
            f.write(img_path + "\n")

    # Check if the size of the output file is larger than 198MB
    if os.path.getsize(output_jsonl_path) > 198 * 1024 * 1024:
        print(f"The size of the output file is larger than 198MB")
        raise ValueError("The size of the output file is larger than 198MB")
    
    return output_jsonl_path, output_image_paths_path
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_folder", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/scans")
    parser.add_argument("--raw_images_folder", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/images/scans")
    parser.add_argument("--scene_id", type=str, default="scene0704_01")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=2000)
    parser.add_argument("--output_jsonl_path", type=str, default="batch_inputs.jsonl")
    parser.add_argument("--output_image_paths_path", type=str, default="request_image_paths.txt")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-nano") #gpt-4o-mini
    args = parser.parse_args()

    output_jsonl_path, output_image_paths_path = write_batch_jsonl_scene(args.processed_data_folder, args.raw_images_folder, args.scene_id, args.start_frame, args.end_frame, args.output_jsonl_path, args.output_image_paths_path, args.model_name)

