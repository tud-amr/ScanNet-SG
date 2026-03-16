import json
import os
import argparse
import re

def extract_json_code_block(text):
    # Extract content between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        # Some results don't have marker ```json, we return the whole text
        return text

def fix_unclosed_quotes(json_str):
    # Pattern: match a key, then a string starting with ", that doesn't end with a quote before the next key
    # We'll look for:
    # "key": "some string possibly missing the closing quote,
    # and next line starts with "anotherKey":
    
    pattern = re.compile(r'(".*?":\s*")([^"\n\r]+)([\n\r])')
    
    def replacer(match):
        key_part = match.group(1)
        val_part = match.group(2)
        newline = match.group(3)
        # Add closing quote to end of val_part
        return f'{key_part}{val_part}"{newline}'
    
    fixed_json = pattern.sub(replacer, json_str)
        
    # change ," to ",
    fixed_json = fixed_json.replace(',"', '",')
    return fixed_json


def clean_json_string(json_str):
    """Clean common JSON syntax errors that might occur in AI-generated JSON"""
    json_str = re.sub(r';\s*"(\w+)":', r', "\1":', json_str)
    
    # Remove trailing commas before closing brackets/braces
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Remove trailing commas in object properties
    json_str = re.sub(r',(\s*})', r'\1', json_str)
    
    # Remove trailing commas in array elements
    json_str = re.sub(r',(\s*\])', r'\1', json_str)

    # Remove all ( and ;
    json_str = json_str.replace(')', '')
    json_str = json_str.replace(';', '')

    json_str = json_str.replace('“', '"').replace('”', '"')

    json_str = fix_unclosed_quotes(json_str)

    json_str = json_str.replace(' " ', '')

    # We'll use a regex to match parentheses outside strings
    result = []
    inside_string = False
    escape = False

    for char in json_str:
        if char == '"' and not escape:
            inside_string = not inside_string
        if char == ')' and not inside_string:
            continue  # skip unmatched closing parentheses
        result.append(char)
        escape = (char == '\\') and not escape  # handle escaped quotes
    
    return ''.join(result)


def normalize_json_structure(data):
    # If top-level is a list, wrap it inside {"objects": [...]}
    if isinstance(data, list):
        return {"objects": data}
    
    # If dict with exactly one key, rename that key to "objects"
    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) == 1:
            key = keys[0]
            return {"objects": data[key]}
        else:
            print(f"Expected exactly one top-level key, but got {len(keys)} keys: {keys}")
            return None

    raise ValueError("Unexpected JSON structure: expected list or dict with one key")


def decode_batch_results(batch_results_file, request_image_paths_file, output_folder):
    success_count = 0
    error_count = 0
    
    with open(request_image_paths_file, "r") as f:
        request_image_paths = [line.strip() for line in f]

    scene_id_list = [img_path.split("/")[-2] for img_path in request_image_paths]
    frame_id_list = [img_path.split("/")[-1].split(".")[0].split("-")[-1] for img_path in request_image_paths]
    frame_id_int_list = [int(frame_id) for frame_id in frame_id_list]
    
    with open(batch_results_file, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            custom_id = data.get("custom_id", f"line_{i}")
            print(custom_id)
            
            # Navigate to assistant's content
            choices = data.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                print(f"[!] No choices in response for {custom_id}")
                continue

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                print(f"[!] Empty content for {custom_id}")
                continue

            # print(f"\n--- {custom_id} ---")
            
            # Extract JSON from markdown code block
            # print(content)
            json_str = extract_json_code_block(content)
            # print(json_str)
            if not json_str:
                print(content)
                print(json_str)
                print(f"[!] No JSON code block found for {custom_id}")
                continue

            # Clean the JSON string to fix common syntax errors
            json_str_cleaned = clean_json_string(json_str)

            # Parse and print object descriptions
            try:
                obj_data = json.loads(json_str_cleaned)
                obj_data = normalize_json_structure(obj_data)
                if obj_data is None:
                    error_count += 1
                    print(f"[!] Data format error for {custom_id}")
                    continue
                
                if not obj_data.get("objects", []):
                    error_count += 1
                    print(f"[!] No objects found for {custom_id}")
                    continue

            except json.JSONDecodeError as e:
                error_count += 1
                print(custom_id)
                print(f"[!] JSON decode error for {custom_id}: {e}")
                print(f"[!] Problematic JSON: {json_str}")
                print(f"[!] Cleaned JSON: {json_str_cleaned}")
                continue
            
            try:
                # If an object has no description, remove it
                obj_data["objects"] = [obj for obj in obj_data.get("objects", []) if obj.get("description", "")]
                # If object has no name, remove it
                obj_data["objects"] = [obj for obj in obj_data.get("objects", []) if obj.get("name", "")]
            except Exception as e:
                error_count += 1
                print(custom_id)
                print(f"[!] Error for {custom_id}: {e}")
                print(f"[!] Problematic JSON: {json_str}")
                print(f"[!] Cleaned JSON: {json_str_cleaned}")
                continue

            save_folder = os.path.join(output_folder, scene_id_list[i], "refined_instance")
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, f"{frame_id_int_list[i]}.json"), "w") as f:
                json.dump(obj_data, f)

            # print(f"Saved to {save_folder}")
            success_count += 1

    print(f"Decoded {len(frame_id_list)} frames, {success_count} success, {error_count} error")

    return success_count, error_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_results_file", type=str, help="The batch results file", default="batch_results.jsonl")
    parser.add_argument("--request_image_paths_file", type=str, default="request_image_paths.txt")
    parser.add_argument("--file_or_folder", type=str, default="file")

    parser.add_argument("--input_folder", type=str, default="/media/cc/Expansion/scannet/backup")
    parser.add_argument("--output_folder", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans")
    args = parser.parse_args()

    if args.file_or_folder == "file":
        success_count, error_count = decode_batch_results(args.batch_results_file, args.request_image_paths_file, args.output_folder)
        print(f"Decoded {success_count} frames, {error_count} error")
    elif args.file_or_folder == "folder":
        # Find all the jsonl files start with "batch_results_"
        batch_results_files = [f for f in os.listdir(args.input_folder) if f.startswith("batch_results_") and f.endswith(".jsonl")]
        print(batch_results_files)
        corresponding_txts = []
        for batch_results_file in batch_results_files:
            corresponding_txt = batch_results_file.replace("batch_results_", "batch_")
            corresponding_txt = corresponding_txt.replace(".jsonl", ".txt")
            corresponding_txts.append(corresponding_txt)

        # Decode the batch results
        total_success_count = 0
        total_error_count = 0
        for batch_results_file, corresponding_txt in zip(batch_results_files, corresponding_txts):
            success_count, error_count = decode_batch_results(os.path.join(args.input_folder, batch_results_file), os.path.join(args.input_folder, corresponding_txt), args.output_folder)
            total_success_count += success_count
            total_error_count += error_count
        print(f"Decoded {total_success_count} frames, {total_error_count} error")

    else:
        print(f"Invalid input file or folder: {args.file_or_folder}")
        exit()