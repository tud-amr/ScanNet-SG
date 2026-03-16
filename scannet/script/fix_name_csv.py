''''
Read all the instance_name_map.csv in the subfolders of the processed_dataset_dir
and fix the name of the instances by removing the second ",", "(" and ")", and replace "/" with Space
'''

import os
import argparse

def fix_csv_format(csv_path):
    """
    Fix CSV files that have multiple commas by keeping only the first comma as a delimiter.
    This function reads the file, processes each line to keep only the first comma,
    and overwrites the original file with the cleaned data.
    
    Args:
        csv_path (str): Path to the CSV file to fix
    """
    import tempfile
    import shutil
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Find the first comma
                first_comma_idx = line.find(',')
                if first_comma_idx == -1:
                    # No comma found, write the line as is
                    outfile.write(line + '\n')
                else:
                    # Split only on the first comma
                    first_part = line[:first_comma_idx]
                    second_part = line[first_comma_idx + 1:]
                    
                    # Remove any remaining commas from the second part
                    second_part = second_part.replace(',', '')
                    second_part = second_part.replace('(', '')
                    second_part = second_part.replace(')', '')
                    second_part = second_part.replace('/', ' or ')
                    # Write the cleaned line
                    outfile.write(f"{first_part},{second_part}\n")
        
        # Overwrite the original file with the cleaned temporary file
        shutil.move(temp_path, csv_path)
        
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dataset_dir", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans/to_add/openset_scans")
    args = parser.parse_args()

    processed_dataset_dir = args.processed_dataset_dir

    for scene_folder in os.listdir(processed_dataset_dir):
        instance_name_map_path = os.path.join(processed_dataset_dir, scene_folder, "instance_name_map.csv")
        if os.path.exists(instance_name_map_path):
            fix_csv_format(instance_name_map_path)
                