import os
import subprocess
import sys
import threading

# Set paths
parent_folder = "/media/cc/My Passport/dataset/scannet/data"
#parent_folder = "/media/cc/My Passport/dataset/scannet/data/test"
output_base_folder = "/media/cc/My Passport/dataset/scannet/images"
cpp_executable = "/media/cc/My Passport/dataset/ScanNet/SensReader/c++/sens"

# Variable to control stop
stop_flag = False

def listen_for_quit():
    global stop_flag
    print("Press 'q' + Enter at any time to stop...")
    while True:
        key = input()
        if key.lower() == 'q':
            stop_flag = True
            print("Stopping after current file...")
            break

# Start the listening thread
listener_thread = threading.Thread(target=listen_for_quit, daemon=True)
listener_thread.start()

# Walk through the directory
for root, dirs, files in os.walk(parent_folder):
    if stop_flag:
        break
    for file in files:
        if file.endswith(".sens"):
            if stop_flag:
                break

            # Full path to the .sens file (F1)
            sens_file_path = os.path.join(root, file)
            print(f"Found: {sens_file_path}")

            # Extract relative path after parent_folder
            relative_path = os.path.relpath(root, parent_folder)

            # Destination folder path (P2)
            destination_folder = os.path.join(output_base_folder, relative_path)

            # Check if destination folder exists and is non-empty
            if os.path.isdir(destination_folder) and any(os.scandir(destination_folder)):
                print(f"Skipping {sens_file_path} because {destination_folder} already exists and is not empty.")
                continue

            # Create destination folder if it does not exist
            os.makedirs(destination_folder, exist_ok=True)

            # Run the C++ executable with the sens file path and destination folder
            try:
                subprocess.run([cpp_executable, sens_file_path, destination_folder], check=True)
                print(f"Processed {sens_file_path} into {destination_folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {sens_file_path}: {e}")

print("Finished.")
