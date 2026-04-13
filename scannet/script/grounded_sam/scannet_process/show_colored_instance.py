import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def random_color_map(num_colors, seed=42):
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors

def colorize_labels(image):
    unique_vals = np.unique(image)
    color_map = random_color_map(len(unique_vals))
    color_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    for idx, val in enumerate(unique_vals):
        color_image[image == val] = color_map[idx]
    
    return color_image

def read_and_colorize_labeled_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Image could not be read. Check the path or format.")

    print(f"Image dtype: {image.dtype}, shape: {image.shape}")
    
    if image.dtype == np.uint16 or image.dtype == np.uint8:
        colorized = colorize_labels(image)
    else:
        raise ValueError("Only mono8 (uint8) or mono16 (uint16) images supported.")
    

    # Add text numbers of the instance id at cluster center given in the mono8 image to the colorized image
    for i in range(len(np.unique(image))):
        mask = image == i
        if np.sum(mask) > 0:
            center = np.mean(np.where(mask), axis=1)
            cv2.putText(colorized, str(i), tuple(center.astype(int)[::-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Calculate the maximum and minimum values in the image
    max_val = np.max(image)
    min_val = np.min(image)
    print(f"Max value: {max_val}, Min value: {min_val}")

    cv2.imshow("Colorized Image", colorized)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # Example usage with command line arguments
    parser = argparse.ArgumentParser(description="Show colored instance images")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    read_and_colorize_labeled_image(args.image_path)
    cv2.waitKey(0)
    exit()

    # Show images in a folder. Break the loop when q is pressed
    #folder_path = "/media/cc/DATA/dataset/scannet/data/scans/scene0000_00/scene0000_00_2d-instance-filt/instance-filt"
    # folder_path = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/openset_scans/scene0704_00/refined_instance"
    # for file in os.listdir(folder_path):
    #     if file.endswith(".png") and "annotated" not in file:
    #         print(f"Showing {file}")
    #         read_and_colorize_labeled_image(os.path.join(folder_path, file))
    #         key = cv2.waitKey(0)
    #         if key == ord('q'):
    #             break
