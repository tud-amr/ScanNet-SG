import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add paths for imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.utils.result_visualization import generate_instance_colors


def visualize_instance_masks(processed_data_path, images_path, frame_scene_id, frame_ids, 
                            output_dir, alpha=0.5, show_images=False):
    """
    Visualize instance masks overlaid on color images and save them.
    
    Args:
        processed_data_path: Path to the processed data directory
        images_path: Path to the images directory
        frame_scene_id: Scene ID of the frames
        frame_ids: List of frame IDs to process
        output_dir: Directory to save the output images
        alpha: Transparency factor for overlay (0.0 to 1.0)
        show_images: Whether to display images using cv2.imshow
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate instance colors (RGB format, values in [0, 255])
    instance_colors = generate_instance_colors(0, 255)
    
    # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
    instance_colors_bgr = instance_colors[:, ::-1]
    
    print(f"Processing {len(frame_ids)} frame(s) from scene {frame_scene_id}")
    print(f"Output directory: {output_dir}")
    
    for frame_idx, frame_id in enumerate(frame_ids):
        print(f"\nProcessing frame {frame_idx + 1}/{len(frame_ids)}: frame_id={frame_id}")
        
        # Construct file paths
        color_image_path = os.path.join(images_path, frame_scene_id, f"frame-{frame_id:06d}.color.jpg")
        instance_mask_path = os.path.join(processed_data_path, frame_scene_id, "refined_instance", f"{frame_id}.png")
        
        # Check if files exist
        if not os.path.exists(color_image_path):
            print(f"Warning: Color image not found: {color_image_path}. Skipping frame {frame_id}.")
            continue
        
        if not os.path.exists(instance_mask_path):
            print(f"Warning: Instance mask not found: {instance_mask_path}. Skipping frame {frame_id}.")
            continue
        
        # Load images
        color_image = cv2.imread(color_image_path)
        instance_mask = cv2.imread(instance_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if color_image is None:
            print(f"Warning: Failed to load color image: {color_image_path}. Skipping frame {frame_id}.")
            continue
        
        if instance_mask is None:
            print(f"Warning: Failed to load instance mask: {instance_mask_path}. Skipping frame {frame_id}.")
            continue
        
        # Check if dimensions match
        if color_image.shape[:2] != instance_mask.shape[:2]:
            print(f"Warning: Image dimensions don't match. Color: {color_image.shape[:2]}, Mask: {instance_mask.shape[:2]}. Skipping frame {frame_id}.")
            continue
        
        # Color the instance mask
        # instance_mask contains instance IDs (0 = background, 1-255 = instance IDs)
        # Map each instance ID to its corresponding color
        instance_mask_colored = instance_colors_bgr[instance_mask]
        
        # Create overlay: start with color image
        overlay = color_image.copy()
        
        # Apply colored mask only where instance_mask > 0 (non-background pixels)
        mask_nonzero = instance_mask > 0
        overlay[mask_nonzero] = instance_mask_colored[mask_nonzero]
        
        # Blend overlay with original image using alpha transparency
        # alpha controls the transparency: 0.0 = fully transparent (original image), 1.0 = fully opaque (colored mask)
        blended_image = cv2.addWeighted(overlay, alpha, color_image, 1.0 - alpha, 0, color_image)
        
        # Save the blended image
        output_filename = f"frame_{frame_scene_id}_{frame_id:06d}_instance_mask.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, blended_image)
        print(f"Saved: {output_path}")
        
        # Optionally show the image
        if show_images:
            window_name = f"Frame {frame_id} - Instance Mask"
            cv2.imshow(window_name, blended_image)
            cv2.waitKey(100)  # Wait 100ms for key press
    
    if show_images:
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\nCompleted processing {len(frame_ids)} frame(s)")


def main():
    parser = argparse.ArgumentParser(description="Visualize instance masks overlaid on color images")
    
    # Required arguments
    parser.add_argument("--processed_data_path", type=str, default="/media/cc/Expansion/scannet/processed/scans",
                       help="Path to the processed data directory")
    parser.add_argument("--images_path", type=str, default="/media/cc/Extreme SSD/dataset/scannet/images/scans",
                       help="Path to the images directory")
    parser.add_argument("--frame_scene_id", type=str, required=True,
                       help="Scene ID of the frames (e.g., 'scene0001_01')")
    parser.add_argument("--frame_id", type=int, nargs='+', required=True,
                       help="Frame ID(s) to process. Can specify multiple frame IDs")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory to save the visualized images")
    
    # Optional arguments
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Transparency factor for overlay (0.0 to 1.0). Default: 0.5")
    parser.add_argument("--show_images", action="store_true", default=False,
                       help="Display images using cv2.imshow (default: False)")
    
    args = parser.parse_args()
    
    # Validate alpha value
    if not 0.0 <= args.alpha <= 1.0:
        print(f"Warning: alpha value {args.alpha} is out of range [0.0, 1.0]. Clamping to valid range.")
        args.alpha = max(0.0, min(1.0, args.alpha))
    
    # Convert frame_id to list if single value
    if not isinstance(args.frame_id, list):
        args.frame_id = [args.frame_id]
    
    # Run visualization
    visualize_instance_masks(
        processed_data_path=args.processed_data_path,
        images_path=args.images_path,
        frame_scene_id=args.frame_scene_id,
        frame_ids=args.frame_id,
        output_dir=args.output_dir,
        alpha=args.alpha,
        show_images=args.show_images
    )


if __name__ == "__main__":
    main()

