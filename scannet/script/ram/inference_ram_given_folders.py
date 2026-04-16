import argparse
import os
from tqdm import tqdm
from pathlib import Path
import sys


def _bootstrap_repo_root() -> None:
    cur = Path(__file__).resolve()
    for parent in [cur.parent, *cur.parents]:
        if (parent / "scannet" / "script" / "thirdparty").is_dir():
            p = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return


_bootstrap_repo_root()

from scannet.script.thirdparty.ensure_thirdparty import add_to_syspath, ensure_recognize_anything

# Lazily provision recognize-anything (provides the `ram` python package)
add_to_syspath(ensure_recognize_anything(from_file=__file__))

# from inference_ram_plus import run_inference
from inference_ram_plus_openset import RAMPlusOpensetInference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tag2Text inferece for tagging and captioning')
    parser.add_argument('--scans-folder',
                        metavar='DIR',
                        help='path to imagescans folder',
                        default='/media/cc/My Passport/dataset/scannet/images/scans')
    parser.add_argument('--output_json_folder',
                        default='output_json',
                        help='path to output json folder')
    parser.add_argument('--start_scene_id',
                        default=200,
                        type=int,
                        help='start scene id (default: 200)')
    parser.add_argument('--end_scene_id',
                        default=400,
                        type=int,
                        help='end scene id (default: 400)')
    
    parser.add_argument('--image',
                        metavar='DIR',
                        help='path to dataset',
                        default='No need to be set')
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
    parser.add_argument('--save_json',
                        default=True,
                        type=bool,
                        help='save json file (default: True)')

    parser.add_argument('--process_every_n_images',
                        default=3,
                        type=int,
                        help='process every n images (default: 3)')
    parser.add_argument('--llm_tag_des',
                        metavar='DIR',
                        help='path to LLM tag descriptions',
                        default='/home/cc/chg_ws/ros_ws/topomap_ws/src/semantic_topo_map/scannet/script/ram/scannet509.json')


    args = parser.parse_args()

    # get all the folders in the scans folder
    print("Loading scan folders...")
    scan_folders = [f for f in os.listdir(args.scans_folder) if os.path.isdir(os.path.join(args.scans_folder, f))]
    
    # filter the scans_folders by the start and end scene id
    print("Filtering scan folders...")
    filtered_scan_folders = []
    for scan_folder in scan_folders:
        scene_id = int(scan_folder.split('_')[0].split('scene')[-1])
        if scene_id >= args.start_scene_id and scene_id <= args.end_scene_id:
            filtered_scan_folders.append(scan_folder)

    print(f"Filtered scans folders: {filtered_scan_folders}")

    ram_plus_openset_inference = RAMPlusOpensetInference(args.pretrained, args.image_size, args.llm_tag_des)

    # for each filtered scans folder, run the inference_ram_plus.py
    for scan_folder in tqdm(filtered_scan_folders):
        args.image = os.path.join(args.scans_folder, scan_folder)

        ram_plus_openset_inference.run_inference(args)
