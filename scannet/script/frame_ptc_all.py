import os
import sys
import tqdm
import argparse

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, ".."))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PER_FRAME_EXE = "scannet_per_frame_points"


def resolve_per_frame_points_exec_dir(repo_root):
    """Prefer standalone CMake build in-repo; fall back to classic catkin devel layout."""
    candidates = [
        os.path.join(repo_root, "scannet", "cpp", "build"),
        os.path.normpath(
            os.path.join(repo_root, "..", "..", "devel", "lib", "semantic_topo_map")
        ),
    ]
    for d in candidates:
        exe = os.path.join(d, PER_FRAME_EXE)
        if os.path.isfile(exe):
            return d
    tried = "\n  ".join(candidates)
    raise FileNotFoundError(
        f"Could not find ./{PER_FRAME_EXE}. Tried:\n  {tried}\n"
        "Build with CMake under scannet/cpp/build, or install the ROS package so the "
        "binary exists under <ws>/devel/lib/semantic_topo_map/."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_scene_seq", type=int, default=301)
    parser.add_argument("--end_scene_seq", type=int, default=706)
    parser.add_argument("--processed_data_folder", type=str, default="/media/cc/Expansion/scannet/processed/scans")
    parser.add_argument("--raw_images_folder", type=str, default="/media/cc/My Passport/dataset/scannet/images/scans")
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="If set, save per-frame PLYs (otherwise passes 'false' to the executable).",
    )
    args = parser.parse_args()

    # Get scene names between 0000 to 0090.
    start_scene_seq = args.start_scene_seq
    end_scene_seq = args.end_scene_seq

    scene_names = [f"scene{i:04d}" for i in range(start_scene_seq, end_scene_seq + 1)]

    print("Scene names to process:")
    print(scene_names)

    folder_path = args.processed_data_folder
    exec_path = resolve_per_frame_points_exec_dir(root_path)
    print(f"Exec path: {exec_path}")
    print(f"Save ply: {args.save_ply}")

    all_subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    folders_with_wanted_scene_names = [f for f in all_subfolders if any(scene_name in f for scene_name in scene_names)]

    # Remove the prefix and '/' of the folder path
    folders_with_wanted_scene_names = [f.replace(folder_path + "/", "") for f in folders_with_wanted_scene_names]

    print(folders_with_wanted_scene_names)

    # Now generate data for all

    ## Generate PLY map ./scannet_ply_map <scene_name> <start_frame> <end_frame> <step>
    for scene_folder in tqdm.tqdm(folders_with_wanted_scene_names):
        os.chdir(exec_path)
        if args.save_ply:
            os.system("./{} '{}' '{}' '{}'".format(PER_FRAME_EXE, scene_folder, args.processed_data_folder, args.raw_images_folder))
        else:
            os.system("./{} '{}' '{}' '{}' '{}' ".format(PER_FRAME_EXE, scene_folder, args.processed_data_folder, args.raw_images_folder, "false"))
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))

        