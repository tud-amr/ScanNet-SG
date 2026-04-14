# ScanNet Topology Map Generation Openset

__Difference__: For open-set objects, we use the OpenAI API + Grounded Segment Anything to find objects. Therefore, only RGB-D images and the corresponding camera poses are needed.
Each instance in a frame is first given a frame_instance_id. We use a comprehensive overlapping score to merge masks of the same instance across frames and give a (global) instance_id. Also, the name of the same instance observed from different frames can be different while their bert feature should be close. Therefore, in the final_instance.json, we store ```frame_instance_id```, ```bert_embedding```, and ```discription``` (text given by Openai API) in addtion to the items used in fixed set scans.

We assume you have the RGB-D images and camera poses in ScanNet format.

### Expected ScanNet-format layout (minimum)

- **Folder per scan**: `<scans-folder>/sceneXXXX_YY/`
- **RGB frames**: `frame-000000.color.jpg`, `frame-000001.color.jpg`, ...
- **Depth frames** (optional for this openset pipeline if you only generate 2D masks): `frame-000000.depth.png`, ...
- **Camera poses**: `pose/000000.txt`, `pose/000001.txt`, ... (4x4 camera-to-world, whitespace separated)
- **Intrinsics**: `intrinsic/intrinsic_color.txt` (and optionally `intrinsic/intrinsic_depth.txt`)

This matches the standard ScanNet extracted frame format where frame index `000000` corresponds to `frame-000000.*`.


## Prerequisite: Generate Fine Segmentation Masks

First, do tagging using either `openai_tools` or `recognize anything (RAM)`.

To use `openai_tools` to get the names and descriptions of objects in images, check the [README](script/openai_tools/readme.md) in the `openai_tools` folder for details.

Alternatively, run 
```bash
python scannet/script/ram/inference_ram_given_folders.py --scans-folder folder_with_rgbd_images_scan_in scannet_format  --output_json_folder xxx --start_scene_id e.g.0 --end_scene_id e.g.100 --pretrained xxx/ram_plus_swin_large_14m.pth --process_every_n_images 3
```
to use RAM for tagging. Check `inference_ram_given_folders.py` for detailed input parameters. Make sure you have downloaded the `.pth` model (for example, `ram_plus_swin_large_14m.pth`) from [RAM](https://github.com/xinyu1205/recognize-anything).

Then run Grounded-SAM to generate fine segmentation masks:

```bash
python scannet/script/grounded_sam/scannet_process/get_seg_openset.py --image_folder folder_with_rgbd_images_scan_in scannet_format --json_folder output_json_folder_of_the_last_step --confidence_threshold 0.4
```

Notes:
- The first time you run either RAM or Grounded-SAM scripts, this repo will **auto-clone** the upstream projects into `scannet/script/thirdparty/`:
  - `Grounded-Segment-Anything` (Grounded-SAM): `https://github.com/IDEA-Research/Grounded-Segment-Anything`
  - `recognize-anything` (RAM): `https://github.com/xinyu1205/recognize-anything`
- You still need to install the Python dependencies required by those projects (PyTorch, etc.) in your environment. A ready-to-use conda/mamba env is provided at repo root: `environment.yml`.


## Generate Topology Map

To generate the topology map for all scenes after the fine segmentation masks are generated, build the C++ tools (they are plain CMake targets; ROS is not required), then run `map_generator_openset_all.py`.

### Build the C++ tools (CMake)

1) One way is to put the entire repo in a ROS workspace (tested on ROS1 noetic) and catkin build. Then the `--exec_path` parameter should be set to the `devel/lib/scannet_sg` folder of your workspace when running `map_generator_openset_all.py`.

2) Another way is to build in the Ubuntu system directly. The repo can be placed anywhere.
   
Ubuntu example dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
  libeigen3-dev libboost-filesystem-dev \
  libopencv-dev libpcl-dev \
  libdw-dev libelf-dev
```

Configure + build:

```bash
mkdir scannet/cpp/build
cd scannet/cpp/build
cmake ..
cmake --build . -j"$(nproc)"
```

This produces binaries such as `openset_ply_map` and `generate_json` under `scannet/cpp/build/`.

`map_generator_openset_all.py` defaults to that directory.

Then run:
```bash
python scannet/script/map_generator_openset_all.py \
  --raw_images_parent_dir folder_with_rgbd_images_scan_in scannet_format \
  --processed_dataset_dir output_json_folder_of_the_last_step \
  --max_depth 2.0 /
  --post_filter
```
Note: the Python post-processing uses sentence embeddings (via `sentence-transformers` in our conda env), not a separate “BERT pip package”.
The following will be generated for each scan.

```
├── scene0000_00
│   ├── averaged_instance_features.json
│   ├── colored_instances.ply
│   ├── instance_bert_embeddings.json
│   ├── instance_cloud.ply
│   ├── instance_name_map.csv
│   ├── refined_instance
│   └── topology_map.json

```


You can also run the following command step by step to generate the topology map for a single scene.


### Generate Scene PLY

```bash
cd scannet/cpp/build
./openset_ply_map scene_folder 0 processed_dataset_dir raw_images_parent_dir max_depth subsample_factor
```

### Generate Topology Map for a Scene

```bash
cd scannet/cpp/build
./generate_json scene_instances_ply_file_path visualize_flag three_channel_id_flag edge_distance_threshold
```

### Visualize the Topology Map

```
./read_and_visualize_map <map_file>
```

## Generate Aligned Instances for Scenes with More Than One Scan

Some scenes have more than one scan (e.g., scene0000_00, scene0000_01, scene0000_02).
We want to test finding the node observed in scene0000_01 with map built in scene0000_00. So we need to align the instance id of scene0000_01 and scene0000_00. We do that by aligning the Scene PLY with RANSAC + ICP first to get the transformation (from scenexxxx_00 to scenexxxx_0x). Then find the instance correspondence with point overlapping and bert name correspondence or direct name comparing (default: fast and less false). The transformation matrix will be saved as transformation.npy in scenexxxx_0x's folder.(transformation: _00 -> _0x, inv_transformation: _0x -> _00). The aligned cloud, for visualization and checking, is saved as aligned_cloud_with_scan_00.ply.

Run:
```bash
python scannet/script/align_instances.py --source_dir path_to_scene0000_00 --target_dir path_to_scene0000_01 --visualize --ori_pt_transform --use_bert_embeddings --three_channel_id 
```
to get the alignment from scene0000_01 to scene0000_00. This will generate a csv file named ``matched_instance_correspondence_to_00.csv'' in the folder of scene0000_01.

To run the script for all scenes
```bash
python scannet/script/align_instances_for_all.py --data_dir xxx/scans --skip_existing --openset_scans
```

There will be another csv generated in scene0000_01.
```
├── scene0000_01
│   ├── averaged_instance_features.json
│   ├── colored_instances.ply
│   ├── instance_bert_embeddings.json
│   ├── instance_cloud.ply
│   ├── instance_name_map.csv
│   ├── __matched_instance_correspondence_to_00.csv__
│   ├── refined_instance
│   └── topology_map.json

```

__The following README is from the fixed-set setting and has not been validated yet.__

__Different scenes might need different parameters to get a good result__. Run the following to examine the results:

```bash
python alignment_examine.py --dataset_dir xxx/processed/scans --new
```
Add `--new` the first time you examine results. This will create a `to_examine.csv` containing the scenes to be examined (unviewed and negatives). When you run it a second time, remove `--new`. When examining, press `p` (positive), `n` (negative), or `q`/Esc (quit).

After examination, tune the parameters in ```align_instances.py``` and run 
```bash
python align_instances_for_all.py --data_dir xxx/processed/scans --use_scene_csv
```
With `--use_scene_csv`, only the scenes listed in the CSV will be considered, to avoid re-aligning scenes that are already good.


## Per Frame Data Finalize
To train a matching network from a single frame to the graph. We need to further get bbox (with size, center position and orientation) of each instance in a frame in the camera coordinate. Meanwhile, we can get the point cloud of each instance for the usage of some matching model that requires point cloud feature.

Run the following to get the point cloud in a PLY file and the bounding box in a JSON file. The files will be saved in a new folder `openset_scans/per_frame_points` for each scan.
```bash
python scannet/script/frame_ptc_all.py --start_scene_seq xx --end_scene_seq xx --processed_data_folder xxx/openset_scans --raw_images_folder xxx/scannet/images/scans
```

Next, run
```bash
python scannet/script/add_pose_bbox_to_frame_json.py --scans_folder xxx/openset_scans --openset_scans --skip_existing
```
To update the per frame json in the ```refined_instance``` folder. The updated json will be named ```frameid_final_instance.json```.


## Generate Data Used for Training the Matcher

```bash
python matcher_data_generation.py --map_folder xxx/processed/scans --data_output_dir xxx --save_every_n_scenes 100
```

The training data (features, normalized object (keypoint) positions, and optional bounding-box sizes) will be saved in PKL files. An additional TXT file will also be generated with the following content per line:
```
map_scene frame_scene frame_id frame_transformed_pose_in_map(matrix with 16 floats)
```

## Quick Evaluation
Here we provide code to quickly evaluate how the visual-language features from GroundingDINO perform for relocalization.

### Pre-request
__Generate Fine Segmentation Masks__
__Generate Topology Map__
__Generate Aligned Instance for Scenes with more than One Scans__

Run
```
python feature_comparison_test.py
```

Here are the parameters you need to provide:
```
parser.add_argument("--map_folder", type=str, help="The folder that contains the topology map or scenes with topology maps", default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans")
    parser.add_argument("--check_top_k", type=int, default=5)
    parser.add_argument("--frame_folder", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans/scene0000_02/refined_instance")
    parser.add_argument("--id_correction_csv_for_frames", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/scans/scene0000_02/matched_instance_correspondence_to_00.csv")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--cross_scene_test", action="store_true")
```

If `--map_folder` is not a specific scene and ends with `scans`, we compute the average matching success rate for all scenes. If `--cross_scene_test` is set, performance is evaluated from scene0000_01/02 to scene0000_00, etc. Otherwise, it is evaluated for scene0000_00→scene0000_00 and scene0000_01→scene0000_01, etc.

If `--map_folder` and `--frame_folder` are specific scenes, evaluation from `frame_folder` to `map_folder` is performed. If they are from different scans of the same scene, `--id_correction_csv_for_frames` from the prerequisite step must be specified.
