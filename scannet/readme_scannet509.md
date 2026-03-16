# ScanNet Topology Map Generation

## Prerequest: Generate Fine Segmentation Masks

See https://github.com/g-ch/text_seg_anything/tree/topomap/script/scannet_process

Use ```get_grounded_seg_features.py``` to generate the fine segmentation masks.

__IMPORTANT:__ Not all instances labeled in scannet can be detected by GroundedDINO. Therefore, the number of refined masks is usually smaller than the number of masks in Scannet. The masks that are not detected will be labeled as unknown in the final topology map and the visual feature is zero. We should avoid using these undetected objects. 


## Generate Topology Map

To generate the topology map for all scenes after the Fine Segmentation Masks are generated.
First change the following in ```map_generator_all.py``` to your path. The step parameter should be the same as the one used in Fine Segmentation generation.

```
processed_dataset_dir = "/media/cc/My Passport/dataset/scannet/processed"
exec_path = "/home/cc/chg_ws/ros_ws/topomap_ws/devel/lib/semantic_topo_map"
step = 3 # Processed every 3 frames in the processed dataset
```

Then run script
```
python map_generator_all.py
```
Note this requires BERT to be installed in your conda environment. Everything will be saved in the corresponding scene folder. E.g., ```/media/cc/My Passport/dataset/scannet/processed/scans/```, as this:

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

This uses the segmentation result of ```get_grounded_seg_features.py``` to generate the scene PLY.

```
./scannet_ply_map or rosrun semantic_topo_map scannet_ply_map <scene_name> <start_frame> <end_frame> <step>
```

Note that the default dataset path is: 
```
    std::string raw_images_parent_dir = "/media/cc/My Passport/dataset/scannet/images/scans/";
    std::string refined_instance_parent_dir = "/media/cc/My Passport/dataset/scannet/processed/scans/";
    std::string output_ply_dir = "/media/cc/My Passport/dataset/scannet/processed/scans/" + scene_name + "/";
```
Change the path in the code to your own path and recompile the code. 
TODO: Put the path in a config file.


### Generate Instance to Object Name Map

```
python get_instance_names.py --input_folder "/media/clarence/My Passport/dataset/scannet/processed/scans/scene0000_00" --scene_id "scene0000_00"
```

This will generate a instance_name_map.csv and a instance_embeddings.json in the input folder. The instance embeddings uses BERT to encode the instance name.


### Get Fused Visual-Language Features for Instances in a Scene

```
rosrun semantic_topo_map get_fused_object_features <path_to_json_files> <result_save_folder>";
```

- __path_to_json_files__: e.g. scannet/processed/scans/scene0000_00/refined_instance
- __result_save_folder__: e.g. scannet/processed/scans/scene0000_00


### Generate Topology Map for a scene

```
rosrun semantic_topo_map generate_json '/media/cc/My Passport/dataset/scannet/processed/scans/scene0000_00/instance_cloud.ply' 0 (or 1 for visualizing the result)
```

### Visualize the Topology Map

```
./read_and_visualize_map <map_file>
```

## Generate Aligned Instance for Scenes with more than One Scans

Some scenes have more than one scans (e.g. scene0000_00, scene0000_01, scene0000_02).
We want to test finding the node observed in scene0000_01 with map built in scene0000_00. So we need to align the instance id of scene0000_01 and scene0000_00. We do that by aligning the Scene PLY with RANSAC + ICP first to get the transformation (from scenexxxx_00 to scenexxxx_0x). Then find the instance correspondence with point overlapping and bert name correspondence or direct name comparing (default: fast and less false). The transformation matrix will be saved as transformation.npy in scenexxxx_0x's folder. (transformation: _00 -> _0x, inv_transformation: _0x -> _00).
The aligned cloud, for visualization and checking, is saved as aligned_cloud_with_scan_00.ply.



run
```bash
python scannet/script/align_instances.py --source_dir path_to_scene0000_00 --target_dir path_to_scene0000_01 --visualize --ori_pt_transform
```
to get the alignment from scene0000_01 to scene0000_00. This will generate a csv file named ``matched_instance_correspondence_to_00.csv'' in the folder of scene0000_01.

To run the script for all scenes
```bash
python scannet/script/align_instances_for_all.py --data_dir xxx/scans --skip_existing
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

__Differen Scenes Might need different Parameters to Get a Fine Result__. Run the following to examine the results

```bash
python alignment_examine.py --dataset_dir xxx/processed/scans --new
```
Add ```--new``` when you examine for the first time. This will create a to_examine.csv containing the scenes to be examined (unviewed and negatives). When use it for the second time, remove ```--new```. When examining, press 'p' (positive, nice alignment), 'n' (negative) to label, or 'q'/Esc (quit).

After examination, tune the parameters in ```align_instances.py``` and run 
```bash
python align_instances_for_all.py --data_dir xxx/processed/scans --use_scene_csv
```
With ```--use_scene_csv``` added only the scenes listed in the csv will be considered to avoid align again for the already good scenes.


## Per Frame Data Finalize
To train a matching network from a single frame to the graph. We need to further get bbox (with size, center position and orientation) of each instance in a frame in the camera coordinate. Meanwhile, we can get the point cloud of each instance for the usage of some matching model that requires point cloud feature.

Run the following to get point cloud in a ply, and bbox in a json. The files will be saved in a new folder '''scans/per_frame_points''' for each scan.
```bash
python scannet/script/frame_ptc_all.py --start_scene_seq xx --end_scene_seq xx --processed_data_folder xxx/scans --raw_images_folder xxx/scannet/images/scans
```

Next, run
```bash
python scannet/script/add_pose_bbox_to_frame_json.py --scans_folder xxx/scans --skip_existing
```
To update the per frame json in the ```refined_instance``` folder. The updated json will be named ```frameid_final_instance.json```.

## Generate Data used for training our Matcher

```bash
python matcher_data_generation.py --add_cross_scan --use_scene_exclude_csv --map_folder xxx/processed/scans --data_output_dir xxx --save_every_n_scenes 100
```

The training data containing features, normalized object (keypoint) positions and bbox sizes (optionally) will be saved in pkl files. An additional txt file will also be generated with the following content perline:
```
map_scene frame_scene frame_id frame_transformed_pose_in_map(matrix with 16 floats)
```

## Map PLY Outiler Removing
```
python script/map_ply_post_filter.py xxx/scans
```
This will remove outliers of each instance and save a "instance_cloud_cleaned.ply".

## Quick Evaluation
Here we provide code to quickly evaluate how the visual-language feature from grounded_dino work perform for relocalization.

### Pre-request
__Generate Fine Segmentation Masks__
__Generate Topology Map__
__Generate Aligned Instance for Scenes with more than One Scans__

Run
```bash
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

If --map_folder is not a specific scene and ends with ```scans```, we do average matching success rate for all scenes. If --cross_scene_test is added, performance is for scene0000_01,2 to scene0000_00, etc. Otherwise it is for scene0000_00 to scene0000_00 and scene0000_01 to scene0000_01, etc.

If --map_folder and --frame_folder are specific scenes, evaluation from frame_folder to map_folder is performed. If they are from different scans of the same scene, --id_correction_csv_for_frames from the prerequist step needs to be specified.


# Addtional tools
Use ```utils/data_analysis.py``` to analyze pkls to know how many objects are in the frames and scans stochastically.