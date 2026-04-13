# ScanNet-SG Dataset Download and Unzip Guide

Our dataset has three subsets:
- `ScanNet-SG-509`
- `ScanNet-SG-GPT`
- `ScanNet-SG-Subscan`

Each subset uses the same zip naming and extracted structure.

Before downloading the data, please fill in the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfIxKLDshDNEWlv2y14DWPNavEl0WLEJCdKsf-AQrrlknCjKA/viewform?usp=publish-editor) to help us understand who is using our dataset and for what purpose (one minute). Please read [Terms of Use](/download/Terms%20of%20Use.pdf)

Use the following command to download and unzip the data:
```
python download/download_and_upzip.py your_path_to_store_the_dataset
```


For each subset, the script will download and unzip the following zip files:
- `meta_data.zip`
- `pkl.zip`
- `training_maps.zip`
- `training_refined_instance.zip` (optional)
- `test_maps.zip`
- `test_refined_instance.zip` (optional)
- `test_per_frame_points_ply.zip` (optional)


## Extracted Data Structure

After extraction, each subset root is expected to look like:

```text
<subset_root>/
├── meta_data/
├── pkl/
├── training/scans/
│   └── <scene_id>/
│       ├── instance_cloud_background.ply
│       ├── instance_cloud_cleaned.ply
│       ├── instance_name_map.csv
│       ├── topology_map.json
│       ├── matched_instance_correspondence_to_00.csv (optional)
│       ├── transformation.npy (optional)
│       ├── inv_transformation.txt (optional)
│       └── .../refined_instance/
│           ├── <number>.png
│           └── <number>_final_instance.json
└── test/scans/
    └── <scene_id>/
        ├── instance_cloud_background.ply
        ├── instance_cloud_cleaned.ply
        ├── instance_name_map.csv
        ├── topology_map.json
        ├── matched_instance_correspondence_to_00.csv (optional)
        ├── transformation.npy (optional)
        ├── inv_transformation.txt (optional)
        └── .../refined_instance/
            ├── <number>.png
            └── <number>_final_instance.json
```


## Data Explanation

### meta_data

Includes `meta_data.json`, which contains the range of normalized x, y, z positions and the dimension of the visual-language features stored in the pkl files. The same `meta_data.json` is used everywhere in OpenSGA.

There are also some general data statistics files in the `meta_data` folder.

### pkl

This folder stores the packed pkl files for training and testing in OpenSGA. The pkl files can be used directly in OpenSGA scripts. Check the OpenSGA code for details.

Each pkl contains the items like the following:
```bash
"keypoints0": normalized keypoint positions of the nodes in a frame,
"descriptors0": grounding DINO visual language feature of the nodes in a frame,
"bbox0": normalized bbox size of the nodes in a frame,
"text_embedding0": s-bert embedding of node names in a frame,
"keypoints1": normalized keypoint positions of the nodes in a scene,
"descriptors1": grounding DINO visual language feature of the nodes in a scene,
"text_embedding1": normalized bbox size of the nodes in a scene,
"bbox1": normalized bbox size of the nodes in a scene,
"matches0": ground truth matching vector. The same size as the number of nodes in the frame. 3 means matched with the third node in keypoints1. -1 means no match.
"frame_id": frame id of keypoints0 in Scannet,
"frame_scene": scene id of keypoints0 in Scannet,
"scene_graph_id": scene id of keypoints1 in Scannet
```

### training and test folders

We split the training set of the ScanNet dataset into our own training (`Scene0000-0599`) and test (`Scene0600-0706`) sets. (Scenes in the test/validation set of the original ScanNet dataset are not used because they do not contain rescans of the same scene.)

In each scan folder (e.g., `Scene0000_01`), the files include:

- `instance_cloud_background.ply`: background-only point cloud.
- `instance_cloud_cleaned.ply`: instance-only point cloud. The point color corresponds to the ID of an instance. In `ScanNet-SG-509` and `ScanNet-SG-Subscan`, the values of a point's R, G, B channels are the same and represent the ID (`0-1` corresponding to `0~255`). `ScanNet-SG-GPT` has many more instances, and the ID is decoded by first mapping `0-1` to `0~255`, then calculating `ID = R + G * 255 + B * 255 * 255`.
- `instance_name_map.csv`: ID vs. instance-name mapping.
- `topology_map.json`: scene graph json file, including the ID, name, bbox, feature, etc. of each node.
- `matched_instance_correspondence_to_00.csv`: instance ID correspondence to the first scan (`00`) of a scene. Only exists in the rescan files, i.e., `Scene0000_01` rather than `Scene0000_00`.
- `transformation.npy`: transformation matrix to the first scan (`00`). Only exists in the rescan files, i.e., `Scene0000_01` rather than `Scene0000_00`.
- `inv_transformation.txt`: inverse transformation matrix stored in txt format.
- `refined_instance` folder: contains 1) `<number>.png`, which is the mono8 segmentation image (`0`: background, `1~255`: instance IDs in this image), and 2) `<number>_final_instance.json`, which stores the `instance_id` of an object in this frame and in the scene graph, `object_name`, `description`, `confidence`, `bbox`, etc. `ScanNet-SG-Subscan` does not have this folder.


For original RGB and depth images and camera pose, please download from the ScanNet website.
