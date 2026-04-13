# ScanNet-SG

This repository contains the code for the __ScanNet-SG__ Dataset.

This dataset is built on top of ScanNet by adding 3D scene graphs that contain open-set visual-language (GroundingDINO) features, BERT features, bounding boxes, etc., for each object in each scene.
The dataset is mainly designed for frame-to-scan and subscan-to-subscan scene graph alignment. But it can also be used for the validation of navigation.

For more details, please refer to our paper:
__ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment__ and 
__OpenSGA: Efficient 3D Scene Graph Alignment in the Open World__ (Coming soon).


## Dataset Download
To download our dataset, please check [here](/download/download_and_upzip.py)


## Environment Installation

This section explains how to prepare your machine to work with ScanNet-SG. What you install depends on how you plan to use the project: many users only need 1) a lightweight setup to load the data and run the Python utilities, while others will 2) reproduce our full pipeline for building new scene graphs and alignment data. The instructions below walk through both cases step by step. For 1), we provide both python and C++ usage interface and examples.

Clone code:
```bash
git clone git@github.com:tud-amr/ScanNet-SG.git --recurse-submodule
cd ScanNet-SG
```

__1) Usage only environment__

If you only want to use the dataset, install the environment as follows (using mamba instead of conda will be much faster):
```bash
conda create -n scannet-sg python=3.10
conda activate scannet-sg
conda install -c conda-forge numpy matplotlib open3d -y
# If you wish to have the full visualization functions (for images in ScanNet), also install opencv with the following command
pip install opencv-python
```

To use C++ interface, do the following:
```bash
cmake -S src -B build_read_and_visualize_map
cmake --build build_read_and_visualize_map
```

__2) Environment for building new scene graphs and alignment data__

If you wish to generate new scene graphs and alignment data with our tools, install the environment by:

```bash
conda env create -f environment.yml
conda activate scannet-sg
```

Or update an existing env:

```bash
conda env update -f environment.yml --prune
```

This installation includes a single environment spec that is intended to run:
- Grounded-SAM openset masking (`scannet/script/grounded_sam/...`)
- RAM tagging (`scannet/script/ram/...`)
- OpenAI batch tooling (`scannet/script/openai_tools/...`)
- Visualization / utilities under `script/` and `scannet/script/`

Notes:
- The optional third-party repos are cloned on demand into `scannet/script/thirdparty/` when you run the scripts.
- Model checkpoints (e.g. SAM, GroundingDINO, RAM++ weights) are **not** included in the environment and must be downloaded separately.


Some generation scripts also call small C++ tools (for example `openset_ply_map` and `generate_json`). Those are built with plain CMake (ROS is not required):

```bash
sudo apt install -y build-essential cmake pkg-config \
  libeigen3-dev libboost-filesystem-dev \
  libopencv-dev libpcl-dev \
  libdw-dev libelf-dev

cmake -S scannet/cpp -B scannet/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build scannet/cpp/build -j"$(nproc)"
```


## Map Interface Usage

### Python version:
The Python class for I/O of the SceneGraph (or TopologyMap) JSON file is defined in `script/include/topology_map.py` as `TopologyMap`.
Check the examples below to learn how to use the interface. (The interface also contains free-space nodes, but we do not use them in the current version.)

- Read a scene graph from a json file
```bash
python script/read_map.py
```

- Visualize a scene graph using the following command
```bash
python script/visualize_map.py --show_bboxes --show_edges
```
Add `--map_ply_path xxx.ply --topology_map_path xxx.json` to specify the data. Add `--enable_picking` to use interactive mode: the name of a node will be printed when you press `Shift` and left-click the blue sphere of a node.
By default, example data in `sample_data/scans/scene0000_00` will be used. You will see an image like the following:

![image](/sample_data/scans/scene0000_00/scene_0000.png)



- Generate a random scene graph
```bash
python script/random_map_generator.py
```

### C++ version:

C++ data structure is defined in `include/topology_map.h` 
Check the example in the following to know how to use the C++ version interface.

To use the example, we recommend putting this repo in a ROS1 workspace and running `catkin build` first to compile.

- Read and visualize a scene graph
```bash
./read_and_visualize_map <map_file>
```

ROS2 support is on the way. We only use the libraries and CMake in ROS to make the installation easier. No communication is established.


## Generate Scene Graphs with Your Own Data
Please refer to [OpenSet F2S data generation](scannet/readme_openset.md) and [S2S data generation](scannet/readme_subscan.md)


## Citation
```
@dataset{scannet_sg,
  author    = {Gang Chen and Sebastián Barbas Laina and Javier Alonso-Mora},
  title     = {ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment},
  year      = {2026},
  doi       = {10.4121/bebe8bd4-cf91-4f86-a28a-87cb870f6cea}, 
  url       = {https://data.4tu.nl/datasets/bebe8bd4-cf91-4f86-a28a-87cb870f6cea}
}
```

## Licence
The code in this repo uses the Apache-2.0 licence.
The dataset uses CC BY-NC 4.0 licence.
