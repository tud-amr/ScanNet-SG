# ScanNet-SG

This repository contains the code for the __ScanNet-SG__ Dataset.

This dataset is built on top of ScanNet by adding 3D Scene Graphs that contains open-set visual-language (groundingDINO) feature, Bert feature, bounding box, etc., of each object for each scene. 
The dataset is mainly designed for frame-to-scan and subscan-to-subscan scene graph alignment. But it can also be used for the validation of navigation.

For more details, please refer to our paper:
__ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment__ and 
__OpenSGA: Efficient 3D Scene Graph Alignment in the Open World__ (Coming soon).


## Dataset Download
To download our dataset, please check [here](/download/download_and_upzip.py)


## Environment Installation
Clone code:
```bash
git clone git@github.com:tud-amr/ScanNet-SG.git --recurse-submodule
cd ScanNet-SG
```

__Skip__ the following installation __if__ you already installed the environment for OpenSGA. 
OpenSGA's environment is fully competable with ScanNet-SG.


If you only want to use the Dataset, please install the environment by:
```bash
conda create -n scannet-sg python=3.10
conda activate scannet-sg
conda install -c conda-forge numpy matplotlib open3d -y
# If you wish to have the full visualization functions (for images in ScanNet), also install opencv with the following command
pip install opencv-python
```

If you wish to generate new data with our tools, please install the environment by:
```
Coming soon.
```


## Map Interface Usage

### Python version:
The python class for the IO of the SceneGraph(or TopologyMap) json file is defined in ```script/include/topology_map.py``` with class ```TopologyMap```.
Check the examples in the following to know how to use the interface. (The interface also contains free space node but we don't use it for now in the current version.)

- Read a scene graph from a json file
```bash
python script/read_map.py
```

- Visualize a scene graph using the following command
```bash
python script/visualize_map.py --show_bboxes --show_edges
```
Add ```--map_ply_path xxx.ply --topology_map_path xxx.json``` to specify the data. Add ```--enable_picking``` to use the interactive mode, the name of a node will be printed when you press `shift` and left click the blue sphere of a node.
By default, example data in ```sample_data/scans/scene0000_00``` will be used. You will see an image like the following:

![image](/sample_data/scans/scene0000_00/scene_0000.png)



- Generate a random scene graph
```bash
python script/random_map_generator.py
```

### C++ version:

C++ data structure is defined in `include/topology_map.h` 
Check the example in the following to know how to use the C++ version interface.

To use the example, we recommend user to put this repo to a ROS1 workspace and run ```catkin build``` first to compile.

- Read and visualize a scene graph
```bash
./read_and_visualize_map <map_file>
```

ROS2 support is on the way. We only use the libraries and CMake in ROS to make the installation easier. No communication is established.


## Generate Addtional Data
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
