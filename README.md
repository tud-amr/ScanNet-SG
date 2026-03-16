# ScanNet-SG

This repository contains the code for the ScanNet-SG Dataset.


## Dataset Download

Coming soon...


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
Add ```--map_ply_path xxx.ply --topology_map_path xxx.json``` to specify the data. By default, example data in ```sample_data/scans/scene0000_00``` will be used.

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
Coming soon


## Liciense
Apache-2.0