from include.topology_map import *

if __name__ == "__main__":
    ## Load the topology map from a json file
    #json_path = "topology_map.json"
    json_path = "/media/cc/My Passport/dataset/scannet/processed/scans/scene0000_00/topology_map.json"
    with open(json_path, "r") as f:
        topology_map = TopologyMap()
        topology_map.read_from_json(f.read())
        # Print all the hypotheses and the first object node
        for hypothesis in topology_map.edge_hypotheses.values():
            print(f"Hypothesis {hypothesis.id} has {len(hypothesis.edges)} edges")
        for node in topology_map.object_nodes.nodes.values():
            print(f"Object node {node.id} name: {node.name}")
