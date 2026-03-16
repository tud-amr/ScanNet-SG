import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import sys
import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)

from shape import Cylinder, OrientedBox
import json
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    '''
    A custom JSON encoder that can handle numpy arrays.
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Node: 
    '''
    Common class for all nodes in the topology map.
    Currently, only contains a unique identifier for the node.
    '''
    def __init__(self, id: str):
        self.id = id
        # The following position is global and is only used for testing!!!
        # TODO: Remove this position
        self.position = np.array([0, 0, 0]) 

    @classmethod
    def from_dict(cls, data):
        node = cls(data['id'])
        # TODO: Remove this
        node.position = np.array(data['position'])
        return node

class ObjectNode(Node):
    '''
    An object node that represents an object in the environment.
    '''
    def __init__(
        self,
        id: str,
        name: str,
        visual_embedding: np.ndarray,
        text_embedding: np.ndarray,
        shape: Union[Cylinder, OrientedBox],
        position: np.ndarray
    ):
        super().__init__(id)
        self.name = name    
        self.visual_embedding = visual_embedding
        self.text_embedding = text_embedding
        self.shape = shape
        # The following position is global and is only used for testing!!!
        self.position = position

    def update_visual_embedding(self, visual_embedding: np.ndarray):
        self.visual_embedding = (visual_embedding + self.visual_embedding) / 2

    def update_text_embedding(self, text_embedding: np.ndarray):
        self.text_embedding = (text_embedding + self.text_embedding) / 2

    def update_shape(self, shape: Union[Cylinder, OrientedBox]):
        self.shape = shape
    
    def update_name(self, name: str):
        self.name = name

    @classmethod
    def from_dict(cls, data):
        shape_data = data['shape']
        if "radius" in shape_data:
            shape = Cylinder.from_dict(shape_data)
        else:
            shape = OrientedBox.from_dict(shape_data)
        
        return cls(
            id=data['id'],
            name=data['name'],
            visual_embedding=np.array(data['visual_embedding']),
            text_embedding=np.array(data['text_embedding']),
            shape=shape,
            position=np.array(data['position'])
        )
        

class FreeSpaceNode(Node):
    '''
    A free space node that represents a free space in the environment.
    '''
    def __init__(self, id: str, radius: float, position: np.ndarray):
        super().__init__(id)
        self.radius = radius
        # The following position is global and is only used for testing!!!
        self.position = position

    def update_radius(self, new_radius: float):
        self.radius = new_radius
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            radius=data['radius'],
            position=np.array(data['position'])
        )


class Edge:
    '''
    An edge that connects two nodes.
    '''
    def __init__(
        self,
        source_id: str,
        target_id: str,
        distance: float,
        direction: np.ndarray,  # 3D vector representing direction
        description: str
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.distance = distance
        self.direction = direction
        self.description = description

    def update_spatial_info(self, distance: float, direction: np.ndarray):
        self.distance = distance
        self.direction = direction

    def update_description(self, description: str):
        self.description = description

    @classmethod
    def from_dict(cls, data):
        return cls(
            source_id=data['source_id'],
            target_id=data['target_id'],
            distance=data['distance'],
            direction=np.array(data['direction']),
            description=data['description']
        )


class ObjectNodesHashTable:
    '''
    A hash_table of object nodes. Used to store all possible object nodes.
    '''
    def __init__(self):
        self.nodes: Dict[str, ObjectNode] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def get_node(self, id: str) -> Optional[Node]:
        return self.nodes.get(id)

    def remove_node(self, id: str):
        del self.nodes[id]


class FreespaceNodesHashTable:
    '''
    A hash table of free space nodes. Used to store all possible free space nodes.
    '''
    def __init__(self):
        self.nodes: Dict[str, FreeSpaceNode] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def get_node(self, id: str) -> Optional[Node]:
        return self.nodes.get(id)

    def remove_node(self, id: str):
        del self.nodes[id]
    

class TopologyMapHypothesis:
    '''
    A hypothesis of the topology map that stores one hypothesis of edges that connect the object nodes and free space nodes.
    '''
    def __init__(self, id: str, confidence: float):
        self.edges: Dict[str, Edge] = {}
        self.id = id
        self.confidence = confidence

    def add_edge(self, edge: Edge):
        id = f"{edge.source_id}-{edge.target_id}"
        self.edges[id] = edge

    def remove_edge(self, edge: Edge):
        id = f"{edge.source_id}-{edge.target_id}"
        del self.edges[id]

    def remove_edge_by_ids(self, source_id: str, target_id: str):
        id = f"{source_id}-{target_id}"
        del self.edges[id]
        
    def get_edge(self, id: str) -> Optional[Edge]:
        # ID should be in the format "source_id-target_id"
        return self.edges.get(id)
    
    def update_confidence(self, confidence: float):
        self.confidence = confidence
    

class TopologyMap:
    '''
    A topology map that stores all possible object nodes, free space nodes, and edge hypotheses.
    '''
    def __init__(self):
        self.object_nodes = ObjectNodesHashTable()
        self.free_space_nodes = FreespaceNodesHashTable()
        self.edge_hypotheses: Dict[str, TopologyMapHypothesis] = {}

    def add_edge_hypothesis(self, edge_hypothesis: TopologyMapHypothesis):
        self.edge_hypotheses[edge_hypothesis.id] = edge_hypothesis

    def remove_edge_hypothesis(self, edge_hypothesis: TopologyMapHypothesis):
        del self.edge_hypotheses[edge_hypothesis.id]

    def write_to_json(self) -> str:
        '''
        Write the topology map to a JSON string.
        '''
        return json.dumps(self, default=lambda o: o.__dict__ if not isinstance(o, np.ndarray) else o.tolist(), indent=4)
    
    def read_from_json(self, json_str: str):
        '''
        Read the topology map from a JSON string.
        '''
        data = json.loads(json_str)
        if data is None:
            print("Warning: JSON parsing returned None")
            return
        
        # Initialize empty hash tables
        self.object_nodes = ObjectNodesHashTable()
        self.free_space_nodes = FreespaceNodesHashTable()
        self.edge_hypotheses = {}

        # Handle object nodes
        if data.get('object_nodes') and data['object_nodes'].get('nodes'):
            for node_id, node_data in data['object_nodes']['nodes'].items():
                node = ObjectNode.from_dict(node_data)
                self.object_nodes.add_node(node)
        else:
            print("No object nodes found or object_nodes is null")

        # Handle free space nodes
        if data.get('free_space_nodes') and data['free_space_nodes'].get('nodes'):
            for node_id, node_data in data['free_space_nodes']['nodes'].items():
                node = FreeSpaceNode.from_dict(node_data)
                self.free_space_nodes.add_node(node)
        else:
            # print("No free space nodes found or free_space_nodes is null")
            pass

        # Handle edge hypotheses
        if data.get('edge_hypotheses'):
            for hypothesis_id, hypothesis_data in data['edge_hypotheses'].items():
                hypothesis = TopologyMapHypothesis(
                    hypothesis_id, 
                    hypothesis_data.get('confidence', 0.0)
                )
                if 'edges' in hypothesis_data:
                    for edge_id, edge_data in hypothesis_data['edges'].items():
                        edge = Edge.from_dict(edge_data)
                        hypothesis.add_edge(edge)
                self.add_edge_hypothesis(hypothesis)
        else:
            print("No edge hypotheses found or edge_hypotheses is null")

            
    # def visualize(self):
    ### Visualizzation is not easy because no global reference frame is available. Every thing is relative.
    #     '''
    #     Visualize the topology map.
    #     '''
    #     # Create a 3D plot
    #     fig = plt.figure()  
    
        