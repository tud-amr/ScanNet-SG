from include.topology_map import *
from include.shape import *
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

big_object_list = [
    "chair",
    "table",
    "desk",
    "couch",
    "bed",
]

small_object_list = [
    "lamp",
    "plant",
    "picture",
    "vase",
    "clock",
]

spatial_relations = [
    "next to",
    "on",
    "under",
    "above",
    "below"
]


def create_cylinder(ax, center, radius, height, color='blue', alpha=0.3):
    '''
    Create a cylinder in the 3D plot.
    '''
    # Create the cylinder
    z = np.linspace(0, height, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    
    # Draw the surface
    ax.plot_surface(x_grid, y_grid, z_grid + center[2], color=color, alpha=alpha)
    
    # Draw the circles at top and bottom
    circle_bottom = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    circle_top = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    
    ax.add_patch(circle_bottom)
    art3d.pathpatch_2d_to_3d(circle_bottom, z=center[2])
    
    ax.add_patch(circle_top)
    art3d.pathpatch_2d_to_3d(circle_top, z=center[2] + height)

def is_circle_overlapping(center1: tuple[float, float], radius1: float, center2: tuple[float, float], radius2: float):
    '''
    Check if two circles are overlapping.
    '''
    return (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2 < (radius1 + radius2)**2


def generate_circles_with_no_overlap(num_circles: int, radius_range: list[float], room_range: list[float], max_attempts: int = 100):
    '''
    Generate a list of circles with no overlap.
    '''
    centers = []
    radiuses = []

    added_circles = 0
    for i in range(max_attempts):
        center = np.array([random.uniform(0, room_range[0]), random.uniform(0, room_range[1])])
        radius = random.uniform(radius_range[0], radius_range[1])

        # Check if the circle is overlapping with any existing circles
        if any(is_circle_overlapping(center, radius, existing_center, existing_radius) for existing_center, existing_radius in zip(centers, radiuses)):
            continue

        centers.append(center)
        radiuses.append(radius)
        added_circles += 1

        if added_circles == num_circles:
            break

    return centers, radiuses


def add_gaussian_noise_by_percentage(value: float, percentage: float = 0.1):
    '''
    Add gaussian noise to the value by a percentage.
    '''
    stddev = abs(value) * percentage
    return value + np.random.normal(0, stddev)


def generate_random_map(num_big_objects: int, num_small_objects: int, room_range: list[float], free_space_nodes_num: int, distance_threshold_for_edges: float, visualize: bool = False) -> TopologyMap:
    '''
    Generate a random topology map.
    '''
    # Set the parameters for the random map
    big_objects_num = num_big_objects
    big_object_radius_range = [1, 2]
    big_object_height_range = [0.5, 1]
    
    small_objects_num = num_small_objects # number of small objects should be less than the number of big objects
    small_object_radius_range = [0.1, 0.2]
    small_object_height_range = [0.1, 0.2]
    
    topology_map = TopologyMap()

    ''' Nodes generation '''
    # These two lists store the positions used only for edge generation!
    object_node_positions_dict = {}
    free_space_node_positions_dict = {}

    # Generate a random layout for the big objects
    big_objects_centers, big_objects_radiuses = generate_circles_with_no_overlap(big_objects_num, big_object_radius_range, room_range)
    big_objects_heights = [random.uniform(big_object_height_range[0], big_object_height_range[1]) for _ in range(len(big_objects_centers))]
    big_objects_num = len(big_objects_centers) # correct the number of big objects

    # Make centers 3D by adding a z-coordinate of 0
    big_objects_centers = [np.array([center[0], center[1], 0]) for center in big_objects_centers]

    print(f"big_objects_centers: {big_objects_centers}")
    print(f"big_objects_radiuses: {big_objects_radiuses}")
    print(f"big_objects_heights: {big_objects_heights}")
    print(f"valid big_objects_num: {big_objects_num}")

    if small_objects_num >= big_objects_num:
        raise ValueError("Number of small objects should be less than the number of big objects")

    # Add big objects to the topology map
    for i in range(big_objects_num):
        big_object_node = ObjectNode(
            id=f"big_object_{i}",
            name=random.choice(big_object_list),
            shape=Cylinder(radius=big_objects_radiuses[i], height=big_objects_heights[i], orientation=Orientation(0, 0, 0, 1)),
            visual_embedding = i * np.ones(100) * 0.01,
            text_embedding = i * np.ones(100) * 0.01,
            position = big_objects_centers[i]
        )
        object_node_positions_dict[big_object_node.id] = big_objects_centers[i]
        topology_map.object_nodes.add_node(big_object_node)
        
    # Generate a random layout for the small objects. Small objects are placed either on or above the big objects.
    small_objects_radiuses = [random.uniform(small_object_radius_range[0], small_object_radius_range[1]) for _ in range(small_objects_num)]
    small_objects_heights = [random.uniform(small_object_height_range[0], small_object_height_range[1]) for _ in range(small_objects_num)]
    print(f"small_objects_radiuses: {small_objects_radiuses}")
    print(f"small_objects_heights: {small_objects_heights}")
    print(f"small_objects_num: {small_objects_num}")

    # Add small objects to the topology map
    for i in range(small_objects_num):
        small_object_position = big_objects_centers[i] + np.array([0, 0, big_objects_heights[i]])
        small_object_node = ObjectNode(
            id=f"small_object_{i}",
            name=random.choice(small_object_list),
            shape=Cylinder(radius=small_objects_radiuses[i], height=small_objects_heights[i], orientation=Orientation(0, 0, 0, 1)),
            visual_embedding = i * np.ones(100) * 0.015,
            text_embedding = i * np.ones(100) * 0.015,
            position = small_object_position
        )
        
        # We assume that small objects are placed on or above the big objects
        topology_map.object_nodes.add_node(small_object_node)

        object_node_positions_dict[small_object_node.id] = small_object_position

    print(f"Generated {len(topology_map.object_nodes.nodes)} object nodes")

    # Generate free space nodes. Randomly generate a number of free space nodes that are not overlapping with the big objects.
    attempts = 0
    free_space_node_count = 0
    max_attempts = 1000
    while attempts < max_attempts:
        if free_space_node_count >= free_space_nodes_num or attempts >= max_attempts:
            break

        position = (random.uniform(0, room_range[0]), random.uniform(0, room_range[1]), 0)
        radius = random.uniform(0.1, 0.5)
        # Check if the free space node is overlapping with any big objects
        overlapping = False
        for j in range(len(big_objects_centers)):
            if is_circle_overlapping(position, radius, big_objects_centers[j], big_objects_radiuses[j]):
                overlapping = True
                break
        if overlapping:
            attempts += 1
            continue

        free_space_node = FreeSpaceNode(id=f"free_space_{free_space_node_count}", radius=radius, position=position)
        topology_map.free_space_nodes.add_node(free_space_node)

        # Record the free space node for edge generation
        free_space_node_positions_dict[free_space_node.id] = position
        free_space_node_count += 1

    print(f"Generated {len(topology_map.free_space_nodes.nodes)} free space nodes")


    '''Edges generation'''
    edge_hypothesis = TopologyMapHypothesis(
        id=f"hypothesis_0",
        confidence=1.0
    )

    # Generate edges between all topology_map.object_nodes if their distance is less than the threshold    
    for node_id_i, node_i in topology_map.object_nodes.nodes.items():
        for node_id_j, node_j in topology_map.object_nodes.nodes.items():
            if node_id_i == node_id_j:
                continue

            distance = np.linalg.norm(np.array(object_node_positions_dict[node_id_i]) - np.array(object_node_positions_dict[node_id_j]))
            if distance < distance_threshold_for_edges:
                source_id = node_i.id
                target_id = node_j.id
                epsilon = 1e-10
                if distance < epsilon:
                    # Handle the case where nodes are at the same position
                    direction = np.zeros_like(object_node_positions_dict[node_id_j])  # or some default direction
                else:
                    direction = (object_node_positions_dict[node_id_j] - object_node_positions_dict[node_id_i]) / distance
                
                if "small" in source_id and "small" in target_id:
                    description = "next to"
                elif "small" in source_id and "big" in target_id:
                    description = "on"
                elif "big" in source_id and "small" in target_id:
                    description = "under"
                elif "big" in source_id and "big" in target_id:
                    description = "next to"
                else:
                    raise ValueError("Invalid object types")
                
                # Add gaussian noise to the direction and distance
                direction = add_gaussian_noise_by_percentage(direction, 0.1)
                distance = add_gaussian_noise_by_percentage(distance, 0.1)

                edge = Edge(source_id=source_id, target_id=target_id, distance=distance, direction=direction, description=description)
                edge_hypothesis.add_edge(edge)

    print(f"Generated {len(edge_hypothesis.edges)} edges for {len(topology_map.object_nodes.nodes)} objects")

    # Generate edges between all topology_map.object_nodes and topology_map.free_space_nodes if their distance is less than the threshold
    for node_id_i, node_i in topology_map.object_nodes.nodes.items():
        for node_id_j, node_j in topology_map.free_space_nodes.nodes.items():
            distance = np.linalg.norm(np.array(object_node_positions_dict[node_id_i]) - np.array(free_space_node_positions_dict[node_id_j]))
            if distance < distance_threshold_for_edges:
                source_id = node_i.id
                target_id = node_j.id
                epsilon = 1e-10
                if distance < epsilon:
                    # Handle the case where nodes are at the same position
                    direction = np.zeros_like(object_node_positions_dict[node_id_j])  # or some default direction
                else:
                    direction = (free_space_node_positions_dict[node_id_j] - object_node_positions_dict[node_id_i]) / distance
                
                # Add gaussian noise to the direction and distance
                direction = add_gaussian_noise_by_percentage(direction, 0.1)
                distance = add_gaussian_noise_by_percentage(distance, 0.1)

                edge = Edge(source_id=source_id, target_id=target_id, distance=distance, direction=direction, description="next to")
                edge_hypothesis.add_edge(edge)

    print(f"Generated {len(edge_hypothesis.edges)} edges for {len(topology_map.object_nodes.nodes)} objects and {len(topology_map.free_space_nodes.nodes)} free space nodes")

    # Generate edges between all topology_map.free_space_nodes if their distance is less than the threshold and the edge doesn't collide with any big objects
    for node_id_i, node_i in topology_map.free_space_nodes.nodes.items():
        for node_id_j, node_j in topology_map.free_space_nodes.nodes.items():
            if node_id_i == node_id_j:
                continue

            distance = np.linalg.norm(np.array(free_space_node_positions_dict[node_id_i]) - np.array(free_space_node_positions_dict[node_id_j]))
            if distance < distance_threshold_for_edges:
                source_id = node_i.id
                target_id = node_j.id

                # Check if the edge collides with any big objects
                overlapping = False
                for k in range(len(big_objects_centers)):
                    if is_circle_overlapping(free_space_node_positions_dict[node_id_i], node_i.radius, big_objects_centers[k], big_objects_radiuses[k]):
                        overlapping = True
                        break
                if overlapping:
                    continue
                
                # Add gaussian noise to the direction and distance
                direction = add_gaussian_noise_by_percentage(direction, 0.1)
                distance = add_gaussian_noise_by_percentage(distance, 0.1)

                edge = Edge(source_id=source_id, target_id=target_id, distance=distance, direction=direction, description="next to")
                edge_hypothesis.add_edge(edge)

    print(f"Generated {len(edge_hypothesis.edges)} edges for {len(topology_map.free_space_nodes.nodes)} free space nodes")

    topology_map.add_edge_hypothesis(edge_hypothesis)

    ''' Visualization '''
    # Use the positions here to visualize the topology map
    if visualize:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the object nodes
        for object_node in topology_map.object_nodes.nodes.values():
            object_position = object_node_positions_dict[object_node.id]
            if isinstance(object_node.shape, Cylinder):
                create_cylinder(ax, object_position, object_node.shape.radius, object_node.shape.height, color='red', alpha=0.3)
            elif isinstance(object_node.shape, OrientedBox):
                ax.scatter(object_position[0], object_position[1], object_position[2], c='blue', marker='s')

        # Plot the free space nodes
        for free_space_node in topology_map.free_space_nodes.nodes.values():
            free_space_position = free_space_node_positions_dict[free_space_node.id]
            ax.scatter(free_space_position[0], free_space_position[1], free_space_position[2], c='green', marker='x')

        # Plot the edges
        for edge in edge_hypothesis.edges.values():
            source_id = edge.source_id
            target_id = edge.target_id
            
            if "free" in source_id:
                source_position = free_space_node_positions_dict[source_id]
            else:
                source_position = object_node_positions_dict[source_id]

            if "free" in target_id:
                target_position = free_space_node_positions_dict[target_id]
            else:
                target_position = object_node_positions_dict[target_id]

            ax.plot([source_position[0], target_position[0]], [source_position[1], target_position[1]], [source_position[2], target_position[2]], c='black')
        # plt.savefig('topology_map.png')
        plt.show()

    return topology_map

if __name__ == "__main__":
    ## generate a random topology map and save it to a json file
    topology_map = generate_random_map(10, 5, [20, 10], 40, 4.0, visualize=True)
    with open("topology_map.json", "w") as f:
        f.write(topology_map.write_to_json())

    
