// utils.h
#pragma once

#include "topology_map.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

void showTopologyMap(const TopologyMap& map, const std::string& hypothesis_id = "default_hypothesis")
{   

    /// TODO: Change the visualization to use relative positions rather than absolute positions

    pcl::visualization::PCLVisualizer viewer("Topology Map");

    // Define a lambda function to add lines to the viewer
    auto addLine = [&](Eigen::Vector3f p1, Eigen::Vector3f p2, float r = 1.0, float g = 0.0, float b = 0.0, std::string id = "line", double width = 0.01) {
        pcl::PointXYZ p1_pcl(p1[0], p1[1], p1[2]);
        pcl::PointXYZ p2_pcl(p2[0], p2[1], p2[2]);
        viewer.addLine(p1_pcl, p2_pcl, r, g, b, id, width);
    };

    std::unordered_map<std::string, Eigen::Vector3f> id_to_node_position_map;

    // Visualize objects's shapes and centers
    for (const auto& [id, node] : map.object_nodes) {
        id_to_node_position_map[id] = node.position;

        if (node.shape) {

            Eigen::Vector3f center = node.position;
            Eigen::Vector3f box_size;
            Eigen::Matrix3f rotation_matrix;

            if (node.shape->type() == ShapeType::CYLINDER) {
                const auto& cylinder = dynamic_cast<const Cylinder&>(*node.shape);
                float radius = cylinder.radius;
                float height = cylinder.height;

                // Turn the cylinder into a oriented bounding box
                box_size << radius*2, radius*2, height;
                rotation_matrix = cylinder.orientation.toRotationMatrix();
                
            }else if(node.shape->type() == ShapeType::ORIENTED_BOX){
                const auto& box = dynamic_cast<const OrientedBox&>(*node.shape);
                box_size << box.length, box.width, box.height;
                rotation_matrix = box.orientation.toRotationMatrix();
                
            }else{
                continue;
            }

            // Get the 8 corners of the box to show in the viewer
            Eigen::Vector3f half_size = 0.5f * box_size;
            std::vector<Eigen::Vector3f> corners_local = {
                {-half_size[0], -half_size[1], -half_size[2]},
                { half_size[0], -half_size[1], -half_size[2]},
                { half_size[0],  half_size[1], -half_size[2]},
                {-half_size[0],  half_size[1], -half_size[2]},
                {-half_size[0], -half_size[1],  half_size[2]},
                { half_size[0], -half_size[1],  half_size[2]},
                { half_size[0],  half_size[1],  half_size[2]},
                {-half_size[0],  half_size[1],  half_size[2]},
            };

            // Transform corners to world coordinates
            std::vector<Eigen::Vector3f> corners_world;
            for (const auto& c : corners_local) {
                corners_world.push_back(rotation_matrix * c + center);
            }

            // Draw box edges
            std::vector<std::pair<int, int>> edges = {
                {0,1},{1,2},{2,3},{3,0},
                {4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}
            };
            int box_edge_id = 0;
            for (auto [i, j] : edges) {
                Eigen::Vector3f p1 = corners_world[i];
                Eigen::Vector3f p2 = corners_world[j];
                std::string edge_id = "box_edge_" + id + "_" + std::to_string(box_edge_id);
                addLine(p1, p2, 1.0, 0, 0, edge_id);
                box_edge_id ++;
            }

            // Add center sphere
            pcl::PointXYZ sphere_center(center[0], center[1], center[2]);
            std::string sphere_id = "sphere_" + id;
            viewer.addSphere(sphere_center, 0.05, 0.0, 1.0, 0.0, sphere_id);

            // Add text label of object name
            pcl::PointXYZ text_position(center[0], center[1], center[2] + 0.2);
            Eigen::Vector3f euler_angles = rotation_matrix.eulerAngles(0, 1, 2);
            double orientation_array[3] = {euler_angles[0], euler_angles[1], euler_angles[2]};
            std::string text_id = "text_" + id;
            std::string text_to_show = node.id + " " + node.name; 

            viewer.addText3D(text_to_show, text_position, orientation_array, 0.05, 1.0, 0.0, 0.0, text_id);
        }
    }

    // Visualize free space nodes
    for (const auto& [id, node] : map.free_space_nodes) {
        id_to_node_position_map[id] = node.position;

        pcl::PointXYZ sphere_center(node.position[0], node.position[1], node.position[2]);
        std::string sphere_id = "free_space_sphere_" + id;
        viewer.addSphere(sphere_center, 0.1, 0.0, 0.0, 1.0, sphere_id);
    }

    // Visualize edge hypotheses with the given hypothesis id
    for (const auto& [id, hypothesis] : map.edge_hypotheses) {
        if (id == hypothesis_id) {
            for (const auto& [edge_id, edge] : hypothesis.edges) {
                std::string source_id = edge.source_id;
                std::string target_id = edge.target_id;

                Eigen::Vector3f source_position = id_to_node_position_map[source_id];
                Eigen::Vector3f target_position = id_to_node_position_map[target_id];

                // Draw a white line between the source and target positions
                std::string line_id = "edge_" + edge_id;
                addLine(source_position, target_position, 1.0, 1.0, 1.0, line_id);
            }
        }
    }
    
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}



