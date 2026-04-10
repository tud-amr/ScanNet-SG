/*
 * Author: Clarence Chen
 * This script is used to generate the ply map for a scene from the openset annotation.
 */

#include "read_instance.h"
#include <boost/filesystem.hpp>
#include <iomanip>
#include <json/single_include/nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <random>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward{
    backward::SignalHandling sh;
}

using json = nlohmann::json;

struct InstanceInfo {
    int frame_id;
    int instance_id;
    int frame_instance_id;
    std::string object_name;
    std::string object_description;
    float confidence;
    std::vector<float> feature;
    std::vector<float> bert_embedding;
};

// Global instance id and accumulated points
struct GlobalInstance {
    int global_id;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    std::vector<float> bert_embedding;
    std::vector<float> avg_feature;
    std::string object_name;
    std::string object_description;
    float confidence;
    int count;
    int largest_point_count;

    GlobalInstance() : 
        global_id(-1),
        accumulated_points(new pcl::PointCloud<pcl::PointXYZ>()),
        bert_embedding(),
        avg_feature(),
        object_name(),
        object_description(),
        confidence(0.0f),
        count(1),
        largest_point_count(0) {
    }
};

int next_global_id = 1;
std::vector<GlobalInstance> global_instances;
std::vector<std::vector<InstanceInfo>> updated_instance_frames;

// Cosine similarity of BERT features
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6);
}


std::vector<float> compute3DMIoU(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud1,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud2,
    float dist_threshold = 0.03f)
{
    if (cloud1->empty() || cloud2->empty()) {
        return std::vector<float>{0.0f, 0.0f, 0.0f};
    }

    // Build KD-trees for both clouds
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1, kdtree2;
    kdtree1.setInputCloud(cloud1);
    kdtree2.setInputCloud(cloud2);

    std::vector<int> indices;
    std::vector<float> dists;

    int matched1_count = 0, matched2_count = 0;

    // Check each point in cloud1 against cloud2
    for (size_t i = 0; i < cloud1->size(); ++i) {
        if (kdtree2.radiusSearch(cloud1->at(i), dist_threshold, indices, dists) > 0) {
            matched1_count++;
        }
    }

    // Check each point in cloud2 against cloud1
    for (size_t i = 0; i < cloud2->size(); ++i) {
        if (kdtree1.radiusSearch(cloud2->at(i), dist_threshold, indices, dists) > 0) {
            matched2_count++;
        }
    }

    int intersection = matched1_count + matched2_count;
    int union_size = cloud1->size() + cloud2->size() - matched1_count - matched2_count;

    std::vector<float> miou_vector;
    miou_vector.push_back(union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f);
    miou_vector.push_back(static_cast<float>(intersection) / cloud1->size());
    miou_vector.push_back(static_cast<float>(intersection) / cloud2->size());
    return miou_vector;
}



int matchInstanceToGlobal(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& instance_cloud,
                         const std::vector<float>& embedding,
                         const std::vector<GlobalInstance>& global_instances,
                         float sim_threshold = 0.8f,
                         float miou_threshold = 0.3f) {
    if (!instance_cloud || instance_cloud->points.size() < 5 || embedding.empty() || global_instances.empty()) {
        return -1;
    }

    for (const auto& pt : *instance_cloud) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            std::cerr << "Invalid point coordinates!" << std::endl;
            break;
        }
    }

    float best_score = -1.0f;
    int best_id = -1;

    for (const auto& global_inst : global_instances) {
        if (!global_inst.accumulated_points || global_inst.accumulated_points->empty()) {
            continue;
        }

        float sim = cosineSimilarity(embedding, global_inst.bert_embedding);
        if (sim < sim_threshold) continue;

        std::vector<float> miou = compute3DMIoU(instance_cloud, global_inst.accumulated_points);
        if (miou[0] < miou_threshold) continue;

        float score = 0.5f * sim + 0.5f * miou[0];
        if (score > best_score) {
            best_score = score;
            best_id = global_inst.global_id;
        }
    }

    return best_id;
}


void saveInstanceJson(const std::vector<InstanceInfo>& data, const std::string& filename) {
    nlohmann::json j;
    for (const auto& inst : data) {
        if (inst.instance_id == -1) continue;
        nlohmann::json item;
        item["instance_id"] = inst.instance_id;
        item["frame_instance_id"] = inst.frame_instance_id;
        item["object_name"] = inst.object_name;
        item["object_description"] = inst.object_description;
        item["confidence"] = inst.confidence;
        item["feature"] = inst.feature;
        item["bert_embedding"] = inst.bert_embedding;
        j.push_back(item);
    }
    std::ofstream(filename) << j.dump(2);
}

void saveFeatureJson(const std::vector<GlobalInstance>& global_instances, const std::string& filename) {
    nlohmann::json j;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        nlohmann::json item;
        item["instance_id"] = instance.global_id;
        item["occurance"] = instance.count;
        item["feature"] = instance.avg_feature;
        j.push_back(item);
    }
    std::ofstream(filename) << j.dump(2);
}

void saveBertEmbeddingJson(const std::vector<GlobalInstance>& global_instances, const std::string& filename) {
    nlohmann::json j;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        std::string instance_id = std::to_string(instance.global_id);
        j[instance_id] = instance.bert_embedding;
    }
    std::ofstream(filename) << j.dump(2);
}

int main(int argc, char** argv) {
     // Default parameters
    std::string scene_name = "scene0702_00";
    int if_visualize = 1;
    float max_depth = 0.0f;  // 0 means no depth filtering
    int subsample_factor = 1;  // 1 means read all rows/cols

    std::string raw_images_parent_dir = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/images/scans/";
    std::string refined_instance_parent_dir = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/openset_scans/";
  
    if (argc == 1) {
        // Default parameters
    }else if (argc == 2) {
        scene_name = argv[1];
    }else if (argc == 3) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
    }else if (argc == 5) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
        refined_instance_parent_dir = argv[3];
        raw_images_parent_dir = argv[4];
    }else if (argc == 7) {
        scene_name = argv[1];
        if_visualize = std::stoi(argv[2]);
        refined_instance_parent_dir = argv[3];
        raw_images_parent_dir = argv[4];
        max_depth = std::stof(argv[5]);
        subsample_factor = std::stoi(argv[6]);
    }else{
        std::cout << "Usage: " << argv[0] << " <scene_name> <if_visualize 0 or 1> <refined_instance_parent_dir> <raw_images_parent_dir> [max_depth] [subsample_factor]" << std::endl;
        std::cout << "  max_depth: Maximum depth threshold in meters (0 = no filtering)" << std::endl;
        std::cout << "  subsample_factor: Subsample factor for rows/cols (1 = read all, 2 = read every other row/col, etc.)" << std::endl;
        return 1;
    }

    // Add "/" to the end of the refined_instance_parent_dir if not exists
    if (refined_instance_parent_dir.back() != '/') {
        refined_instance_parent_dir += "/";
    }
    if (raw_images_parent_dir.back() != '/') {
        raw_images_parent_dir += "/";
    }

    std::cout << "refined_instance_parent_dir: " << refined_instance_parent_dir << std::endl;
    std::cout << "raw_images_parent_dir: " << raw_images_parent_dir << std::endl;
    std::cout << "scene_name: " << scene_name << std::endl;
    std::cout << "if_visualize: " << if_visualize << std::endl;
    std::cout << "max_depth: " << max_depth << " meters (0 = no filtering)" << std::endl;
    std::cout << "subsample_factor: " << subsample_factor << std::endl;

    std::string output_ply_dir = refined_instance_parent_dir + scene_name + "/";
    
    std::string meta_file = raw_images_parent_dir + scene_name + "/_info.txt";
    
    InstanceCloudGenerator cloud_generator(meta_file);

    // Find all the frame id in the refined_instance_parent_dir
    std::vector<int> frame_id_list;
    for (const auto& entry : boost::filesystem::directory_iterator(refined_instance_parent_dir + scene_name + "/refined_instance/")) {
        if (boost::filesystem::is_regular_file(entry.path()) && entry.path().extension() == ".png") {
            // Get the instance id. Name of the png is like id.png
            std::string filename = entry.path().stem().string();
            int frame_id = std::stoi(filename);
            frame_id_list.push_back(frame_id);
        }
    }

    if (if_visualize) std::cout << "Found " << frame_id_list.size() << " frames" << std::endl;

    // Sort the frame id list
    std::sort(frame_id_list.begin(), frame_id_list.end());
    
    // Process the frames one by one
    for (int frame : frame_id_list) {
        if (if_visualize) std::cout << "*** Processing frame " << frame << std::endl;

        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << frame;
        std::string six_digits_frame = oss.str();
        std::string original_digits_frame = std::to_string(frame);
        
        std::string depth_path = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".depth.pgm";
        std::string instance_path = refined_instance_parent_dir + scene_name + "/refined_instance/" + original_digits_frame + ".png";
        std::string pose_path = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".pose.txt";
        std::string json_path = refined_instance_parent_dir + scene_name + "/refined_instance/" + original_digits_frame + "_instance.json";
        
        if (!boost::filesystem::exists(instance_path)) {
            std::cout << "*** Instance file does not exist: " << instance_path << "Will try the next frame" << std::endl;
            continue;
        }
        
        if (!boost::filesystem::exists(depth_path)) {
            std::cout << "*** Depth file does not exist: " << depth_path << std::endl;
            return 1;
        }
        
        if (!boost::filesystem::exists(pose_path)) {
            std::cout << "*** Pose file does not exist: " << pose_path << std::endl;
            return 1;
        }

        if (!boost::filesystem::exists(json_path)) {
            std::cout << "*** JSON file does not exist: " << json_path << std::endl;
            return 1;
        }

        pcl::PointCloud<pcl::PointXYZRGB> cloud_instances;
        cloud_generator.processFrame(depth_path, instance_path, pose_path, cloud_instances, true, true, false, max_depth, subsample_factor);

        // Load the json file
        std::ifstream json_file(json_path);
        json json_data;
        json_file >> json_data;

        std::vector<InstanceInfo> instances;
        for (const auto& item : json_data) {
            InstanceInfo inst;
            inst.instance_id = item["instance_id"];
            inst.frame_instance_id = item["frame_instance_id"];
            inst.object_name = item["object_name"];
            inst.object_description = item["object_description"];
            inst.confidence = item.value("confidence", 0.0f);
            inst.feature = item.value("feature", std::vector<float>{});
            inst.bert_embedding = item.value("bert_embedding", std::vector<float>{});
            inst.frame_id = frame;
            instances.push_back(inst);
        }

        // Process the instances one by one
        for (auto& inst : instances) {
            // Extract point cloud segment for this instance
            pcl::PointCloud<pcl::PointXYZ>::Ptr instance_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& pt : cloud_instances) {
                if (pt.r == inst.frame_instance_id) { //In one frame, Max id is 255
                    instance_cloud->push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                }
            }
            if (!instance_cloud) {
                std::cerr << "[DIAG] instance_cloud is null!" << std::endl;
                continue;
            }
            if (instance_cloud->empty()) continue;
            if (inst.bert_embedding.empty()) continue;

            // Try to match with global instances
            /// TODO: Tune the parameters for matching
            int matched_id = matchInstanceToGlobal(instance_cloud, inst.bert_embedding, global_instances);

            if (matched_id == -1) {
                // No match → assign new global ID
                GlobalInstance new_global;
                new_global.global_id = next_global_id++;
                new_global.accumulated_points->points = instance_cloud->points;
                new_global.bert_embedding = inst.bert_embedding;
                new_global.avg_feature = inst.feature;
                new_global.object_name = inst.object_name;
                new_global.object_description = inst.object_description;
                new_global.confidence = inst.confidence;
                new_global.count = 1;
                new_global.largest_point_count = instance_cloud->points.size();
                global_instances.push_back(new_global);
                matched_id = new_global.global_id;

                if (if_visualize) std::cout << "*** No match → assign new global ID: " << new_global.global_id << std::endl;
            } else {
                // Update existing global instance by adding the instance points to the global instance
                for (auto& g : global_instances) {
                    if (g.global_id == matched_id) {
                        for (const auto& pt : instance_cloud->points) {
                            g.accumulated_points->points.push_back(pt);
                        }
                        // Update the avg visual language feature by averaging the feature of the instance
                        for (size_t i = 0; i < g.avg_feature.size(); ++i) {
                            g.avg_feature[i] = (g.avg_feature[i] * g.count + inst.feature[i]) / (g.count + 1);
                        }
                        // Update the bert embedding and object name, description, confidence if the instance has more points
                        if (g.largest_point_count < instance_cloud->points.size()) {
                            g.largest_point_count = instance_cloud->points.size();
                            g.bert_embedding = inst.bert_embedding;
                            g.object_name = inst.object_name;
                            g.object_description = inst.object_description;
                            g.confidence = inst.confidence;
                        }

                        g.count += 1;
                        break;
                    }

                    // If the points are too many, we do downsampling by skipping one point every 2 points
                    if (g.accumulated_points && g.accumulated_points->points.size() > 10000) {
                        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                        for (size_t i = 0; i < g.accumulated_points->points.size(); i += 2) {
                            downsampled_cloud->points.push_back(g.accumulated_points->points[i]);
                        }
                        *g.accumulated_points = *downsampled_cloud;
                    }
                }
                if (if_visualize) std::cout << "*** Update existing global instance: " << matched_id << std::endl;
            }

            // Update the instance id
            inst.instance_id = matched_id;
        }

        // Update the updated_instance_frames
        updated_instance_frames.push_back(instances);
    }

    // Compute the center of the global instances for quick filtering
    std::vector<float> global_instances_center_x;
    std::vector<float> global_instances_center_y;
    std::vector<float> global_instances_center_z;
    for (auto& instance : global_instances) {
        float center_x = 0.0f;
        float center_y = 0.0f;
        float center_z = 0.0f;
        for (auto& pt : *instance.accumulated_points) {
            center_x += pt.x;
            center_y += pt.y;
            center_z += pt.z;   
        }
        center_x /= instance.accumulated_points->points.size();
        center_y /= instance.accumulated_points->points.size();
        center_z /= instance.accumulated_points->points.size();
        global_instances_center_x.push_back(center_x);
        global_instances_center_y.push_back(center_y);
        global_instances_center_z.push_back(center_z);
    }

    // Further merge the global instances by cosine similarity and 3D MIoU
    if (if_visualize) std::cout << "*** Further merge the global instances by cosine similarity and 3D MIoU" << std::endl;
    
    std::unordered_map<int, int> merge_map;
    for (size_t i = 0; i < global_instances.size(); ++i) {
        GlobalInstance& instance = global_instances[i];
        for (size_t j = 0; j < global_instances.size(); ++j) {
            if (i==j) continue;

            GlobalInstance& other_instance = global_instances[j];
            if (instance.global_id == other_instance.global_id || other_instance.global_id == -1 || instance.global_id == -1) continue;

            float distance = std::sqrt(
                (global_instances_center_x[i] - global_instances_center_x[j]) * (global_instances_center_x[i] - global_instances_center_x[j]) +
                (global_instances_center_y[i] - global_instances_center_y[j]) * (global_instances_center_y[i] - global_instances_center_y[j]) +
                (global_instances_center_z[i] - global_instances_center_z[j]) * (global_instances_center_z[i] - global_instances_center_z[j])
            );
            if (distance > 2.0f) continue; // Filter out the instances that are too far away to accelerate the merging

            float cosine_similarity = cosineSimilarity(instance.bert_embedding, other_instance.bert_embedding);
            std::vector<float> miou = compute3DMIoU(instance.accumulated_points, other_instance.accumulated_points, 0.03f);
            if (miou[0] > 0.7f || (cosine_similarity > 0.7f && (miou[0] > 0.3f || miou[1] > 0.8f || miou[2] > 0.8f))) {
                if (if_visualize) {
                    std::cout << "*** Merging the two instances: " << instance.global_id << " and " << other_instance.global_id << std::endl;
                    std::cout << "*** The merging instance name is " << instance.object_name << " with " << other_instance.object_name << std::endl;
                }
                
                // Merge the two instances
                for (auto& pt : other_instance.accumulated_points->points) {
                    instance.accumulated_points->points.push_back(pt);
                }   
                
                // Update the bert embedding and object name, description, confidence if the instance has more points
                if (instance.largest_point_count < other_instance.largest_point_count) {
                    instance.bert_embedding = other_instance.bert_embedding;
                    instance.object_name = other_instance.object_name;
                    instance.object_description = other_instance.object_description;
                    instance.confidence = other_instance.confidence;
                }
                if (if_visualize) std::cout << "merged instance name: " << instance.object_name << std::endl;

                // Update the avg visual language feature by averaging the feature of the instance
                for (size_t i = 0; i < instance.avg_feature.size(); ++i) {
                    instance.avg_feature[i] = (instance.avg_feature[i] * instance.count + other_instance.avg_feature[i]) / (instance.count + 1);
                }

                instance.count += other_instance.count;
                instance.largest_point_count = std::max(instance.largest_point_count, other_instance.largest_point_count);
                merge_map[other_instance.global_id] = instance.global_id;
                other_instance.global_id = -1;
            }
        }
    }

    // Remove the global instances that is observed less than 2 times
    if (if_visualize) std::cout << "*** Remove the global instances that is observed less than 2 times" << std::endl;
    int removed_instances = 0;
    for (auto& instance : global_instances) {
        if (instance.count < 2) {
            merge_map[instance.global_id] = -1; // Will not be saved to jsons
            instance.global_id = -1;
            removed_instances += 1;
        }
    }

    // std::cout << "*** Global instances size: " << global_instances.size() << std::endl;
    // std::cout << "*** Removed " << removed_instances << " instances" << std::endl;

    // Update the instance id in the updated_instance_frames
    if (if_visualize) std::cout << "*** Update the instance id in the updated_instance_frames" << std::endl;
    for (auto& frame : updated_instance_frames) {
        for (auto& inst : frame) {
            // If the instance id is in the merge_map, update it iteratively until it is not in the merge_map
            int id = inst.instance_id;
            while (merge_map.find(id) != merge_map.end()) {
                int new_id = merge_map[id];
                id = new_id;
                if (new_id == -1) break;
                // std::cout << "Update the instance id: " << id << " to " << new_id << std::endl;
            }
            inst.instance_id = id;
        }
    }

    // Save the instance name map to a csv
    std::ofstream instance_name_map_file(output_ply_dir + "instance_name_map.csv");
    instance_name_map_file << "instance_id,name" << std::endl;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        instance_name_map_file << instance.global_id << "," << instance.object_name << std::endl;
    }
    instance_name_map_file.close();

    // Save the instance id, count, and visual language feature to a json file
    if (if_visualize) std::cout << "*** Save the instance id, count, and visual language feature to a json file" << std::endl;
    saveFeatureJson(global_instances, output_ply_dir + "averaged_instance_features.json");
    saveBertEmbeddingJson(global_instances, output_ply_dir + "instance_bert_embeddings.json");

    // Save the updated_instance_frames to a new json file by each frame
    if (if_visualize) std::cout << "*** Save the updated_instance_frames to a new json file by each frame" << std::endl;
    for (auto& frame : updated_instance_frames) {
        if (frame.empty()) continue;
        std::string save_path = refined_instance_parent_dir + scene_name + "/refined_instance/" + std::to_string(frame[0].frame_id) + "_updated_instance.json";
        saveInstanceJson(frame, save_path);
    }
    
    // Make a ply file for the global instances
    pcl::PointCloud<pcl::PointXYZRGB> global_instances_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> global_instances_cloud_random_color;
    std::vector<int> global_instances_id_list;
    for (const auto& instance : global_instances) {
        if (instance.global_id == -1) continue;
        int id = instance.global_id;
        // rgb color is id % 255, id / 255 % 255, id / 255 / 255 % 255
        int r = id % 255;
        int g = (id / 255) % 255;
        int b = (id / 255 / 255) % 255;
        // Random color
        int r_random = rand() % 255;
        int g_random = rand() % 255;
        int b_random = rand() % 255;
        for (const auto& pt : *instance.accumulated_points) {
            pcl::PointXYZRGB p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.r = r;
            p.g = g;
            p.b = b;
            global_instances_cloud.points.push_back(p);

            // Random color
            p.r = r_random;
            p.g = g_random;
            p.b = b_random;
            global_instances_cloud_random_color.points.push_back(p);
        }
        global_instances_id_list.push_back(id);
    }

    if (if_visualize) std::cout << "*** Global instances id size: " << global_instances_id_list.size() << std::endl;


    // Show the global instances cloud
    if (if_visualize) {
        pcl::visualization::PCLVisualizer viewer("Global Instances");
        viewer.addPointCloud<pcl::PointXYZRGB>(global_instances_cloud_random_color.makeShared(), "global_instances");
        viewer.spin();
    }

    // Save the global instances cloud
    pcl::io::savePLYFileBinary(output_ply_dir + "instance_cloud.ply", global_instances_cloud);
    pcl::io::savePLYFileBinary(output_ply_dir + "colored_instances.ply", global_instances_cloud_random_color);

    return 0;
}