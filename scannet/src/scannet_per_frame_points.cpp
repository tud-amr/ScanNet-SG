/*
 * Author: Clarence Chen
 * This script is used to generate the per-frame instance point cloud and bbox of each instance in a scene.
 * The output will be saved in a ply file and a json file for each frame.
 * The json file will contain the instance ID, and the bbox of the instance.
 * The point cloud of all instances will be saved in a ply file with the instance ID as the R channel of the point.
 */

#include "ptc_operations.h"
#include "read_instance.h"
#include <boost/filesystem.hpp>
#include <iomanip>
#include <fstream>
#include <json/single_include/nlohmann/json.hpp>

// #define BACKWARD_HAS_DW 1
// #include "backward.hpp"
// namespace backward{
//     backward::SignalHandling sh;
// }

using json = nlohmann::json;

/// @brief Save the instance cloud to a ply file
/// @param cloud The instance cloud to save
/// @param file_path The path of the file to save the instance cloud
void saveCloud(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::string& file_path) {
    // Save as binary ply file
    pcl::io::savePLYFileBinary(file_path, cloud);
}

/// @brief Save instance information to JSON file
/// @param instance_info_map Map of instance ID to InstanceInfo
/// @param file_path The path to save the JSON file
void saveInstanceInfoToJSON(const std::unordered_map<int, InstanceInfo>& instance_info_map, 
                           const std::string& file_path) {
    json j;
    
    for (const auto& [instance_id, info] : instance_info_map) {
        json instance_json;
        instance_json["instance_id"] = instance_id;
        
        // Save center position
        json center_json;
        center_json["x"] = info.center.x();
        center_json["y"] = info.center.y();
        center_json["z"] = info.center.z();
        instance_json["center"] = center_json;
        
        // Save bbox size
        json bbox_json;
        bbox_json["x"] = info.bbox_size.x();
        bbox_json["y"] = info.bbox_size.y();
        bbox_json["z"] = info.bbox_size.z();
        instance_json["bbox_size"] = bbox_json;
        
        // Save rotation matrix
        json rotation_json;
        for (int i = 0; i < 3; ++i) {
            json row_json;
            for (int j = 0; j < 3; ++j) {
                row_json.push_back(info.rotation(i, j));
            }
            rotation_json.push_back(row_json);
        }
        instance_json["rotation"] = rotation_json;
        
        j.push_back(instance_json);
    }
    
    std::ofstream file(file_path);
    file << j.dump(4);
}

int main(int argc, char** argv) {
    // Default parameters
    std::string scene_name = "scene0000_00";
    std::string refined_instance_parent_dir = "/media/cc/My Passport/dataset/scannet/processed/scans/";
    std::string raw_images_parent_dir = "/media/cc/My Passport/dataset/scannet/images/scans/";
    bool save_ply = false;
    
    if (argc == 2) {
        scene_name = argv[1];
    } else if (argc == 3) {
        scene_name = argv[1];
        refined_instance_parent_dir = argv[2];
    } else if (argc == 4) {
        scene_name = argv[1];
        refined_instance_parent_dir = argv[2];
        raw_images_parent_dir = argv[3];
    }
    else if (argc == 5) {
        scene_name = argv[1];
        refined_instance_parent_dir = argv[2];
        raw_images_parent_dir = argv[3];
        if (argv[4] == "true" || argv[4] == "True") {
            save_ply = true;
        }
    }
    else if (argc == 1) {
        // Default parameters
    } else {
        std::cout << "Usage: " << argv[0] << " <scene_name> <refined_instance_parent_dir. e.g. xxx/processed/scans/> <raw_images_parent_dir. e.g. xxx/images/scans/>" << std::endl;
        return 1;
    }

    // If refined_instance_parent_dir or raw_images_parent_dir not end with '/', add it
    if (refined_instance_parent_dir.back() != '/') {
        refined_instance_parent_dir += "/";
    }
    if (raw_images_parent_dir.back() != '/') {
        raw_images_parent_dir += "/";
    }

    // std::cout << "Processing scene: " << scene_name << std::endl;
    std::string meta_file = raw_images_parent_dir + scene_name + "/_info.txt";
    std::string output_dir = refined_instance_parent_dir + scene_name + "/per_frame_points/";
    
    // Create output directory if it doesn't exist
    boost::filesystem::create_directories(output_dir);

    std::string refined_instance_dir = refined_instance_parent_dir + scene_name + "/refined_instance/";

    // Check if refined_instance directory exists
    if (!boost::filesystem::exists(refined_instance_dir)) {
        std::cout << "*** Refined instance directory does not exist: " << refined_instance_dir << std::endl;
        return 1;
    }
    
    InstanceCloudGenerator cloud_generator(meta_file);

    // Find all PNG files in the refined_instance directory
    std::vector<std::string> png_files;
    for (const auto& entry : boost::filesystem::directory_iterator(refined_instance_dir)) {
        if (boost::filesystem::is_regular_file(entry.path()) && entry.path().extension() == ".png") {
            png_files.push_back(entry.path().filename().string());
        }
    }
    
    // Sort by numerical frame number
    std::sort(png_files.begin(), png_files.end(), [](const std::string& a, const std::string& b) {
        // Extract frame numbers from filenames (e.g., "51.png" -> 51)
        int frame_a = std::stoi(a.substr(0, a.find('.')));
        int frame_b = std::stoi(b.substr(0, b.find('.')));
        return frame_a < frame_b;
    });
    std::cout << "Found " << png_files.size() << " PNG files in " << refined_instance_dir << " Processing..." <<std::endl;

    for (const auto& png_file : png_files) {
        // Extract frame number from PNG filename (e.g., "51.png" -> 51)
        std::string frame_str = png_file.substr(0, png_file.find('.'));
        int frame = std::stoi(frame_str);
        
        // std::cout << "*** Processing frame " << frame << " (file: " << png_file << ")" << std::endl;
        
        // Construct corresponding depth and pose filenames
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << frame;
        std::string six_digits_frame = oss.str();

        std::string depth_file = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".depth.pgm";
        std::string instance_file = refined_instance_dir + png_file;
        std::string pose_file = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".pose.txt";

        // Check if files exist
        if (!boost::filesystem::exists(depth_file)) {
            std::cout << "*** Depth file does not exist: " << depth_file << " for instance file " << png_file << std::endl;
            continue;
        }
        
        if (!boost::filesystem::exists(pose_file)) {
            std::cout << "*** Pose file does not exist: " << pose_file << " for instance file " << png_file << std::endl;
            continue;
        }

        // Process the frame to get instance point cloud (in camera frame)
        pcl::PointCloud<pcl::PointXYZRGB> cloud_instances;
        cloud_generator.processFrame(depth_file, instance_file, pose_file, cloud_instances, false, false, true); //Keep background

        if (cloud_instances.empty()) {
            std::cout << "*** No instances found in frame " << frame << std::endl;
            continue;
        }

        // Convert to PCL cloud pointer for processing
        CloudT::Ptr cloud_ptr(new CloudT(cloud_instances));

        // Get unique instance IDs
        std::vector<int> instance_ids = getInstanceIDs(cloud_ptr);
        // std::cout << "Found " << instance_ids.size() << " instances in frame " << frame << std::endl;

        // Process each instance
        std::unordered_map<int, InstanceInfo> instance_info_map;
        pcl::PointCloud<pcl::PointXYZRGB> all_instances_cloud;

        for (int instance_id : instance_ids) {            
            // Extract instance point cloud
            CloudT::Ptr instance_cloud = extractInstance(cloud_ptr, instance_id);
            
            // Filter the instance cloud
            // CloudT::Ptr filtered_cloud = filterByClustering(instance_cloud); // Not useful and slow. Remove.
            // CloudT::Ptr cleaned_cloud = removeOutliersMahalanobis(instance_cloud);

            CloudT::Ptr cleaned_cloud = instance_cloud;

            if (cleaned_cloud->size() < 10) {
                std::cout << "Instance " << instance_id << " has too few points after filtering (" << cleaned_cloud->size() << "), skipping" << std::endl;
                continue;
            }

            // Calculate bounding box
            if (instance_id != 0) {
                InstanceInfo info = computeOBB(cleaned_cloud);
                // Check if instance is too large (ignore if any dimension > 3m). Keep all instances.
                // float size_threshold = 3.0f;
                // if (info.bbox_size.x() > size_threshold || info.bbox_size.y() > size_threshold || info.bbox_size.z() > size_threshold) {
                //     std::cout << "Instance " << instance_id << " is too big, ignored" << std::endl;
                //     continue;
                // }

                // Store instance info
                instance_info_map[instance_id] = info;
            }

            // Add filtered points to the combined cloud
            for (const auto& pt : cleaned_cloud->points) {
                pcl::PointXYZRGB new_pt;
                new_pt.x = pt.x;
                new_pt.y = pt.y;
                new_pt.z = pt.z;
                new_pt.r = instance_id;
                new_pt.g = instance_id;
                new_pt.b = instance_id;
                all_instances_cloud.points.push_back(new_pt);
            }
        }

        if (instance_info_map.empty()) {
            std::cout << "*** No valid instances found in frame " << frame << std::endl;
            continue;
        }

        // Set cloud properties
        all_instances_cloud.width = all_instances_cloud.points.size();
        all_instances_cloud.height = 1;
        all_instances_cloud.is_dense = true;

        // Save files
        std::string ply_filename = output_dir + frame_str + "_instances.ply";
        std::string json_filename = output_dir + frame_str + "_instances.json";

        if (save_ply) {
            saveCloud(all_instances_cloud, ply_filename);
        }
        saveInstanceInfoToJSON(instance_info_map, json_filename);

        // std::cout << "Saved " << instance_info_map.size() << " instances to:" << std::endl;
        // std::cout << "  PLY: " << ply_filename << std::endl;
        // std::cout << "  JSON: " << json_filename << std::endl;
    }

    std::cout << "Processing completed!" << std::endl;
    return 0;
}

