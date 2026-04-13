/***
 * This file is used to get the fused features for each instance in a scene
 * The features are averaged from the features of the instance clouds
 * Author: Clarence Chen
 * Date: 2025-06-19
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <json/single_include/nlohmann/json.hpp>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Structure to store accumulated feature and count
struct FeatureAccumulator {
    std::vector<float> feature_sum;
    int count = 0;

    void add(const std::vector<float>& vec) {
        if (feature_sum.empty()) feature_sum.resize(vec.size(), 0.0f);
        for (size_t i = 0; i < vec.size(); ++i) {
            feature_sum[i] += vec[i];
        }
        count++;
    }

    std::vector<float> average() const {
        std::vector<float> avg(feature_sum.size(), 0.0f);
        if (count > 0) {
            for (size_t i = 0; i < feature_sum.size(); ++i) {
                avg[i] = feature_sum[i] / count;
            }
        }
        return avg;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: get_fused_object_features <path_to_json_files> <result_save_folder>\n";
        // path_to_json_files: e.g. scannet/processed/scans/scene0000_00/refined_instance
        // result_save_folder: e.g. scannet/processed/scans/scene0000_00
        return 1;
    }

    std::string path = argv[1];
    std::string result_save_folder = argv[2];

    // Check if path exists and get json files in the path if it exist
    if (!fs::exists(path)) {
        std::cerr << "Error: path " << path << " does not exist\n";
        return 1;
    }

    // Get the json files in the path
    std::vector<std::string> json_files;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".json") {
            json_files.push_back(entry.path().string());
        }
    }
    std::cout << "Found " << json_files.size() << " json files in " << path << "\n";

    std::unordered_map<int, FeatureAccumulator> instance_features;
    
    // Read all json files in the path and accumulate the features
    for (int i = 0; i < json_files.size(); ++i) {
        std::string filename = json_files[i];
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: could not open file " << filename << "\n";
            continue;
        }

        json j;
        file >> j;

        for (const auto& item : j) {
            int id = item["instance_id"];
            std::vector<float> feature = item["feature"];
            instance_features[id].add(feature);
        }

        // Print progress every 100 files
        if (i % 100 == 0) {
            std::cout << "Processed " << i << " files\n";
        }
    }

    // Output JSON to the result save folder
    json result = json::array();
    for (const auto& [id, acc] : instance_features) {
        result.push_back({
            {"instance_id", id},
            {"feature", acc.average()},
            {"occurance", acc.count}
        });
    }

    std::ofstream out(result_save_folder + "/averaged_instance_features.json");
    out << result.dump(2);
    out.close();

    std::cout << "Averaged features saved to " << result_save_folder + "/averaged_instance_features.json" << "\n";
    return 0;
}
