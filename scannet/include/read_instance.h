#pragma once

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>

struct CameraIntrinsics {
    float fx, fy, cx, cy;
};

struct Metadata {
    CameraIntrinsics depthIntrinsics;
    CameraIntrinsics colorIntrinsics;
    int depthWidth = 0, depthHeight = 0;
    int colorWidth = 0, colorHeight = 0;
    float depthShift = 1000.0f;
};

class InstanceCloudGenerator {
public:
    /// @brief Constructor
    /// @param infoFile Path to the info file containing the camera intrinsics and other metadata
    explicit InstanceCloudGenerator(const std::string& infoFile) {
        metadata_ = readMetadata(infoFile);
        std::cout << "Metadata read successfully" << std::endl;
    }

    /// @brief Process a single frame
    /// @param depthPath Path to the depth image
    /// @param instancePath Path to the instance image
    /// @param posePath Path to the pose file
    /// @param apply_filter Whether to apply a filter to the instance clouds
    /// @param global_frame Whether to use the global frame or the camera frame
    /// @param add_background Whether to add background points
    /// @param max_depth Maximum depth threshold in meters (points beyond this will be filtered out, 0 means no filtering)
    /// @param subsample_factor Subsample factor for rows and cols (1 = read all, 2 = read every other row/col, etc.)
    void processFrame(const std::string& depthPath,
                      const std::string& instancePath,
                      const std::string& posePath,
                      pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances,
                      bool global_frame = true,
                      bool apply_filter = true,
                      bool add_background = false,
                      float max_depth = 0.0f,
                      int subsample_factor = 1) {
        cv::Mat depth = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
        cv::Mat instance = cv::imread(instancePath, cv::IMREAD_UNCHANGED);
        Eigen::Matrix4f pose = loadPose(posePath);
        extractInstances(depth, instance, pose, cloud_instances, global_frame, apply_filter, add_background, max_depth, subsample_factor);
    }



    /// @brief Parse the intrinsics from the line
    /// @param line The line to parse
    /// @param intrinsics The intrinsics to parse
    static void parseIntrinsics(const std::string& line, CameraIntrinsics& intrinsics) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) tokens.push_back(token);

        intrinsics.fx = std::stof(tokens[2]);
        intrinsics.fy = std::stof(tokens[7]);
        intrinsics.cx = std::stof(tokens[4]);
        intrinsics.cy = std::stof(tokens[8]);
    }

    /// @brief Read the metadata from the file
    /// @param filename The filename to read
    /// @return The metadata
    static Metadata readMetadata(const std::string& filename) {
        Metadata meta{};
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("m_depthWidth") == 0) meta.depthWidth = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_depthHeight") == 0) meta.depthHeight = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_colorWidth") == 0) meta.colorWidth = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_colorHeight") == 0) meta.colorHeight = std::stoi(line.substr(line.find('=') + 1));
            else if (line.find("m_depthShift") == 0) meta.depthShift = std::stof(line.substr(line.find('=') + 1));
            else if (line.find("m_calibrationColorIntrinsic") == 0) parseIntrinsics(line, meta.colorIntrinsics);
            else if (line.find("m_calibrationDepthIntrinsic") == 0) parseIntrinsics(line, meta.depthIntrinsics);
        }
        std::cout << "Depth width: " << meta.depthWidth << std::endl;
        std::cout << "Depth height: " << meta.depthHeight << std::endl;
        std::cout << "Color width: " << meta.colorWidth << std::endl;
        std::cout << "Color height: " << meta.colorHeight << std::endl;
        std::cout << "Depth shift: " << meta.depthShift << std::endl;
        std::cout << "Color intrinsics: " << meta.colorIntrinsics.fx << " " << meta.colorIntrinsics.fy << " " << meta.colorIntrinsics.cx << " " << meta.colorIntrinsics.cy << std::endl;
        std::cout << "Depth intrinsics: " << meta.depthIntrinsics.fx << " " << meta.depthIntrinsics.fy << " " << meta.depthIntrinsics.cx << " " << meta.depthIntrinsics.cy << std::endl;
        return meta;
    }

    /// @brief Load the camera pose from the file
    /// @param posePath The path to the pose file
    /// @return The pose
    static Eigen::Matrix4f loadPose(const std::string& posePath) {
        std::ifstream file(posePath);
        Eigen::Matrix4f pose;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                file >> pose(i, j);
        return pose;
    }


    /// @brief Apply the SOR filter to the point cloud
    /// @param cloud_in The input point cloud
    /// @param cloud_out The output point cloud
    void sorFilter(pcl::PointCloud<pcl::PointXYZRGB>& cloud_in, pcl::PointCloud<pcl::PointXYZRGB>& cloud_out, float mean_k = 50, float stddev_mul_thresh = 1.0) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(cloud_in));
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_in_ptr);
        sor.setMeanK(mean_k);  
        sor.setStddevMulThresh(stddev_mul_thresh);
        sor.filter(cloud_out);
    }


    /// @brief Apply the voxel filter to the point cloud
    /// @param cloud_in The input point cloud
    /// @param cloud_out The output point cloud
    void voxelFilter(pcl::PointCloud<pcl::PointXYZRGB>& cloud_in, pcl::PointCloud<pcl::PointXYZRGB>& cloud_out, float leaf_size = 0.02f) {
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(cloud_in));
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_in.makeShared());
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(cloud_out);
    }

    /// @brief Apply the voxel filter to the point cloud
    /// @param cloud_in The input point cloud
    /// @param cloud_out The output point cloud
    /// @param leaf_size The leaf size of the voxel grid
    void voxelFilter(pcl::PointCloud<pcl::PointXYZ>& cloud_in, pcl::PointCloud<pcl::PointXYZ>& cloud_out, float leaf_size = 0.05f) {
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_ptr(new pcl::PointCloud<pcl::PointXYZ>(cloud_in));
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_in.makeShared());
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(cloud_out);
    }

    /// @brief Extract the instances from the depth and instance images
    /// @param depth The depth image
    /// @param instance The instance image
    /// @param T The pose
    /// @param cloud_instances The instance cloud
    /// @param global_frame Whether to use the global frame or the camera frame
    /// @param apply_filter Whether to apply a filter to the instance clouds
    /// @param add_background Whether to add background points
    /// @param max_depth Maximum depth threshold in meters (points beyond this will be filtered out, 0 means no filtering)
    /// @param subsample_factor Subsample factor for rows and cols (1 = read all, 2 = read every other row/col, etc.)
    void extractInstances(const cv::Mat& depth,
                          const cv::Mat& instance,
                          const Eigen::Matrix4f& T,
                          pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances,
                          bool global_frame = true,
                          bool apply_filter = true,
                          bool add_background = false,
                          float max_depth = 0.0f,
                          int subsample_factor = 1) {
                      
        float fx = metadata_.depthIntrinsics.fx, fy = metadata_.depthIntrinsics.fy;
        float cx = metadata_.depthIntrinsics.cx, cy = metadata_.depthIntrinsics.cy;
        float rgb_fx = metadata_.colorIntrinsics.fx, rgb_fy = metadata_.colorIntrinsics.fy;
        float rgb_cx = metadata_.colorIntrinsics.cx, rgb_cy = metadata_.colorIntrinsics.cy;
        float shift = metadata_.depthShift;

        pcl::PointCloud<pcl::PointXYZRGB> rawInstances;

        // Subsample the depth image by reading every n rows and cols
        for (int v = 0; v < depth.rows; v += subsample_factor) {
            for (int u = 0; u < depth.cols; u += subsample_factor) {
                uint16_t d = depth.at<uint16_t>(v, u);
                if (d == 0) continue;

                float z = d / shift;
                
                // Filter out depth values beyond the threshold
                if (max_depth > 0.0f && z > max_depth) continue;

                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                Eigen::Vector4f pt_cam(x, y, z, 1.0f);
                Eigen::Vector4f pt_world = T * pt_cam;

                // project to color image
                float u_rgb = x * rgb_fx / z + rgb_cx;
                float v_rgb = y * rgb_fy / z + rgb_cy;

                if (u_rgb >= 0 && u_rgb < instance.cols && v_rgb >= 0 && v_rgb < instance.rows) {
                    uint8_t instance_id = instance.at<uint8_t>(v_rgb, u_rgb);
                    if (instance_id == 0 && !add_background) continue;
                    
                    if (global_frame) {
                        pcl::PointXYZRGB pt;
                        pt.x = pt_world.x();
                        pt.y = pt_world.y();
                        pt.z = pt_world.z();
                        pt.r = instance_id;
                        pt.g = instance_id;
                        pt.b = instance_id;
                        rawInstances.push_back(pt);
                    } else {
                        pcl::PointXYZRGB pt;
                        pt.x = x;
                        pt.y = y;
                        pt.z = z;
                        pt.r = instance_id;
                        pt.g = instance_id;
                        pt.b = instance_id;
                        rawInstances.push_back(pt);
                    }
                    
                    // std::cout << "Added point " << pt_world.x() << ", " << pt_world.y() << ", " << pt_world.z() << " to instance " << static_cast<int>(instance_id) << std::endl;
                }
            }
        }

        if (apply_filter) {
            sorFilter(rawInstances, rawInstances);
            voxelFilter(rawInstances, rawInstances);
            cloud_instances = rawInstances;
        } else {
            cloud_instances = rawInstances;
        }
    }

private:
    Metadata metadata_;
};