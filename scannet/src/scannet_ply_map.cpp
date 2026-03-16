/*
 * Author: Clarence Chen
 * This script is used to generate the ply map for a scene.
 */

 #include "read_instance.h"
 #include <boost/filesystem.hpp>
 #include <iomanip>
 
 #define BACKWARD_HAS_DW 1
 #include "backward.hpp"
 namespace backward{
     backward::SignalHandling sh;
 }
 
 
 // Utility: generate random RGB colors
 std::tuple<uint8_t, uint8_t, uint8_t> getColorForInstance(int id) {
     static std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> predefined = {
         {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
         {255, 255, 0}, {0, 255, 255}, {255, 0, 255},
         {128, 128, 0}, {0, 128, 128}, {128, 0, 128}
     };
     if (id < predefined.size()) return predefined[id];
 
     // Random fallback
     std::mt19937 rng(id);
     std::uniform_int_distribution<int> dist(64, 255);
     return {static_cast<uint8_t>(dist(rng)), static_cast<uint8_t>(dist(rng)), static_cast<uint8_t>(dist(rng))};
 }
 
 /// @brief Visualize the colored instances
 /// @param cloud_instances The instance cloud whose rgb is the id.
 void visualizeReColoredInstances(const pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances, std::string output_dir, bool show_visualizer) {
     std::cout << "Visualizing " << cloud_instances.points.size() << " points" << std::endl;
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
 
     for (const auto& pt : cloud_instances.points) {
         auto [r, g, b] = getColorForInstance(pt.r);
         pcl::PointXYZRGB pt_rgb;
         pt_rgb.x = pt.x;
         pt_rgb.y = pt.y;
         pt_rgb.z = pt.z;
         pt_rgb.r = r;
         pt_rgb.g = g;
         pt_rgb.b = b;
         coloredCloud->points.push_back(pt_rgb);
     }
 
     coloredCloud->width = coloredCloud->points.size();
     coloredCloud->height = 1;
     coloredCloud->is_dense = true;
 
     // Save to ply file
     pcl::io::savePLYFileASCII(output_dir + "colored_instances.ply", *coloredCloud);
 
     // Visualizer
     if (show_visualizer) {
         pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Instance Cloud Viewer"));
         viewer->addPointCloud<pcl::PointXYZRGB>(coloredCloud, "colored_instances");
         viewer->setBackgroundColor(0, 0, 0);
         viewer->addCoordinateSystem(0.1);
         viewer->initCameraParameters();
 
         while (!viewer->wasStopped())
             viewer->spinOnce(10);
     }
 }
 
 
 /// @brief Save the instance cloud to a ply file
 /// @param cloud The instance cloud to save
 /// @param file_path The path of the file to save the instance cloud
 void saveCloud(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::string& file_path) {
     pcl::io::savePLYFileASCII(file_path, cloud);
 }
 
 
 template <typename PointT>
 void printCloudStats(const pcl::PointCloud<PointT>& cloud, const std::string& name) {
     if (cloud.empty()) {
         std::cout << "[STATS] " << name << " is empty\n";
         return;
     }
 
     double min_x = std::numeric_limits<double>::infinity();
     double min_y = std::numeric_limits<double>::infinity();
     double min_z = std::numeric_limits<double>::infinity();
     double max_x = -std::numeric_limits<double>::infinity();
     double max_y = -std::numeric_limits<double>::infinity();
     double max_z = -std::numeric_limits<double>::infinity();
     std::size_t nan_count = 0;
 
     for (const auto& p : cloud.points) {
         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
             ++nan_count;
             continue;
         }
         min_x = std::min(min_x, static_cast<double>(p.x));
         min_y = std::min(min_y, static_cast<double>(p.y));
         min_z = std::min(min_z, static_cast<double>(p.z));
         max_x = std::max(max_x, static_cast<double>(p.x));
         max_y = std::max(max_y, static_cast<double>(p.y));
         max_z = std::max(max_z, static_cast<double>(p.z));
     }
 
     std::cout << "[STATS] " << name << " size=" << cloud.size()
               << " bbox=[(" << min_x << "," << min_y << "," << min_z << ") -> ("
               << max_x << "," << max_y << "," << max_z << ")]"
               << " NaN/inf=" << nan_count << std::endl;
 }
 
 
 /// @brief Merge the instance clouds
 /// @param ori_cloud The original instance clouds
 /// @param new_cloud The new instance clouds to merge
 void mergeCloud(pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances,
                 pcl::PointCloud<pcl::PointXYZRGB>& cloud_instances_new) {
     for (const auto& pt : cloud_instances_new.points) {
         // Remove invalid points
         if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
             continue;
         }
         if (std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z)) {
             continue;
         }
         cloud_instances.points.push_back(pt);
     }
 }
 
 int main(int argc, char** argv) {
     // Default parameters
     std::string scene_name = "scene0000_00";
     int start_frame = 0;
     int end_frame = 100;
     int step = 3; // The files were processed every 3 frames
 
     if (argc == 5) {
         scene_name = argv[1];
         start_frame = std::stoi(argv[2]);
         end_frame = std::stoi(argv[3]);
         step = std::stoi(argv[4]);
     }else if (argc == 4) {
         scene_name = argv[1];
         start_frame = std::stoi(argv[2]);
         end_frame = std::stoi(argv[3]);
         // Default step is 3
     }else if (argc == 2) {
         scene_name = argv[1];
     }else if (argc == 1) {
         // Default parameters
     }else{
         std::cout << "Usage: " << argv[0] << " <scene_name> <start_frame> <end_frame> <step>" << std::endl;
         return 1;
     }
 
     std::cout << "Processing scene: " << scene_name << ", start frame: " << start_frame << ", end frame: " << end_frame << ", step: " << step << std::endl;
 
     ///TODO: Change the path to input variables
     std::string raw_images_parent_dir = "/media/cc/Extreme SSD/dataset/scannet/images/scans/";
     std::string refined_instance_parent_dir = "/media/cc/Expansion/scannet/processed/scans/";
     std::string output_ply_dir = "/media/cc/Expansion/scannet/processed/scans/" + scene_name + "/";
 
     std::string meta_file = raw_images_parent_dir + scene_name + "/_info.txt";
 
     pcl::PointCloud<pcl::PointXYZRGB> cloud_instance_accumulated;
     
     InstanceCloudGenerator cloud_generator(meta_file);
 
     int filter_counter = 0;
 
     for (int frame = start_frame; frame <= end_frame; frame += step) {
         std::cout << "*** Processing frame " << frame << std::endl;
         
         pcl::PointCloud<pcl::PointXYZRGB> cloud_instances;
 
         std::ostringstream oss;
         oss << std::setw(6) << std::setfill('0') << frame;
         std::string six_digits_frame = oss.str();
         std::string original_digits_frame = std::to_string(frame);
 
         std::string depth_file = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".depth.pgm";
         std::string instance_file = refined_instance_parent_dir + scene_name + "/refined_instance/" + original_digits_frame + ".png";
         std::string pose_file = raw_images_parent_dir + scene_name + "/frame-" + six_digits_frame + ".pose.txt";
         
         if (!boost::filesystem::exists(instance_file)) {
             std::cout << "*** Instance file does not exist: " << instance_file << "Will try the next frame" << std::endl;
             frame -= (step - 1);
             continue;
         }
         
         if (!boost::filesystem::exists(depth_file)) {
             std::cout << "*** Depth file does not exist: " << depth_file << std::endl;
             return 1;
         }
         
         if (!boost::filesystem::exists(pose_file)) {
             std::cout << "*** Pose file does not exist: " << pose_file << std::endl;
             return 1;
         }
 
         cloud_generator.processFrame(depth_file, instance_file, pose_file, cloud_instances, true, true, true);
         mergeCloud(cloud_instance_accumulated, cloud_instances);
         filter_counter++;
 
         // Filter the accumulated instance clouds with a voxel grid every 20 frames to reduce the memory usage
         if (filter_counter % 10 == 0) {
             std::cout <<"size of cloud_instance_accumulated: " << cloud_instance_accumulated.points.size() << std::endl;
             // printCloudStats(cloud_instance_accumulated, "cloud_instance_accumulated BEFORE voxel filter");
             cloud_instance_accumulated.width  = cloud_instance_accumulated.points.size();
             cloud_instance_accumulated.height = 1;
             cloud_instance_accumulated.is_dense = true;
 
             // pcl::PointCloud<pcl::PointXYZRGB> cloud_temp;
             cloud_generator.voxelFilter(cloud_instance_accumulated, cloud_instance_accumulated);
             // cloud_instance_accumulated = cloud_temp;
             std::cout << "Filtered the accumulated instance clouds with a voxel grid" << std::endl;
         }
 
     }
 
     // Filter the accumulated instance clouds with a voxel grid
     pcl::PointCloud<pcl::PointXYZRGB> cloud_instance_accumulated_filtered;
     cloud_generator.voxelFilter(cloud_instance_accumulated, cloud_instance_accumulated_filtered);
 
     // Save the accumulated instance clouds
     saveCloud(cloud_instance_accumulated_filtered, output_ply_dir + "instance_cloud_background.ply");
 
     // Show the accumulated instance clouds
     visualizeReColoredInstances(cloud_instance_accumulated_filtered, output_ply_dir, false);
 
     return 0;
 }