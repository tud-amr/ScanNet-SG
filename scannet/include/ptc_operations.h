#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

struct InstanceInfo {
    Eigen::Vector3f center;
    Eigen::Matrix3f rotation;
    Eigen::Vector3f bbox_size;
};

// Get the instance IDs from the cloud
std::vector<int> getInstanceIDs(const CloudT::Ptr& cloud, bool three_channel_id = false) {
    std::set<int> instance_ids;
    for (const auto& pt : cloud->points) {
        if (three_channel_id) {
            // Decode the three channel id to int
            int id = static_cast<int>(pt.r) + static_cast<int>(pt.g) * 255 + static_cast<int>(pt.b) * 255 * 255;
            instance_ids.insert(id);
        } else {
            instance_ids.insert(static_cast<int>(pt.r)); // R channel as instance ID
        }
    }
    return std::vector<int>(instance_ids.begin(), instance_ids.end());
}

// Extract the instance from the cloud. Assume the instance ID is the R channel of the point when three_channel_id is false.
CloudT::Ptr extractInstance(const CloudT::Ptr& cloud, int instance_id, bool three_channel_id = false) {
    CloudT::Ptr result(new CloudT);
    for (const auto& pt : cloud->points) {
        if (three_channel_id) {
            int id = static_cast<int>(pt.r) + static_cast<int>(pt.g) * 255 + static_cast<int>(pt.b) * 255 * 255;
            if (id == instance_id) {
                result->points.push_back(pt);
            }
        } else {
            if (static_cast<int>(pt.r) == instance_id) {
                result->points.push_back(pt);
            }
        }
    }
    result->width = result->points.size();
    result->height = 1;
    result->is_dense = true;
    return result;
}

// Remove outliers from the cloud using the Mahalanobis distance
CloudT::Ptr removeOutliersMahalanobis(const CloudT::Ptr& cloud, float threshold = 1.2) {
    if (cloud->empty()) return cloud;

    Eigen::MatrixXf data(3, cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        data.col(i) = cloud->points[i].getVector3fMap();
    }

    Eigen::Vector3f mean = data.rowwise().mean();
    Eigen::MatrixXf centered = data.colwise() - mean;
    Eigen::Matrix3f cov = (centered * centered.transpose()) / (data.cols() - 1);

    Eigen::Matrix3f cov_inv = cov.inverse();

    std::vector<int> inlier_indices;
    for (size_t i = 0; i < data.cols(); ++i) {
        Eigen::Vector3f diff = data.col(i) - mean;
        float mahalanobis = std::sqrt(diff.transpose() * cov_inv * diff);
        if (mahalanobis <= threshold) {
            inlier_indices.push_back(static_cast<int>(i));
        }
    }

    CloudT::Ptr inliers(new CloudT);
    for (int idx : inlier_indices) {
        inliers->points.push_back(cloud->points[idx]);
    }
    inliers->width = inliers->points.size();
    inliers->height = 1;
    inliers->is_dense = true;
    return inliers;
}

// Filter the cloud by clustering.
CloudT::Ptr filterByClustering(const CloudT::Ptr& cloud, float cluster_tolerance = 0.1f) {
    if (cloud->empty()) return cloud;

    // Use a KdTree for clustering
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance); // in meters
    ec.setMinClusterSize(10); // ignore very tiny clusters
    ec.setMaxClusterSize(cloud->points.size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) return CloudT::Ptr(new CloudT);

    // Find the size of the largest cluster
    size_t max_size = 0;
    for (const auto& indices : cluster_indices)
        if (indices.indices.size() > max_size)
            max_size = indices.indices.size();

    // Keep only clusters with size >= 1/2 of largest
    CloudT::Ptr filtered(new CloudT);
    for (const auto& indices : cluster_indices) {
        if (indices.indices.size() >= max_size / 2) {
            for (int idx : indices.indices) {
                filtered->points.push_back(cloud->points[idx]);
            }
        }
    }
    filtered->width = filtered->points.size();
    filtered->height = 1;
    filtered->is_dense = true;
    return filtered;
}

// Compute the OBB of the cloud.
InstanceInfo computeOBB(const CloudT::Ptr& cloud) {
    pcl::PCA<PointT> pca;
    pca.setInputCloud(cloud);
    Eigen::Vector3f mean = pca.getMean().head<3>();
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = eigenvectors.transpose();
    transform.block<3,1>(0,3) = -1.0f * (eigenvectors.transpose() * mean);

    CloudT::Ptr transformed(new CloudT);
    pcl::transformPointCloud(*cloud, *transformed, transform);

    PointT min_pt, max_pt;
    pcl::getMinMax3D(*transformed, min_pt, max_pt);
    Eigen::Vector3f size = max_pt.getVector3fMap() - min_pt.getVector3fMap();

    Eigen::Vector3f center_local = 0.5f * (min_pt.getVector3fMap() + max_pt.getVector3fMap());
    Eigen::Vector3f center_world = eigenvectors * center_local + mean;

    return InstanceInfo{center_world, eigenvectors, size};
}
