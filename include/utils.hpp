#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <yaml-cpp/yaml.h>


namespace utils
{

Eigen::Matrix4f quaternion_to_matrix(YAML::Node pose)
{
  double tx = pose["transform"]["translation"]["x"].as<double>();
  double ty = pose["transform"]["translation"]["y"].as<double>();
  double tz = pose["transform"]["translation"]["z"].as<double>();
  double rx = pose["transform"]["rotation"]["x"].as<double>();
  double ry = pose["transform"]["rotation"]["y"].as<double>();
  double rz = pose["transform"]["rotation"]["z"].as<double>();
  double rw = pose["transform"]["rotation"]["w"].as<double>();
  double r00 = 1 - 2 * ry * ry - 2 * rz * rz;
  double r01 = 2 * rx * ry - 2 * rz * rw;
  double r02 = 2 * rx * rz + 2 * ry * rw;
  double r10 = 2 * rx * ry + 2 * rz * rw;
  double r11 = 1 - 2 * rx * rx - 2 * rz * rz;
  double r12 = 2 * ry * rz - 2 * rw * rx;
  double r20 = 2 * rx * rz - 2 * ry * rw;
  double r21 = 2 * ry * rz + 2 * rw * rx;
  double r22 = 1 - 2 * rx * rx - 2 * ry * ry;
  Eigen::Matrix4f pose_mat;
  pose_mat <<
    r00, r01, r02, tx,
    r10, r11, r12, ty,
    r20, r21, r22, tz,
    0,   0,   0,   1;

  return pose_mat;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr depth_to_color_pointcloud(int height,
                                                                 int width,
                                                                 cv::Mat depth,
                                                                 Eigen::Matrix3f cam_K,
                                                                 Eigen::Matrix4f cam_pose,
                                                                 uint8_t r,
                                                                 uint8_t g,
                                                                 uint8_t b)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud->height = height;
  cloud->width = width;
  cloud->points.resize(height * width);
  cloud->is_dense = false;

  for (int v = 0; v < height; v++)
  {
    for (int u = 0; u < width; u++)
    {
      double d = std::numeric_limits<float>::quiet_NaN();
      d = depth.at<float>(v, u);
      Eigen::Vector3f uv(u, v, 1);
      uv = cam_K.inverse() * uv;
      Eigen::Vector4f direction_(uv(0), uv(1), uv(2), 1);
      if (!std::isnan(d))
      {
        direction_(0) *= d;
        direction_(1) *= d;
        direction_(2) = d;
      }
      else
      {
        direction_(0) = std::numeric_limits<float>::quiet_NaN();
        direction_(1) = std::numeric_limits<float>::quiet_NaN();
        direction_(2) = std::numeric_limits<float>::quiet_NaN();
      }
      direction_ = cam_pose * direction_;
      Eigen::Vector3f direction(direction_(0), direction_(1), direction_(2));
      pcl::PointXYZRGB pt(r, g, b);
      pt.x = direction(0);
      pt.y = direction(1);
      pt.z = direction(2);
      cloud->points[v * width + u] = pt;
    }
  }

  return cloud;
}


void plane_detect(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                 pcl::PointIndices::Ptr inliers,
                 pcl::ModelCoefficients::Ptr coefficients,
                 double threshold)
{
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(threshold);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);
}


void plane_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  bool negative)
{
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(negative); // true -> remove plane, false -> remove all except plane
  extract.filter(*cloud);
}


Eigen::Matrix4f compute_normal_to_origin_matrix(pcl::ModelCoefficients::Ptr coefficients)
{
  // a = coefficients[0]
  // b = coefficients[1]
  // c = coefficients[2]
  // d = coefficients[3]
  // ax + by + cz = d
  // normal: (a, b, c)

  Eigen::Vector3f normal(
    coefficients->values[0], coefficients->values[1], coefficients->values[2]);
  normal = normal.normalized();
  Eigen::Quaternionf rot;
  rot.setFromTwoVectors(Eigen::Vector3f::UnitZ(), normal);
  Eigen::Matrix3f rot_mat = rot.toRotationMatrix();
  Eigen::Matrix4f normal_to_origin_mat;
  normal_to_origin_mat <<
    rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), 0,
    rot_mat(1, 0), rot_mat(1, 1), rot_mat(1, 2), 0,
    rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2), 0,
                0,             0,             0, 1;

  return normal_to_origin_mat;
}


std::vector<pcl::PointIndices> euclidean_clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                    double cluster_tolerance,
                                                    int min_cluster_size,
                                                    int max_cluster_size)
{
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);

  // Euclidean Clustering
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  return cluster_indices;
}


void compute_max_min_xyz_from_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                    float *max_x, float *max_y, float *max_z,
                                    float *min_x, float *min_y, float *min_z)
{
  float tmp_max_x, tmp_max_y, tmp_max_z, tmp_min_x, tmp_min_y, tmp_min_z;
  tmp_max_x = tmp_max_y = tmp_max_z = -1000.0;
  tmp_min_x = tmp_min_y = tmp_min_z = 1000.0;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator
         it = cloud->points.begin(); it != cloud->points.end(); ++it)
  {
    *max_x = std::max(it->x, tmp_max_x);
    *max_y = std::max(it->y, tmp_max_y);
    *max_z = std::max(it->z, tmp_max_z);
    *min_x = std::min(it->x, tmp_min_x);
    *min_y = std::min(it->y, tmp_min_y);
    *min_z = std::min(it->z, tmp_min_z);
    tmp_max_x = *max_x;
    tmp_max_y = *max_y;
    tmp_max_z = *max_z;
    tmp_min_x = *min_x;
    tmp_min_y = *min_y;
    tmp_min_z = *min_z;
  }
}

} // namespace utils
