#include <iostream>
#include <random>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>
#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

#include <yaml-cpp/yaml.h>

#include "utils.hpp"

int main(int argc, char** argv)
{
  std::string data_path;
  if (argc == 2)
  {
    data_path = std::string(argv[1]);
  }
  else
  {
    std::cout << "Invalid number of arguments." << std::endl;
    return 1;
  }

  std::string cam_info_file = data_path + "camera_info.yaml";
  YAML::Node cam_info = YAML::LoadFile(cam_info_file);
  int width = cam_info["width"].as<int>();
  int height = cam_info["height"].as<int>();
  std::vector<double> cam_K_ = cam_info["K"].as< std::vector<double> >();
  Eigen::Matrix3f cam_K;
  cam_K <<
    cam_K_[0], cam_K_[1], cam_K_[2],
    cam_K_[3], cam_K_[4], cam_K_[5],
    cam_K_[6], cam_K_[7], cam_K_[8];

  // std::string cam_pose_file = data_path + "tf_camera_rgb_from_base.yaml";
  std::string cam_pose_file = data_path + "tf_base_to_camera.yaml";
  YAML::Node cam_pose_ = YAML::LoadFile(cam_pose_file);
  Eigen::Matrix4f cam_pose = utils::quaternion_to_matrix(cam_pose_);

  // Project 3d PointCloud from depth image
  std::string depth_file = data_path + "depth_obj_n.png";
  cv::Mat depth_raw = cv::imread(depth_file, CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat depth(depth_raw.rows, depth_raw.cols, CV_32FC1);
  for (size_t j = 0; j < depth_raw.rows; ++j)
  {
    for (size_t i = 0; i < depth_raw.cols; ++i)
    {
      float tmp = static_cast<float>(depth_raw.at<unsigned short int>(j, i)) / 1e3;
      if (tmp < 0.3)  // nan for too small depth
      {
        depth.at<float>(j, i) = std::numeric_limits<float>::quiet_NaN();
      }
      else
      {
        depth.at<float>(j, i) = tmp;
      }
    }
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    original_cloud = utils::depth_to_color_pointcloud(height, width, depth, cam_K, cam_pose,
                                               255, 255, 0);

  // Plane extraction
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*original_cloud, *cloud_filtered);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  utils::plane_detect(cloud_filtered, inliers, coefficients, 0.02);
  utils::plane_removal(cloud_filtered, inliers, false);

  // Transform to normal frame
  Eigen::Matrix4f R_normal_to_base = utils::compute_normal_to_origin_matrix(coefficients);
  Eigen::Matrix4f R_base_to_normal = R_normal_to_base.inverse();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    cloud_filtered_plane_frame(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud(
    *cloud_filtered, *cloud_filtered_plane_frame, R_base_to_normal);
  cloud_filtered = cloud_filtered_plane_frame;

  // Euclidean Clustering
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_mono(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud_filtered, *cloud_filtered_mono);
  std::vector<pcl::PointIndices>
    cluster_indices = utils::euclidean_clustering(cloud_filtered_mono, 0.02, 100, height * width);

  // Choose maximum size from clusters
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
  for (std::vector<int>::const_iterator pit = cluster_indices[0].indices.begin(); pit != cluster_indices[0].indices.end(); ++pit)
  {
    cloud_cluster->points.push_back(cloud_filtered_mono->points[*pit]);
  }
  cloud_cluster->width = cloud_cluster->points.size ();
  cloud_cluster->height = 1;
  cloud_cluster->is_dense = true;

  // Compute max_x_pln, max_y_pln, ... , min_z_pln of euclidean-clustered-max-cloud
  float *max_x_pln = new float;
  float *max_y_pln = new float;
  float *max_z_pln = new float;
  float *min_x_pln = new float;
  float *min_y_pln = new float;
  float *min_z_pln = new float;
  utils::compute_max_min_xyz_from_cloud(cloud_cluster,
                                 max_x_pln, max_y_pln, max_z_pln,
                                 min_x_pln, min_y_pln, min_z_pln);

  // Compute centroid
  Eigen::Vector4f pln_centroid;
  pcl::compute3DCentroid(*cloud_cluster, pln_centroid);

  // Load mesh model
  std::string mesh_file = data_path + "data/mvtk/transparent_objects/ShapeNetCore.v2.scaled/02876657/f49d6c4b75f695c44d34bdc365023cf4/models/model_normalized.obj";
  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
  pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPolygonFileOBJ(mesh_file, *mesh) != -1)
  {
    // PolygonMesh -> PointCloud<PointXYZ>
    pcl::fromPCLPointCloud2(mesh->cloud, *obj_cloud);
  }

  // Rotate obj_cloud at random angles
  std::uniform_int_distribution<int> mesh_rot_x_(0, 3);
  std::uniform_int_distribution<int> mesh_rot_y_(0, 3);
  std::uniform_real_distribution<float> mesh_rot_z_(0, 2 * M_PI);
  std::random_device rd;
  std::mt19937 mt(rd());
  float mesh_rot_x = mesh_rot_x_(mt) * M_PI / 2;
  float mesh_rot_y = mesh_rot_y_(mt) * M_PI / 2;
  float mesh_rot_z = mesh_rot_z_(mt);
  Eigen::Matrix4f R_x, R_y, R_z, R;
  R_x <<
    1, 0,               0,                0,
    0, cos(mesh_rot_x), -sin(mesh_rot_x), 0,
    0, sin(mesh_rot_x), cos(mesh_rot_x),  0,
    0, 0,               0,                1;
  R_y <<
    cos(mesh_rot_y),  0, sin(mesh_rot_y), 0,
    0,                1, 0,               0,
    -sin(mesh_rot_y), 0, cos(mesh_rot_y), 0,
    0,                0, 0,               1;
  R_z <<
    cos(mesh_rot_z), -sin(mesh_rot_z), 0, 0,
    sin(mesh_rot_z), cos(mesh_rot_z),  0, 0,
    0,               0,                1, 0,
    0,               0,                0, 1;
  R = R_z * R_y * R_x;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_obj_cloud (new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*obj_cloud, *rotated_obj_cloud, R);

  // Compute max_x_obj, max_y_obj, ... , min_z_obj of rotated_obj_cloud
  float *max_x_obj = new float;
  float *max_y_obj = new float;
  float *max_z_obj = new float;
  float *min_x_obj = new float;
  float *min_y_obj = new float;
  float *min_z_obj = new float;
  utils::compute_max_min_xyz_from_cloud(rotated_obj_cloud,
                                 max_x_obj, max_y_obj, max_z_obj,
                                 min_x_obj, min_y_obj, min_z_obj);

  // Move rotated_obj_cloud onto the plane
  float max_x = *max_x_pln - *max_x_obj;
  float max_y = *max_y_pln - *max_y_obj;
  float min_x = *min_x_pln - *min_x_obj;
  float min_y = *min_y_pln - *min_y_obj;
  std::uniform_real_distribution<float> mesh_orig_x_(min_x, max_x);
  std::uniform_real_distribution<float> mesh_orig_y_(min_y, max_y);
  float mesh_orig_x = mesh_orig_x_(mt);
  float mesh_orig_y = mesh_orig_y_(mt);
  float mesh_orig_z = pln_centroid[2] - *min_z_obj;
  Eigen::Matrix4f transform = R;
  transform(0, 3) = mesh_orig_x;
  transform(1, 3) = mesh_orig_y;
  transform(2, 3) = mesh_orig_z;  // normal -> mesh
  transform = transform * R_base_to_normal;  // base -> normal -> mesh
  std::cout << "transform :" << std::endl << transform << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_on_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = rotated_obj_cloud->points.begin(); it != rotated_obj_cloud->points.end(); ++it)
  {
    pcl::PointXYZRGB pt(255, 0, 0);
    pt.x = mesh_orig_x + it->x;
    pt.y = mesh_orig_y + it->y;
    pt.z = mesh_orig_z + it->z;
    obj_cloud_on_plane->points.push_back(pt);
  }
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_cluster->points.begin(); it != cloud_cluster->points.end(); ++it)
  {
    pcl::PointXYZRGB pt(255, 255, 255);
    pt.x = it->x;
    pt.y = it->y;
    pt.z = it->z;
    obj_cloud_on_plane->points.push_back(pt);
  }

  pcl::transformPointCloud(
    *obj_cloud_on_plane, *obj_cloud_on_plane, R_normal_to_base);
  pcl::transformPointCloud(
    *cloud_cluster, *cloud_cluster, R_normal_to_base);

  // View PointCloud
  pcl::visualization::CloudViewer viewer("obj_cloud_on_plane");
  viewer.showCloud (obj_cloud_on_plane);
  while (!viewer.wasStopped()) {}

  original_cloud->clear();
  cloud_filtered->clear();
  cloud_filtered_plane_frame->clear();
  cloud_filtered_mono->clear();
  cloud_cluster->clear();
  obj_cloud->clear();
  obj_cloud_on_plane->clear();

  return (0);
}
