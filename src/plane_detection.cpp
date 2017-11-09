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

#include <yaml-cpp/yaml.h>

#include <pcl/visualization/cloud_viewer.h>


void planeDetect(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                 pcl::PointIndices::Ptr inliers,
                 pcl::ModelCoefficients::Ptr coefficients,
                 double threshold)
{
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE); // model
  seg.setMethodType(pcl::SAC_RANSAC); // detection method
  seg.setDistanceThreshold(threshold); // 0.5 or so ?
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);
}


void planeRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  bool negative)
{
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(negative); // true -> remove plane, false -> remove all except plane
  extract.filter(*cloud);
}


int main(int argc, char** argv)
{
  std::string data_path;
  if (argc == 2) {
    data_path = std::string(argv[1]);
  }
  else {
    std::cout << "Invalid number of arguments." << std::endl;
    return 1;
  }

  std::string cam_info_file = data_path + "camera_info.yaml";
  std::cout << "cam_info_file : " << cam_info_file << std::endl;
  YAML::Node cam_info = YAML::LoadFile(cam_info_file);
  std::vector<double> cam_K_ = cam_info["K"].as< std::vector<double> >();
  int width = cam_info["width"].as<int>();
  int height = cam_info["height"].as<int>();
  Eigen::Matrix3f cam_K;
  cam_K <<
    cam_K_[0], cam_K_[1], cam_K_[2],
    cam_K_[3], cam_K_[4], cam_K_[5],
    cam_K_[6], cam_K_[7], cam_K_[8];
  std::cout << "width : " << width << std::endl;
  std::cout << "height : " << height << std::endl;
  std::cout << "cam_K : " << std::endl << cam_K << std::endl;

  // std::string cam_pose_file = data_path + "tf_camera_rgb_from_base.yaml";
  std::string cam_pose_file = data_path + "tf_base_to_camera.yaml";
  std::cout << "cam_pose_file : " << cam_pose_file << std::endl;
  YAML::Node cam_pose_ = YAML::LoadFile(cam_pose_file);
  double tx = cam_pose_["transform"]["translation"]["x"].as<double>();
  double ty = cam_pose_["transform"]["translation"]["y"].as<double>();
  double tz = cam_pose_["transform"]["translation"]["z"].as<double>();
  double rx = cam_pose_["transform"]["rotation"]["x"].as<double>();
  double ry = cam_pose_["transform"]["rotation"]["y"].as<double>();
  double rz = cam_pose_["transform"]["rotation"]["z"].as<double>();
  double rw = cam_pose_["transform"]["rotation"]["w"].as<double>();
  double r00 = 1 - 2 * ry * ry - 2 * rz * rz;
  double r01 = 2 * rx * ry - 2 * rz * rw;
  double r02 = 2 * rx * rz + 2 * ry * rw;
  double r10 = 2 * rx * ry + 2 * rz * rw;
  double r11 = 1 - 2 * rx * rx - 2 * rz * rz;
  double r12 = 2 * ry * rz - 2 * rw * rx;
  double r20 = 2 * rx * rz - 2 * ry * rw;
  double r21 = 2 * ry * rz + 2 * rw * rx;
  double r22 = 1 - 2 * rx * rx - 2 * ry * ry;
  Eigen::Matrix4f cam_pose;
  cam_pose <<
    r00, r01, r02, tx,
    r10, r11, r12, ty,
    r20, r21, r22, tz,
    0,   0,   0,   1;
  std::cout << "cam_pose : " << std::endl << cam_pose << std::endl << std::endl;

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

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  original_cloud->width = width;
  original_cloud->height = height;

  for (int v = 0; v < height; v++) {
    for (int u = 0; u < width; u++) {
      double d = std::numeric_limits<float>::quiet_NaN();
      d = depth.at<float>(v, u);

      Eigen::Vector3f uv(u, v, 1);
      uv = cam_K.inverse() * uv;
      Eigen::Vector4f direction_(uv(0), uv(1), uv(2), 1);
      if (!std::isnan(d)) {
        direction_(0) *= d;
        direction_(1) *= d;
        direction_(2) = d;
      }
      else {
        direction_(0) = std::numeric_limits<float>::quiet_NaN();
        direction_(1) = std::numeric_limits<float>::quiet_NaN();
        direction_(2) = std::numeric_limits<float>::quiet_NaN();
      }

      direction_ = cam_pose * direction_;
      Eigen::Vector3f direction(direction_(0), direction_(1), direction_(2));

      pcl::PointXYZRGB pt(255, 255, 0);
      pt.x = direction(0);
      pt.y = direction(1);
      pt.z = direction(2);
      original_cloud->points.push_back(pt);
    }
  }

  std::cout << "original_cloud->points.size() : " << original_cloud->points.size() << std::endl;

  // Plane extraction
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*original_cloud, *cloud_filtered);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  planeDetect(cloud_filtered, inliers, coefficients, 0.02);
  std::cout << "inliers->indices.size() : " << inliers->indices.size() << std::endl;
  planeRemoval(cloud_filtered, inliers, false);

  // Creating the KdTree object for the search method of the extraction
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_mono(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud_filtered, *cloud_filtered_mono);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered_mono);

  // Euclidean Clustering
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (height * width);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered_mono);
  ec.extract (cluster_indices);
  std::cout << "cluster_indices.size() : " << cluster_indices.size() << std::endl;
  for (int i = 0; i < cluster_indices.size() ; i++)
  {
    std::cout << "cluster_indices[" << i << "].indices.size() : " << cluster_indices[i].indices.size() << std::endl;
  }
  std::cout << std::endl;

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
  float max_x_pln, max_y_pln, max_z_pln, min_x_pln, min_y_pln, min_z_pln;
  float tmp_max_x_pln, tmp_max_y_pln, tmp_max_z_pln, tmp_min_x_pln, tmp_min_y_pln, tmp_min_z_pln;
  tmp_max_x_pln = tmp_max_y_pln = tmp_max_z_pln = -1000;
  tmp_min_x_pln = tmp_min_y_pln = tmp_min_z_pln = 1000;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_cluster->points.begin(); it != cloud_cluster->points.end(); ++it)
  {
    max_x_pln = std::max(it->x, tmp_max_x_pln);
    max_y_pln = std::max(it->y, tmp_max_y_pln);
    max_z_pln = std::max(it->z, tmp_max_z_pln);
    min_x_pln = std::min(it->x, tmp_min_x_pln);
    min_y_pln = std::min(it->y, tmp_min_y_pln);
    min_z_pln = std::min(it->z, tmp_min_z_pln);
    tmp_max_x_pln = max_x_pln;
    tmp_max_y_pln = max_y_pln;
    tmp_max_z_pln = max_z_pln;
    tmp_min_x_pln = min_x_pln;
    tmp_min_y_pln = min_y_pln;
    tmp_min_z_pln = min_z_pln;
  }

  // Compute centroid
  Eigen::Vector4f pln_centroid;
  pcl::compute3DCentroid(*cloud_cluster, pln_centroid);

  std::cout << "cloud_cluster :" << std::endl;
  std::cout << "  - centroid : (" << pln_centroid[0] << ", " << pln_centroid[1] << ", " << pln_centroid[2] << ")" << std::endl;
  std::cout << "  - x range : " << min_x_pln << " ~ " << max_x_pln << std::endl;
  std::cout << "  - y range : " << min_y_pln << " ~ " << max_y_pln << std::endl;
  std::cout << "  - z range : " << min_z_pln << " ~ " << max_z_pln << std::endl << std::endl;

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
  float max_x_obj, max_y_obj, max_z_obj, min_x_obj, min_y_obj, min_z_obj;
  float tmp_max_x_obj, tmp_max_y_obj, tmp_max_z_obj, tmp_min_x_obj, tmp_min_y_obj, tmp_min_z_obj;
  tmp_max_x_obj = tmp_max_y_obj = tmp_max_z_obj = -1000;
  tmp_min_x_obj = tmp_min_y_obj = tmp_min_z_obj = 1000;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = rotated_obj_cloud->points.begin(); it != rotated_obj_cloud->points.end(); ++it)
  {
    max_x_obj = std::max(it->x, tmp_max_x_obj);
    max_y_obj = std::max(it->y, tmp_max_y_obj);
    max_z_obj = std::max(it->z, tmp_max_z_obj);
    min_x_obj = std::min(it->x, tmp_min_x_obj);
    min_y_obj = std::min(it->y, tmp_min_y_obj);
    min_z_obj = std::min(it->z, tmp_min_z_obj);
    tmp_max_x_obj = max_x_obj;
    tmp_max_y_obj = max_y_obj;
    tmp_max_z_obj = max_z_obj;
    tmp_min_x_obj = min_x_obj;
    tmp_min_y_obj = min_y_obj;
    tmp_min_z_obj = min_z_obj;
  }

  // Move rotated_obj_cloud onto the plane
  float max_x = max_x_pln - max_x_obj;
  float max_y = max_y_pln - max_y_obj;
  float min_x = min_x_pln - min_x_obj;
  float min_y = min_y_pln - min_y_obj;
  std::uniform_real_distribution<float> mesh_orig_x_(min_x, max_x);
  std::uniform_real_distribution<float> mesh_orig_y_(min_y, max_y);
  float mesh_orig_x = mesh_orig_x_(mt);
  float mesh_orig_y = mesh_orig_y_(mt);
  float mesh_orig_z = pln_centroid[2] - min_z_obj;
  Eigen::Matrix4f transform = R;
  transform(0, 3) = mesh_orig_x;
  transform(1, 3) = mesh_orig_y;
  transform(2, 3) = mesh_orig_z;
  std::cout << "transform :" << std::endl << transform << std::endl << std::endl;
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

  // View PointCloud
  // pcl::visualization::CloudViewer viewer0("original_cloud");
  // viewer0.showCloud (original_cloud);
  // while (!viewer0.wasStopped()) {}
  // pcl::visualization::CloudViewer viewer1("cloud_cluster");
  // viewer1.showCloud (cloud_cluster);
  // while (!viewer1.wasStopped()) {}
  // pcl::visualization::CloudViewer viewer2("obj_cloud");
  // viewer2.showCloud (obj_cloud);
  // while (!viewer2.wasStopped()) {}
  pcl::visualization::CloudViewer viewer3("obj_cloud_on_plane");
  viewer3.showCloud (obj_cloud_on_plane);
  while (!viewer3.wasStopped()) {}

  // Save as .pcd file
  std::cout << "cloud_cluster->points.size() : " << cloud_cluster->points.size() << std::endl;
  std::string out_file = data_path + "cloud_obj_n.pcd";
  pcl::io::savePCDFile(out_file, *cloud_cluster);
  std::cout << "Wrote point cloud to: " << out_file << std::endl;

  original_cloud->clear();
  cloud_filtered->clear();
  cloud_filtered_mono->clear();
  cloud_cluster->clear();
  obj_cloud->clear();
  obj_cloud_on_plane->clear();

  return (0);
}
