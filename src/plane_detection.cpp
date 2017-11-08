#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

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


int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

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
  std::cout << "cam_pose : " << std::endl << cam_pose << std::endl;

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

      direction_ = cam_pose * direction_;
      Eigen::Vector3f direction(direction_(0), direction_(1), direction_(2));

      pcl::PointXYZRGB pt(255, 255, 255);
      pt.x = direction(0);
      pt.y = direction(1);
      pt.z = direction(2);
      cloud->points.push_back(pt);
    }
  }
  pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  viewer.showCloud (cloud);
  while (!viewer.wasStopped ()) {}

  planeDetect(cloud, inliers, coefficients, 0.01);
  planeRemoval(cloud, inliers, false);

  // pcl::PCDWriter writer;
  // writer.write("cloud_out.pcd", *cloud, false);
  cloud->clear();

  return (0);
}
