#include "ros/ros.h"
#include "visualization_msgs/Marker.h"

int main (int argc, char **argv) {
  ros::init(argc, argv, "mesh_publisher");
  ros::NodeHandle nh;
  ros::Publisher vis_pub = nh.advertise<visualization_msgs::Marker>("mesh_publisher/output", 0);
  ros::Rate loop_rate(50);

  visualization_msgs::Marker marker;

  while (ros::ok()) {
    marker.header.frame_id = "base_link";
    marker.header.stamp = ros::Time();
    marker.ns = "";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;
    marker.color.a = 0.7; // Don't forget to set the alpha!

    marker.mesh_resource = "file:///home/yuto/dataset/mesh_dataset_shapenet/pet_bottle_1/models/model_normalized.stl";
    vis_pub.publish( marker );

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
