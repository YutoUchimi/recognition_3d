#include "recognition_3d/segment_to_mask_image.h"
#include <jsk_topic_tools/log_utils.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/assign.hpp>

namespace recognition_3d
{

  void SegmentToMaskImage::onInit()
  {
    DiagnosticNodelet::onInit();
    pub_ = advertise<sensor_msgs::Image>(*pnh_, "output", 1);
    onInitPostProcess();
  }

  void SegmentToMaskImage::configCallback(uint32_t level)
  {
    boost::mutex::scoped_lock lock(mutex_);
  }

  void SegmentToMaskImage::subscribe()
  {
    sub_ = pnh_->subscribe("input", 1,
                           &SegmentToMaskImage::convert,
                           this);
    ros::V_string names = boost::assign::list_of("~input");
    jsk_topic_tools::warnNoRemap(names);
  }

  void SegmentToMaskImage::unsubscribe()
  {
    sub_.shutdown();
  }

  void SegmentToMaskImage::convert(const sensor_msgs::Image::ConstPtr& segment_msg)
  {
    cv_bridge::CvImagePtr segment_img_ptr = cv_bridge::toCvCopy(segment_msg, sensor_msgs::image_encodings::BGR8);

    cv::Mat mask_image = cv::Mat::zeros(segment_msg->height,
                                        segment_msg->width,
                                        CV_8UC1);
    for (size_t j = 0; j < segment_img_ptr->image.rows; j++)
      {
        for (size_t i = 0; i < segment_img_ptr->image.cols; i++)
          {
            cv::Vec3b rgb = segment_img_ptr->image.at<cv::Vec3b>(j, i);
            if (rgb[0] != 0 || rgb[1] != 0 || rgb[2] != 0) {
              mask_image.at<uchar>(j, i) = 255;
            }
          }
      }
    pub_.publish(cv_bridge::CvImage(segment_msg->header,
                                    sensor_msgs::image_encodings::MONO8,
                                    mask_image).toImageMsg());
  }

}  // namespace recognition_3d

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(recognition_3d::SegmentToMaskImage, nodelet::Nodelet);
