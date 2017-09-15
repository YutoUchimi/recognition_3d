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
    // dynamic_reconfigure
    // srv_ = boost::make_shared <dynamic_reconfigure::Server<Config> > (*pnh_);
    // dynamic_reconfigure::Server<Config>::CallbackType f =
    //   boost::bind(&SegmentToMaskImage::configCallback, this, _1, _2);
    // srv_->setCallback(f);

    pub_ = advertise<sensor_msgs::Image>(*pnh_, "output", 1);
    onInitPostProcess();
  }

  void SegmentToMaskImage::configCallback(
                                          /*Config &config,*/ uint32_t level)
  {
    boost::mutex::scoped_lock lock(mutex_);
    //label_value_ = config.label_value;
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

  void SegmentToMaskImage::convert(
                                 const sensor_msgs::Image::ConstPtr& segment_msg)
  {
    cv_bridge::CvImagePtr segment_img_ptr = cv_bridge::toCvCopy(
                                                                segment_msg, sensor_msgs::image_encodings::TYPE_8UC3);

    cv::Mat mask_image = cv::Mat::zeros(segment_msg->height,
                                        segment_msg->width,
                                        CV_8UC1);
    for (size_t j = 0; j < segment_img_ptr->image.rows; j++)
      {
        for (size_t i = 0; i < segment_img_ptr->image.cols; i++)
          {
            int segment = segment_img_ptr->image.at<int>(j, i);
            if (segment != 0) {
              mask_image.at<uchar>(j, i) = 255;
            }
          }
      }
    pub_.publish(cv_bridge::CvImage(
                                    segment_msg->header,
                                    sensor_msgs::image_encodings::MONO8,
                                    mask_image).toImageMsg());
  }

}  // namespace recognition_3d

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(recognition_3d::SegmentToMaskImage, nodelet::Nodelet);
