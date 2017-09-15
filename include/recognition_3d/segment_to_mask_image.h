#ifndef RECOGNITION_3D_SEGMENT_TO_MASK_IMAGE_H_
#define RECOGNITION_3D_SEGMENT_TO_MASK_IMAGE_H_

#include <jsk_topic_tools/diagnostic_nodelet.h>
//#include <dynamic_reconfigure/server.h>
//#include <recognition_3d/SegmentToMaskImageConfig.h>
#include <sensor_msgs/Image.h>

namespace recognition_3d
{

  class SegmentToMaskImage: public jsk_topic_tools::DiagnosticNodelet
  {
  public:
    //typedef recognition_3d::SegmentToMaskImageConfig Config;
    SegmentToMaskImage(): DiagnosticNodelet("SegmentToMaskImage") { }
  protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();
    virtual void convert(const sensor_msgs::Image::ConstPtr& segment_msg);
    virtual void configCallback(/*Config &config,*/ uint32_t level);

    boost::mutex mutex_;

    ros::Subscriber sub_;
    ros::Publisher pub_;
    //boost::shared_ptr<dynamic_reconfigure::Server<Config> > srv_;
    //int label_value_;
  private:
  };

}  // namespace recognition_3d

#endif // RECOGNITION_3D_SEGMENT_TO_MASK_IMAGE_H_
