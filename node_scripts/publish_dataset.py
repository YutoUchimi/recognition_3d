#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import skimage.io
import yaml

import cv_bridge
import dynamic_reconfigure.server
import genpy.message
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from recognition_3d.cfg import PublishDatasetConfig


class DatasetCollectedOnShelfMultiViewScenes(object):

    def __init__(self):
        self.scene_ids = []
        self.root = osp.expanduser('~/data/mvtk/transparent_objects/raw_data_sample/f49d6c4b75f695c44d34bdc365023cf4/scenes/for_annotation')  # NOQA
        for scene_id in sorted(os.listdir(self.root)):
            self.scene_ids.append(scene_id)

    def __len__(self):
        return len(self.scene_ids)

    def get_frame(self, scene_idx):
        assert 0 <= scene_idx < len(self.scene_ids)
        scene_id = self.scene_ids[scene_idx]
        scene_dir = osp.join(self.root, scene_id)
        img = skimage.io.imread(osp.join(scene_dir, 'rgb_image.jpg'))
        depth = np.load(osp.join(scene_dir, 'depth.npz'))['arr_0']
        camera_info = yaml.load(
            open(osp.join(scene_dir,
                          'camera_info.yaml')))
        return img, depth, camera_info


class PublishDataset(object):

    def __init__(self):
        self._dataset = DatasetCollectedOnShelfMultiViewScenes()

        self._config_srv = dynamic_reconfigure.server.Server(
            PublishDatasetConfig, self._config_cb)

        self.pub_rgb = rospy.Publisher(
            '~output/rgb/image_rect_color', Image, queue_size=1)
        self.pub_rgb_cam_info = rospy.Publisher(
            '~output/rgb/camera_info', CameraInfo, queue_size=1)
        self.pub_depth = rospy.Publisher(
            '~output/depth_registered/image_rect', Image, queue_size=1)
        self.pub_depth_cam_info = rospy.Publisher(
            '~output/depth_registered/camera_info', CameraInfo, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(1. / 30), self._timer_cb)

    def _config_cb(self, config, level):
        self._scene_idx = config.scene_idx
        return config

    def _timer_cb(self, event):
        img, depth, cam_info = self._dataset.get_frame(
            self._scene_idx)[0:3]

        cam_info_msg = CameraInfo()
        genpy.message.fill_message_args(cam_info_msg, cam_info)
        cam_info_msg.header.stamp = event.current_real

        bridge = cv_bridge.CvBridge()

        imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
        imgmsg.header.frame_id = cam_info_msg.header.frame_id
        imgmsg.header.stamp = event.current_real

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32)
            depth *= 0.001
        depth_msg = bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.frame_id = cam_info_msg.header.frame_id
        depth_msg.header.stamp = event.current_real

        self.pub_rgb.publish(imgmsg)
        self.pub_rgb_cam_info.publish(cam_info_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_depth_cam_info.publish(cam_info_msg)


if __name__ == '__main__':
    rospy.init_node('publish_dataset')
    app = PublishDataset()
    rospy.spin()
