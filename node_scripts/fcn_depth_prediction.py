#!/usr/bin/env python

from distutils.version import LooseVersion
import warnings

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.serializers as S

import matplotlib.cm
import numpy as np

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import rospy
from sensor_msgs.msg import Image

from models.fcn32s import FCN32s
from models.fcn8s import FCN8sAtOnce
from models.fcn8s import FCN8sAtOnceInputRGBD


class FCNDepthPrediction(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.backend = rospy.get_param('~backend', 'chainer')
        self.model_name = rospy.get_param('~model_name')
        self.model_file = rospy.get_param('~model_file')
        self.gpu = rospy.get_param('~gpu', -1)  # -1 is cpu mode
        self.target_names = rospy.get_param('~target_names')
        self.bg_label = rospy.get_param('~bg_label', 0)
        self.proba_threshold = rospy.get_param('~proba_threshold', 0.0)
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self._load_model()
        self.pub_label = self.advertise('~output/label', Image, queue_size=1)
        self.pub_proba = self.advertise(
            '~output/proba_image', Image, queue_size=1)
        self.pub_depth = self.advertise('~output/depth', Image, queue_size=1)
        self.pub_depth_raw = self.advertise(
            '~output/depth_raw', Image, queue_size=1)

    def _load_model(self):
        if self.backend == 'chainer':
            self._load_chainer_model()
        else:
            raise RuntimeError('Unsupported backend: %s', self.backend)

    def _load_chainer_model(self):
        n_class = len(self.target_names)
        # TODO(YutoUchimi): import model class
        if self.model_name == 'fcn32s':
            self.model = FCN32s(n_class=n_class)
        elif self.model_name == 'fcn8s_at_once':
            self.model = FCN8sAtOnce(n_class=n_class)
        elif self.model_name == 'fcn8s_at_once_input_rgbd':
            self.model = FCN8sAtOnceInputRGBD(n_class=n_class)
        else:
            raise ValueError(
                'Unsupported ~model_name: {}'.format(self.model_name))
        rospy.loginfo('Loading trained model: {0}'.format(self.model_file))
        if self.model_file.endswith('.npz'):
            S.load_npz(self.model_file, self.model)
        rospy.loginfo(
            'Finished loading trained model: {0}'.format(self.model_file))
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            self.model.train = False

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2**24)
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_rgb, sub_depth]
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def colorize_depth(self, depth, min_value=None, max_value=None):
        """Colorize depth image with JET colormap."""
        min_value = np.nanmin(depth) if min_value is None else min_value
        max_value = np.nanmax(depth) if max_value is None else max_value
        if np.isinf(min_value) or np.isinf(max_value):
            warnings.warn('Min or max value for depth colorization is inf.')

        colorized = depth.copy()
        nan_mask = np.isnan(colorized)
        colorized[nan_mask] = 0
        colorized = 1. * (colorized - min_value) / (max_value - min_value)
        colorized = matplotlib.cm.jet(colorized)[:, :, :3]
        colorized = (colorized * 255).astype(np.uint8)
        colorized[nan_mask] = (0, 0, 0)
        return colorized

    def transform_rgb(self, rgb_img):
        # RGB -> BGR
        bgr_img = rgb_img[:, :, ::-1]
        bgr_img = bgr_img.astype(np.float32)
        bgr_img -= self.mean_bgr
        # H, W, C -> C, H, W
        bgr_img = bgr_img.transpose((2, 0, 1))
        return bgr_img

    def transform_depth(self, depth):
        min_value = 0.2
        max_value = 3.0
        depth_viz_rgb = self.colorize_depth(
            depth,
            min_value=min_value, max_value=max_value
        )
        # RGB -> BGR
        depth_viz_bgr = depth_viz_rgb[:, :, ::-1]
        depth_viz_bgr = depth_viz_bgr.astype(np.float32)
        depth_viz_bgr -= self.mean_bgr
        # H, W, C -> C, H, W
        depth_viz_bgr = depth_viz_bgr.transpose((2, 0, 1))
        return depth_viz_bgr

    def _cb(self, rgb_msg, depth_msg):
        br = cv_bridge.CvBridge()
        rgb_img = br.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_img = br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth_img.ndim > 2:
            depth_img = np.squeeze(depth_img, axis=2)
        bgr_img = self.transform_rgb(rgb_img)
        depth_viz_bgr = self.transform_depth(depth_img)

        label_pred, proba_img, \
            depth_pred = self.segment_and_depth_predict(
                bgr_img, depth_viz_bgr)

        depth_pred_raw = depth_pred.copy()
        depth_pred[label_pred == 0] = depth_img[label_pred == 0]

        label_msg = br.cv2_to_imgmsg(label_pred.astype(np.int32), '32SC1')
        label_msg.header = rgb_msg.header
        self.pub_label.publish(label_msg)
        proba_msg = br.cv2_to_imgmsg(proba_img.astype(np.float32))
        proba_msg.header = rgb_msg.header
        self.pub_proba.publish(proba_msg)
        depth_msg = br.cv2_to_imgmsg(depth_pred.astype(np.float32))
        depth_msg.header = rgb_msg.header
        self.pub_depth.publish(depth_msg)
        depth_raw_msg = br.cv2_to_imgmsg(depth_pred_raw.astype(np.float32))
        depth_msg.header = rgb_msg.header
        self.pub_depth_raw.publish(depth_raw_msg)

    def segment_and_depth_predict(self, bgr, depth_bgr):
        if self.backend == 'chainer':
            return self._segment_chainer_backend(bgr, depth_bgr)
        raise ValueError('Unsupported backend: {0}'.format(self.backend))

    def _segment_chainer_backend(self, bgr, depth_bgr):
        bgr_data = np.array([bgr], dtype=np.float32)
        depth_bgr_data = np.array([depth_bgr], dtype=np.float32)
        if self.model.__class__.__name__ == 'FCN8sAtOnceInputRGBD':
            if self.gpu != -1:
                bgr_data = cuda.to_gpu(bgr_data, device=self.gpu)
                depth_bgr_data = cuda.to_gpu(depth_bgr_data, device=self.gpu)
            if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
                bgr = chainer.Variable(bgr_data, volatile=True)
                depth_bgr = chainer.Variable(depth_bgr_data, volatile=True)
                self.model(bgr, depth_bgr)
            else:
                with chainer.using_config('train', False):
                    with chainer.no_backprop_mode():
                        bgr = chainer.Variable(bgr_data)
                        depth_bgr = chainer.Variable(depth_bgr_data)
                        self.model(bgr, depth_bgr)
        else:
            if self.gpu != -1:
                bgr_data = cuda.to_gpu(bgr_data, device=self.gpu)
            if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
                bgr = chainer.Variable(bgr_data, volatile=True)
                self.model(bgr)
            else:
                with chainer.using_config('train', False):
                    with chainer.no_backprop_mode():
                        bgr = chainer.Variable(bgr_data)
                        self.model(bgr)

        if self.model.__class__.__name__ == 'FCN32s':
            proba_img = F.softmax(self.model.h_mask_score)
            proba_img = F.transpose(proba_img, (0, 2, 3, 1))
            max_proba_img = F.max(proba_img, axis=-1)
            label_pred = F.argmax(self.model.h_mask_score, axis=1)
            # squeeze batch axis, gpu -> cpu
            proba_img = cuda.to_cpu(proba_img.data)[0]
            max_proba_img = cuda.to_cpu(max_proba_img.data)[0]
            label_pred = cuda.to_cpu(label_pred.data)[0]
            # uncertain because the probability is low
            label_pred[max_proba_img < self.proba_threshold] = self.bg_label
            # get depth_img
            depth_pred = F.sigmoid(self.model.h_depth_upscore)
            depth_pred = cuda.to_cpu(depth_pred.data)[0]
            depth_pred = depth_pred[0, :, :]
            depth_pred *= (self.model.max_depth - self.model.min_depth)
            depth_pred += self.model.min_depth
        elif (self.model.__class__.__name__ == 'FCN8sAtOnce' or
              self.model.__class__.__name__ == 'FCN8sAtOnceInputRGBD'):
            # get label_pred and proba_img
            proba_img = F.softmax(self.model.mask_score)
            proba_img = F.transpose(proba_img, (0, 2, 3, 1))
            max_proba_img = F.max(proba_img, axis=-1)
            label_pred = F.argmax(self.model.mask_score, axis=1)
            # squeeze batch axis, gpu -> cpu
            proba_img = cuda.to_cpu(proba_img.data)[0]
            max_proba_img = cuda.to_cpu(max_proba_img.data)[0]
            label_pred = cuda.to_cpu(label_pred.data)[0]
            # uncertain because the probability is low
            label_pred[max_proba_img < self.proba_threshold] = self.bg_label
            # get depth_img
            depth_pred = F.sigmoid(self.model.depth_score)
            depth_pred = cuda.to_cpu(depth_pred.data)[0]
            depth_pred = depth_pred[0, :, :]
            depth_pred *= (self.model.max_depth - self.model.min_depth)
            depth_pred += self.model.min_depth

        return label_pred, proba_img, depth_pred


if __name__ == '__main__':
    rospy.init_node('fcn_depth_prediction')
    FCNDepthPrediction()
    rospy.spin()
