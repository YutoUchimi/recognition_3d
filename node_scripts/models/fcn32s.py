from __future__ import division


import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np


class FCN32s(chainer.Chain):

    # [0.2, 3]
    min_depth = 0.2
    max_depth = 3.

    def __init__(self, n_class):
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN32s, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 100, **kwargs)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, **kwargs)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, **kwargs)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            self.mask_fr = L.Convolution2D(4096, 2, 1, 1, 0, **kwargs)
            self.depth_fr = L.Convolution2D(4096, 1, 1, 1, 0, **kwargs)

            self.mask_upscore = L.Deconvolution2D(
                2, 2, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.depth_upscore = L.Deconvolution2D(
                1, 1, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def __call__(self, x, mask=None, depth=None):
        # conv1
        h = F.relu(self.conv1_1(x))
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1))
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1))
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1))
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2))
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1))
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2))
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5)
        fc7 = h  # 1/32

        # mask_fr
        h = self.mask_fr(fc7)
        mask_fr = h  # 1/32

        # depth_fr
        h = self.depth_fr(fc7)
        depth_fr = h  # 1/32

        # upscore
        h = self.mask_upscore(mask_fr)
        h = h[:, :, 19:19 + x.shape[2], 19:19 + x.shape[3]]
        self.h_mask_score = h  # 1/1

        # depth upscore
        h = self.depth_upscore(depth_fr)
        h = h[:, :, 19:19 + x.shape[2], 19:19 + x.shape[3]]  # (-inf, inf)
        self.h_depth_upscore = h  # 1/1

        if mask is None or depth is None:
            assert not chainer.config.train
            return

        # segmentation loss
        seg_loss = F.softmax_cross_entropy(
            self.h_mask_score, mask, normalize=True)

        # depth loss
        h = F.sigmoid(self.h_depth_upscore)  # (0, 1)
        h = h[:, 0, :, :]  # N, 1, H, W -> N, H, W
        assert mask.dtype == np.int32
        keep = self.xp.logical_and(mask > 0, ~self.xp.isnan(depth))
        keep = cuda.to_cpu(keep)
        if keep.sum() == 0:
            depth_loss = 0
        else:
            depth_scaled = depth.copy()
            depth_scaled -= self.min_depth
            depth_scaled /= (self.max_depth - self.min_depth)
            # depth_loss = F.mean_squared_error(h[keep], depth_scaled[keep])
            depth_loss = F.mean_absolute_error(h[keep], depth_scaled[keep])

        # depth += np.finfo(np.float32).eps
        # depth = F.log(depth)
        # self.depth_score += np.finfo(np.float32).eps
        # self.depth_score = F.log(self.depth_score)
        # keep_indices = self.xp.logical_and(
        #     self.xp.logical_and(
        #         ~self.xp.isnan(depth.data),
        #         ~self.xp.isinf(depth.data)),
        #     self.xp.logical_and(
        #         ~self.xp.isnan(self.depth_score.data),
        #         ~self.xp.isinf(self.depth_score.data)))
        # depth = depth[keep_indices]
        # self.depth_score = self.depth_score[keep_indices]
        # if depth.data.size == 0:  # this causes nan value as depth_loss
        #     depth.data = self.xp.zeros((1, 1), dtype=np.float32)
        #     self.depth_score.data = self.xp.zeros((1, 1), dtype=np.float32)
        # depth_loss = F.mean_squared_error(
        #     depth, self.depth_score)

        batch_size = len(x)
        assert batch_size == 1

        # N, C, H, W -> C, H, W
        mask = cuda.to_cpu(mask)[0]
        mask_pred = cuda.to_cpu(F.argmax(self.h_mask_score, axis=1).data)[0]
        depth = cuda.to_cpu(depth)[0]

        # (0, 1) -> (min, max)
        depth_pred = cuda.to_cpu(h.data)[0]
        depth_pred *= (self.max_depth - self.min_depth)
        depth_pred += self.min_depth

        # Evaluate Mask IU
        # TODO(YutoUchimi): Support multiclass problem (n_class > 2).
        mask_iu = fcn.utils.label_accuracy_score(
            [mask], [mask_pred], n_class=2)[2]

        # Evaluate Depth Accuracy
        depth_acc = {}
        for thresh in [0.01, 0.02, 0.05]:
            mask_fg = np.logical_and(mask > 0, ~np.isnan(depth))
            if mask_fg.sum() == 0:
                acc = np.nan
            else:
                is_correct = \
                    np.abs(depth[mask_fg] - depth_pred[mask_fg]) < thresh
                acc = 1. * is_correct.sum() / is_correct.size
            depth_acc['%.2f' % thresh] = acc

        # TODO(YutoUchimi): What is proper loss function (alpha)?
        alpha = 100
        loss = seg_loss + alpha * depth_loss
        if np.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')

        chainer.reporter.report({
            'loss': loss,
            'seg_loss': seg_loss,
            'depth_loss': depth_loss,
            'mask_iu': mask_iu,
            'depth_acc<0.01': depth_acc['0.01'],
            'depth_acc<0.02': depth_acc['0.02'],
            'depth_acc<0.05': depth_acc['0.05'],
        }, self)

        return loss

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vTElpa1p3WFNDczQ',
            path=cls.pretrained_model,
            md5='b7f0a2e66229ccdb099c0a60b432e8cf',
        )

    def init_from_vgg16(self, vgg16):
        for l in self.children():
            if l.name.startswith('conv'):
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data[...] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name in ['fc6', 'fc7']:
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]
