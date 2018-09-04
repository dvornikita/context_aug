#!/usr/bin/env python3

import progressbar
import logging
import logging.config
import os

import tensorflow as tf
import numpy as np

from model.resnet import ResNet
from dataset.voc_loader import VOCLoader
from dataset.instance_sampler import InstanceSampler
from utils.utils_tf import fill_and_crop
from configs.paths import CKPT_DIR, RAW_CONTEXT_DIR
from configs.config import (get_logging_config, args,
                            std_data_augmentation_config)

# import matplotlib
# matplotlib.rcParams['backend'] = "Qt4Agg"


slim = tf.contrib.slim

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()


class InferenceModel(object):
    def __init__(self, sess, net, sampler, img_size,
                 folder=None, context_estimation=False):
        self.sess = sess
        self.net = net
        self.sampler = sampler
        self.img_size = img_size
        self.build_context_estimator()

    def restore_from_ckpt(self, ckpt):
        ckpt_path = os.path.join(CKPT_DIR, args.run_name, 'model.ckpt-%i000' % ckpt)
        log.debug("Restoring checkpoint %s" % ckpt_path)
        self.sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, ckpt_path)

    def build_context_estimator(self):
        b = args.test_batch_size
        self.images_ph = tf.placeholder(shape=[b, None, None, 3],
                                        dtype=tf.float32, name='img_ph')
        self.bboxes_ph = tf.placeholder(shape=[b, 4],
                                        dtype=tf.float32, name='bboxes_ph')
        self.frames_ph = tf.placeholder(shape=[b, 4],
                                        dtype=tf.float32, name='frames_ph')
        self.ws = tf.placeholder(shape=[b],
                                 dtype=tf.float32, name='ws_ph')
        self.hs = tf.placeholder(shape=[b],
                                 dtype=tf.float32, name='hs_ph')

        def fn(x):
            return fill_and_crop(x[0], x[1], x[2], x[3], x[4],
                                 std_data_augmentation_config)
        imgs = tf.map_fn(fn, [self.images_ph, self.bboxes_ph,
                              self.frames_ph, self.ws, self.hs],
                         tf.float32, parallel_iterations=4, back_prop=False)
        self.logits = self.net.build_net(imgs, self.sampler.num_classes)
        self.output_probs = tf.nn.softmax(self.logits)

    def estimate_context(self, imgs, bboxes, frames, ws, hs):
        final_probs = []
        b = args.test_batch_size
        n_iters = int(np.floor(imgs.shape[0] / b))
        for i in range(n_iters):
            inds = np.arange(b*i, b*(i + 1))
            feed_dict = {self.images_ph: imgs[inds],
                         self.bboxes_ph: bboxes[inds],
                         self.frames_ph: frames[inds],
                         self.ws: ws[inds],
                         self.hs: hs[inds]}

            probs = self.sess.run(self.output_probs, feed_dict=feed_dict)
            final_probs.append(probs)
        final_probs = np.concatenate(final_probs, axis=0)
        final_bboxes = np.array(bboxes * np.vstack([ws, hs, ws, hs]).T,
                                dtype=int)

        # If sampling more than one context image per box
        # averaging scores over them
        if args.n_neighborhoods > 1:
            nn = args.n_neighbors
            final_bboxes = final_bboxes[::nn]
            all_probs = [final_probs['probs'][i::nn] for i in range(nn)]
            final_probs = np.stack(all_probs, -1).mean(-1)
        return final_probs, final_bboxes


def sample2batch(s):
    N = s['bboxes'].shape[0]
    ws = np.array([s['w']] * N)
    hs = np.array([s['h']] * N)
    imgs = np.array([s['img']] * N)
    return imgs, s['bboxes'], s['frames'], ws, hs


def main(argv=None):  # pylint: disable=unused-argument
    assert args.ckpt > 0
    assert args.test_n % args.n_neighborhoods == 0, "test_n has to be a multiple of n_neighborhoods"
    net = ResNet

    net = net(training=False)

    # extracting cats to exclude
    excluded = [int(c) for c in args.excluded.split('_')] if args.excluded != "" else []

    dataset, split = args.dataset, args.split

    if '0712' in dataset:
        loader = VOCLoader(['07', '12'], 'train', True, subsets=args.subsets,
                           cats_exclude=excluded, cut_bad_names=False)
    elif '12' in dataset:
        loader = VOCLoader('12', split, args.small_data,
                           cats_exclude=excluded,
                           cut_bad_names=False)
    elif '7' in dataset:
        loader = VOCLoader('07', split, cats_exclude=excluded,
                           cut_bad_names=False)
    sampler = InstanceSampler(loader=loader,
                              n_neighborhoods=args.n_neighborhoods)

    suff = '_small' if args.small_data else ''
    context_folder = os.path.join(RAW_CONTEXT_DIR, args.run_name + '-' + dataset
                                  + split + suff + '-%dneib' % args.n_neighborhoods)
    if not os.path.exists(context_folder):
        os.makedirs(context_folder)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        estimator = InferenceModel(sess, net, sampler, args.image_size)
        estimator.restore_from_ckpt(args.ckpt)
        bar = progressbar.ProgressBar()
        for name in bar(loader.filenames):
            save_file = os.path.join(context_folder, name)
            if os.path.exists(save_file):
                continue
            sample = sampler.get_test_sample(name, args.test_n)
            imgs, bboxes, frames, ws, hs = sample2batch(sample)
            probs, bboxes_out = estimator.estimate_context(imgs, bboxes,
                                                           frames, ws, hs)
            context_dict = {'bboxes': bboxes_out, 'probs': probs}
            np.save(save_file, context_dict)
        print('DONE')


if __name__ == '__main__':
    tf.app.run()
