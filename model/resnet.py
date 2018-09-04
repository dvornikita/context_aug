import logging
import os

import tensorflow as tf

from model import resnet_v1
from model import resnet_utils

from model.resnet_v1 import bottleneck
from utils.utils_tf import print_variables
from configs.config import args, MEAN_COLOR, train_dir
from configs.paths import INIT_WEIGHTS_DIR


log = logging.getLogger()
slim = tf.contrib.slim

CKPT_50 = os.path.join(INIT_WEIGHTS_DIR, 'resnet50_full.ckpt')
DEFAULT_SCOPE_50 = 'resnet_v1_50'


class ResNet(object):
    def __init__(self, training, weight_decay=0.0005, depth=50,
                 reuse=False):
        self.weight_decay = weight_decay
        self.reuse = reuse
        self.training = training
        if depth == 50:
            self.num_block3 = 5
            self.scope = DEFAULT_SCOPE_50
            self.ckpt = CKPT_50
        else:
            raise ValueError

    def create_trunk(self, images, global_pool=True):
        red, green, blue = tf.split(images*255, 3, axis=3)
        images = tf.concat([blue, green, red], 3) - MEAN_COLOR

        with slim.arg_scope(resnet_v1.resnet_arg_scope(
                is_training=self.training, weight_decay=self.weight_decay,
                batch_norm_decay=args.bn_decay)):
            blocks = [
                resnet_utils.Block(
                    'block1', bottleneck, [(256, 64, 1)] * 3),
                resnet_utils.Block(
                    'block2', bottleneck, [(512, 128, 2)] + [(512, 128, 1)] * 3),
                resnet_utils.Block(
                    'block3', bottleneck, [(1024, 256, 2)] + [(1024, 256, 1)] * self.num_block3),
                resnet_utils.Block(
                    'block4', bottleneck, [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
            ]

            net, endpoints = resnet_v1.resnet_v1(images, blocks,
                                                 global_pool=global_pool,
                                                 reuse=self.reuse,
                                                 scope=self.scope)
            self.outputs = endpoints

    def build_net(self, images, n_classes):
        self.create_trunk(images)
        net = self.outputs['pooled']
        with tf.variable_scope("final", reuse=self.reuse):
            logits = slim.fully_connected(net, n_classes, activation_fn=None)
            self.outputs['logits'] = logits
        return logits

    def imagenet_init(self, opt, sess):
        """optimizer is useful to extract slots corresponding to Adam or Momentum
        and exclude them from checkpoint assigning"""
        variables = slim.get_variables_to_restore(include=[self.scope])
        slots = set()
        for v in tf.trainable_variables():
            for s in opt.get_slot_names():
                slot = opt.get_slot(v, s)
                if slot is not None:
                    slots.add(slot)
        variables = list(set(variables) - slots)
        init_assign_op, init_feed_dict, init_vars = \
                slim.assign_from_checkpoint(self.ckpt, variables) + (variables, )
        print_variables('init from ImageNet', init_vars)
        sess.run(init_assign_op, feed_dict=init_feed_dict)

    def restore_ckpt(self, sess, saver):
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if args.ckpt == 0:
                ckpt_to_restore = ckpt.model_checkpoint_path
            else:
                ckpt_to_restore = train_dir+'/model.ckpt-%i' % args.ckpt
            log.info("Restoring model %s..." % ckpt_to_restore)
            saver.restore(sess, ckpt_to_restore)
