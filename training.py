#!/usr/bin/env python3

import time
import os
import sys
import socket
import logging
import logging.config
import subprocess

import tensorflow as tf
import numpy as np
# import matplotlib
# matplotlib.use('Agg')

from model.resnet import ResNet
from utils.utils_tf import print_variables
from dataset.data_provider import DataHandler
from configs.config import get_logging_config, args, train_dir

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()

# Fixing the random seeds
# tf.set_random_seed(1234)
# np.random.seed(123)


def objective(logits, labels, n_classes):
    """Defines classification loss and useful metrics"""

    # defining the loss
    cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entr)
    inferred_class = tf.cast(tf.argmax(logits, 1), tf.int32)
    tf.summary.scalar('loss', loss)

    # defining the metrics
    inferred_class = tf.cast(tf.argmax(logits, 1), tf.int32)
    positive_matches = tf.equal(inferred_class, labels)
    precision = tf.reduce_mean(tf.boolean_mask(tf.cast(positive_matches, tf.float32),
                                               tf.cast(inferred_class, tf.bool)))
    recall = tf.reduce_mean(tf.boolean_mask(tf.cast(positive_matches, tf.float32),
                                            tf.cast(labels, tf.bool)))
    train_acc = tf.reduce_mean(tf.cast(positive_matches, tf.float32))
    tf.summary.scalar('metrics/accuracy', train_acc)
    tf.summary.scalar('metrics/precision', precision)
    tf.summary.scalar('metrics/recall', recall)

    # adding up all losses
    the_loss = loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss/weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss/full', the_loss)
    return the_loss, train_acc


def extract_batch(datahand):
    """Extracts a batch from the queue and applies processing"""

    def set_shape(t):
        t_shape = [args.batch_size] + t.get_shape().as_list()[1:]
        t.set_shape(t_shape)

    with tf.device("/cpu:0"):
        sample = datahand.get_feedable_iterator(args.dataset, args.split)
        for t in sample.values():
            # shape info is lost somehow. Restoring it like this.
            set_shape(t)
        return sample['img'], sample['label']


def train(sess, datahand, net):
    """Initialization and training routines"""

    image_ph, labels_ph = extract_batch(datahand)
    tf.summary.image('augmented_images', image_ph, max_outputs=2)

    logits = net.build_net(image_ph, datahand.num_classes)

    loss, train_acc = objective(logits, labels_ph, datahand.num_classes)

    # setting up the learning rate
    global_step = tf.train.get_or_create_global_step()
    learning_rate = args.learning_rate

    learning_rates = [args.warmup_lr, learning_rate]
    steps = [args.warmup_step]

    if len(args.lr_decay) > 0:
        for i, step in enumerate(args.lr_decay):
            steps.append(step)
            learning_rates.append(learning_rate*10**(-i-1))

    learning_rate = tf.train.piecewise_constant(tf.to_int32(global_step),
                                                steps, learning_rates)

    tf.summary.scalar('learning_rate', learning_rate)
    #######

    # Defining an optimizer
    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif args.optimizer == 'nesterov':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError
    ######

    train_vars = tf.trainable_variables()
    print_variables('train', train_vars)

    train_op = slim.learning.create_train_op(
        loss, opt,
        global_step=global_step,
        variables_to_train=train_vars,
        summarize_gradients=args.summarize_gradients)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000,
                           keep_checkpoint_every_n_hours=1)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(os.path.join(train_dir,
                                                            'val'), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if args.random_trunk_init:
        print("Training from scratch")
    else:
        net.imagenet_init(opt, sess)
    net.restore_ckpt(sess, saver)

    starting_step = sess.run(global_step)
    tf.get_default_graph().finalize()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    log.info("Launching prefetch threads")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    log.info("Starting training...")
    for step in range(starting_step, args.max_iterations+1):
        start_time = time.time()
        try:
            train_loss, acc, lr = sess.run([train_op, train_acc,
                                            learning_rate],
                                           datahand.train_fd)
        except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
            break
        duration = time.time() - start_time

        num_examples_per_step = args.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss = %.2f, acc = %.2f, lr=%.3f'
                      '(%.1f examples/sec; %.3f sec/batch)')
        log.info(format_str % (step, train_loss, acc, -np.log10(lr),
                               examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op, datahand.train_fd)
            val_summary_str = sess.run(summary_op, datahand.val_fd)
            summary_writer.add_summary(summary_str, step)
            val_summary_writer.add_summary(val_summary_str, step)

        if step % 1000 == 0 and step > 0:
            summary_writer.flush()
            val_summary_writer.flush()
            log.debug("Saving checkpoint...")
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

    summary_writer.close()

    coord.request_stop()
    coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    net = ResNet(depth=50, training=True, weight_decay=args.weight_decay)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        datahand = DataHandler(sess)
        train(sess, datahand, net)


if __name__ == '__main__':
    exec_string = ' '.join(sys.argv)
    log.debug("Executing a command: %s", exec_string)
    cur_commit = subprocess.check_output("git log -n 1 --pretty=format:\"%H\"".split())
    cur_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    git_diff = subprocess.check_output('git diff --no-color'.split()).decode('ascii')
    log.debug("on branch %s with the following diff from HEAD (%s):" % (cur_branch, cur_commit))
    log.debug(git_diff)
    hostname = socket.gethostname()
    if 'gpuhost' in hostname:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        nvidiasmi = subprocess.check_output('nvidia-smi').decode('ascii')
        log.debug("Currently we are on %s and use gpu%s:" % (hostname, gpu_id))
        log.debug(nvidiasmi)
    tf.app.run()
