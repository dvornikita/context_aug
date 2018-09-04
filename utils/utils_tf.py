import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import logging

from configs.config import args, std_data_augmentation_config

log = logging.getLogger()


def xywh_to_xyxy(xywh):
    x, y, w, h = tf.unstack(xywh, axis=1)
    return tf.stack([x, y, x+w, y+h], axis=1)


def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = tf.unstack(xyxy, axis=1)
    return tf.stack([x1, y1, x2-x1, y2-y1], axis=1)


def central_to_xywh(bbox):
    xc, yc, w, h = tf.unstack(bbox)
    xmin = tf.round(xc - w / 2)
    ymin = tf.round(yc - h / 2)
    return tf.stack([xmin, ymin, w, h])


def xywh_to_central(bbox):
    xmin, ymin, w, h = tf.unstack(bbox)
    xc = xmin + w / 2
    yc = ymin + h / 2
    return tf.stack([xc, yc, w, h])


def photometric_distortions(image, color_ordering, params, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
        else:

            raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def mirror_distortions(image, params):
    return tf.cond(tf.random_uniform([], 0, 1.0) < params['flip_prob'],
                   lambda: tf.image.flip_left_right(image),
                   lambda: image)


def fill_and_crop(image, bbox, frame, wi, he, params):
    xmin, ymin, w, h = tf.unstack(tf.cast(tf.squeeze(bbox) * tf.to_float(
        tf.stack([wi, he, wi, he])), tf.int32))
    xs = tf.range(ymin, ymin + h)
    ys = tf.range(xmin, xmin + w)

    inds = tf.meshgrid(xs, ys, indexing='xy')
    inds = tf.stack(inds, axis=0)
    inds = tf.transpose(tf.reshape(inds, (2, -1)))

    update = tf.constant([1, 1, 1], dtype=tf.float32)
    updates = tf.reshape(update, (1, -1))
    updates = tf.tile(updates, [tf.shape(inds)[0], 1])

    mask = tf.scatter_nd(inds, updates, tf.shape(image))
    inverse_mask = 1 - mask

    color = tf.constant(params['fill_color'], dtype=tf.float32)
    image_erased = image * inverse_mask + mask * color

    x, y, w, h = tf.unstack(frame)
    frame_tf = tf.reshape(tf.stack([y, x, y + h, x + w]), (1, 4))
    processed_image = tf.image.crop_and_resize(image_erased[None, ...],
                                               frame_tf, [0],
                                               [args.image_size] * 2)
    return tf.squeeze(processed_image)


def data_augmentation(sample, is_training=True):
    """Performs D.A and crops out the context image"""
    img = sample['img']
    params = std_data_augmentation_config
    if is_training:
        img = apply_with_random_selector(
            img,
            lambda x, ordering: photometric_distortions(x, ordering, params),
            num_cases=4)

    img = fill_and_crop(img, sample['bbox'], sample['frame'],
                        sample['w'], sample['h'], params)
    img = mirror_distortions(img, params)
    return img, sample['label']


def batch_iou_tf(proposals, gt):
    bboxes = tf.reshape(tf.transpose(proposals), [4, -1, 1])
    bboxes_x1 = bboxes[0]
    bboxes_x2 = bboxes[0]+bboxes[2]
    bboxes_y1 = bboxes[1]
    bboxes_y2 = bboxes[1]+bboxes[3]

    gt = tf.reshape(tf.transpose(gt), [4, 1, -1])
    gt_x1 = gt[0]
    gt_x2 = gt[0]+gt[2]
    gt_y1 = gt[1]
    gt_y2 = gt[1]+gt[3]

    widths = tf.maximum(0.0, tf.minimum(bboxes_x2, gt_x2) -
                        tf.maximum(bboxes_x1, gt_x1))
    heights = tf.maximum(0.0, tf.minimum(bboxes_y2, gt_y2) -
                         tf.maximum(bboxes_y1, gt_y1))
    intersection = widths*heights
    union = bboxes[2]*bboxes[3] + gt[2]*gt[3] - intersection
    return (intersection / union)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def py_deb(arr):
    import ipdb; ipdb.set_trace()
    return arr


def print_variables(name, var_list, level=logging.DEBUG):
    """Handy tool for printing vars"""
    variables = sorted([v.op.name for v in var_list])
    s = "Variables to %s:\n%s" % (name, '\n'.join(variables))
    if level < 0:
        print(s)
    else:
        log.log(level, s)
