import os
import logging
import numpy as np
from math import ceil, floor
from PIL import Image, ImageDraw, ImageFont
from configs.paths import EVAL_DIR

log = logging.getLogger()


colors = np.load(os.path.join(EVAL_DIR, 'Extra/colors.npy')).tolist()
palette = np.load(os.path.join(EVAL_DIR, 'Extra/palette.npy')).tolist()
font = ImageFont.truetype(os.path.join(EVAL_DIR, 'Extra/FreeSansBold.ttf'), 14)


def xy2wh(bbox):
    xmin, ymin, xmax, ymax = bbox
    return np.array([xmin, ymin, xmax-xmin, ymax-ymin])


def wh2xy(bbox):
    xmin, ymin, w, h = bbox
    return np.array([xmin, ymin, xmin+w, ymin+h])


def wh2center(bbox):
    xmin, ymin, w, h = bbox
    xc, yc = bbox[:2] + bbox[2:] / 2
    return np.array([xc, yc, w, h])


def center2wh(bbox):
    xc, yc, w, h = bbox
    x, y = bbox[:2] - bbox[2:] / 2
    return np.array([x, y, w, h])


def nms(dets, scores, thresh=None):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if dets is not None:
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
        else:
            inds = np.where(ovr[i, order[1:]] <= thresh)[0]
        order = order[inds + 1]
    return keep


def batch_iou(proposals, gt, return_union=False, return_intersection=False):
    bboxes = np.transpose(proposals).reshape((4, -1, 1))
    bboxes_x1 = bboxes[0]
    bboxes_x2 = bboxes[0]+bboxes[2]
    bboxes_y1 = bboxes[1]
    bboxes_y2 = bboxes[1]+bboxes[3]

    gt = np.transpose(gt).reshape((4, 1, -1))
    gt_x1 = gt[0]
    gt_x2 = gt[0]+gt[2]
    gt_y1 = gt[1]
    gt_y2 = gt[1]+gt[3]

    widths = np.maximum(0, np.minimum(bboxes_x2, gt_x2) -
                        np.maximum(bboxes_x1, gt_x1))
    heights = np.maximum(0, np.minimum(bboxes_y2, gt_y2) -
                         np.maximum(bboxes_y1, gt_y1))
    intersection = widths*heights
    union = bboxes[2]*bboxes[3] + gt[2]*gt[3] - intersection
    iou = (intersection / union)
    if return_intersection or return_union:
        output = [iou]
        if return_intersection:
            output.append(intersection)
        if return_union:
            output.append(union)
        return tuple(output)
    else:
        return iou


def draw_bbox(img, bboxes=None, scores=None, extra_bboxes=None, extra_scores=None,
              cats=None, show=False, size=None, color='red', text_color='red',
              extra_color='purple', bbox_format='xywh', frame_width=3):
    """Drawing bounding boxes on top of the images fed"""

    def _draw(bboxes, scores, dr, color, cats=None):
        def draw_rectangle(draw, coordinates, color, width=1):
            for i in range(width):
                rect_start = (coordinates[0] - i, coordinates[1] - i)
                rect_end = (coordinates[2] + i, coordinates[3] + i)
                draw.rectangle((rect_start, rect_end), outline=color)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            if bbox_format == 'xyxy':
                x, y, x1, y1 = bbox
            elif bbox_format == 'xywh':
                x, y, w, h = bbox
                x1, y1 = x + w, y + h
            if cats is not None:
                color = colors[cats[i] % len(colors)]
            draw_rectangle(dr, (x, y, x1, y1), color=color, width=frame_width)
            if scores is not None:
                dr.text([x, y], str(scores[i])[:5], fill=color, font=font)

    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, (np.ndarray, np.generic)):
        img = Image.fromarray((img).astype('uint8'))
    if img is None:
        img = Image.new("RGB", size, "white")

    if size is not None:
        img = img.resize(size)

    if bboxes is not None:
        dr = ImageDraw.Draw(img)
        _draw(bboxes, scores, dr, color, cats)
        if extra_bboxes is not None:
            _draw(extra_bboxes, extra_scores, dr, extra_color)
        del dr
    if show:
        img.show()
    return img
