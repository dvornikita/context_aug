#!/usr/bin/python3

import sys
import os
import argparse
import numpy as np
import progressbar
from PIL import Image
from glob import glob

sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
from dataset.voc_loader import VOCLoader, VOC_CATS
from configs.paths import DATASETS_DIR, check_dir
from utils.utils_bbox import batch_iou, draw_bbox
from utils.utils_general import read_textfile, write_file, softlink

parser = argparse.ArgumentParser(description='Train or eval SSD model with goodies.')
parser.add_argument("--split", default='train', choices=['train', 'val'])
args = parser.parse_args()


def copy_images(loader, dest_folder):
    bar = progressbar.ProgressBar()
    for name in bar(loader.filenames):
        source = loader.get_image_path(name)
        dest = os.path.join(dest_folder, name + '.jpg')
        softlink(source, dest)


def get_ends(coords):
    # to fight weird artifacts if possible
    if len(coords) > 4:
        c_min, c_max = coords[2] - 2, coords[-3] + 2
    elif len(coords) > 2:
        c_min, c_max = coords[1] - 1, coords[-2] + 1
    else:
        c_min, c_max = np.min(coords), np.max(coords)
    return c_min, c_max


def extract_instances_from_image_voc(name, loader, instance_root):
    inst_mask = np.array(Image.open(os.path.join(instance_root, name + '.png')))
    gt_bboxes, _, gt_cats, _, _, _ = loader.read_annotations(name)
    instance_inds = np.unique(inst_mask)
    instance_inds = instance_inds[(instance_inds != 0) * (instance_inds != 255)]

    masks, cats = [], []
    for iind in instance_inds:
        mask = np.ones(inst_mask.shape[:2], dtype=bool)
        mask[inst_mask == iind] = 0
        coords_y, coords_x = map(np.unique, np.where(mask == 0))
        x_min, y_min, x_max, y_max = 0, 0, mask.shape[0], mask.shape[1]
        if len(coords_x) > 0:
            x_min, x_max = get_ends(coords_x)
        if len(coords_y) > 0:
            y_min, y_max = get_ends(coords_y)
        mask_box = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        mask_box = np.reshape(mask_box, (-1, 4))
        ious = batch_iou(gt_bboxes, mask_box).ravel()
        best_match = np.argmax(ious)
        cat = gt_cats[best_match]

        masks.append(mask)
        cats.append(cat)
    return masks, cats


def extract_instances_from_image_coco(name, loader):
    masks, cats = loader.get_instance_segmentation(name)
    inds = [c in loader.cats for c in cats]
    masks = [masks[i] for i in range(len(cats)) if inds[i]]
    cats = [cats[i] for i in range(len(cats)) if inds[i]]
    return masks, cats


if __name__ == '__main__':
    if args.split == 'train':
        loader_voc = VOCLoader('12', 'train', True, True)
    else:
        loader_voc = VOCLoader('12', 'val', True)

    # setting up paths
    root = os.path.join(DATASETS_DIR, 'VOC12_instances')
    instance_root = os.path.join(loader_voc.roots[0], 'SegmentationObject')
    objects_dir = os.path.join(root, 'cut_objects')
    backgrounds_dir = os.path.join(root, 'backgrounds')

    # setting up folders
    for folder in [objects_dir, backgrounds_dir]:
        check_dir(folder)
    for cat in VOC_CATS[1:]:
        check_dir(os.path.join(objects_dir, cat))

    # copying (linking) the dataset images into background folders
    print('Copying backgrounds')
    copy_images(loader_voc, backgrounds_dir)

    extract_instances = True
    if extract_instances:
        print('Extracting object instances')
        all_instance_names = [s.split('/')[-1][:-4] for s in
                              glob(os.path.join(instance_root, '*.png'))]
        instance_names = list(set(loader_voc.filenames).intersection(set(all_instance_names)))
        bar = progressbar.ProgressBar()
        for name in bar(instance_names):
            masks, cats = extract_instances_from_image_voc(name, loader_voc,
                                                           instance_root)
            for j in range(len(cats)):
                cat_name = loader_voc.ids_to_cats[cats[j]]
                path = os.path.join(objects_dir, cat_name, name + '_%d' % j)

                mask = Image.fromarray(np.uint8(masks[j] * 255))
                mask.save(path + '.pbm', format='PNG')
                softlink(loader_voc.get_image_path(name), path + '.jpg')
    print('Done')
