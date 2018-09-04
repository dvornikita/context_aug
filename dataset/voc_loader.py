import os

import logging
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from configs.paths import DATASETS_DIR
from utils.utils_general import make_list, read_textfile
from utils.utils_bbox import draw_bbox

log = logging.getLogger()

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


class VOCLoader():
    """Data manager"""

    def __init__(self, year, split, segmentation=False, augmented_seg=False,
                 cats_exclude=[], cats_include=[], subsets=None, is_training=False,
                 gt_seg=False, cut_bad_names=True, names_file=''):
        self.dataset = 'voc'
        self.is_training = is_training
        self.gt_seg = gt_seg
        self.segmentation = segmentation
        self.augmented_seg = augmented_seg

        self.set_up_internal_cats(cats_include, cats_exclude)

        # creating cats' names
        cats = VOC_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(self.cats_include)
        self.categories = cats[1:]

        # Processing possibly many datasets
        year, split = map(make_list, [year, split])
        if not subsets:
            subsets = ['all'] * len(year)
        else:
            for s in subsets:
                assert s in ['all', 'pos', 'neg'],\
                    'Wrong subset must be in [all, pos, neg]'
        self.splits = split
        self.years = year
        self.roots = [os.path.join(DATASETS_DIR, 'VOCdevkit/VOC20%s/' % year)
                      for year in self.years]
        for s in self.splits:
            assert s in ['train', 'val', 'trainval', 'test', 'custom']
        assert len(year) == len(split) == len(subsets),\
            'Different number of components'

        if names_file:
            self.create_filenames_from_file(names_file)
        else:
            self.create_filenames(subsets)

        if cut_bad_names:
            self.cut_bad_names_out()

        # deleting from the filenames images that
        # don't contain cats_include instances
        if cats_exclude != [] and is_training is True:
            self.filter_filenames_by_cat()

    def set_up_internal_cats(self, cats_include, cats_exclude):
        """Defines a set of classes supported by this loader"""

        assert len(cats_include) * len(cats_exclude) == 0, """Only one of cats_include or
        cats_exclude could be set, not together"""
        # splitting cats used and not supported by the loader
        if len(cats_include) > 0:
            self.cats_include = sorted(list(set([0] + cats_include)))
            self.cats_exclude = sorted(list(set(list(range(0, 21))) -
                                            set(self.cats_include)))
        else:
            self.cats_exclude = cats_exclude
            self.cats_include = sorted(list(set(list(range(0, 21))) -
                                            set(cats_exclude)))

        # Defining mappings from internal cats to general and back
        #TODO swap the names of these two dicts
        self.cats_to_cats_include = dict(enumerate(self.cats_include))
        self.cats_include_to_cats = dict(map(reversed,
                                             enumerate(self.cats_include)))

    def cut_bad_names_out(self):
        """Throws away samples with inaccurate annotations"""

        bad_names = ['2007_002403', '2011_000834',
                     '2010_000748', '2009_005069']
        self.filenames = [f for f in self.filenames if f not in bad_names]

    def create_filenames(self, subsets):
        """Loads a file with filenames in memory

        Filters out unwanted names. In case of several datasets merges their
        names together while keeping track of each root"""

        self.filenames = []
        self.name2root = dict()
        for year, root, split, subset in zip(self.years, self.roots,
                                             self.splits, subsets):
            if self.segmentation and year == '12':
                filelist = 'ImageSets/Segmentation/%s.txt'
            else:
                filelist = 'ImageSets/Main/%s.txt'

            filenames = read_textfile(os.path.join(root, filelist % split))
            name2root = dict(zip(filenames, [root] * len(filenames)))
            self.name2root.update(name2root)
            info_message = 'Created a loader VOC%s %s with %i images' \
                           % (year, split, len(filenames))

            # filtering our names that don't belong to the subset
            if subset != 'all':
                pos_names, neg_names = self.split_filenames(filenames)
                filenames = pos_names if subset == 'pos' else neg_names
                info_message = ('%s, after selecting %s subset %i '
                    'images is left') % (info_message, subset, len(filenames))
            log.info(info_message)
            self.filenames += filenames
        self.name2root = {f: self.name2root[f] for f in self.filenames}

    def create_filenames_from_file(self, names_file):
        """ Initializes loader image names support from a file

        Creates self.filenames and self.name2root
        """
        self.filenames = []
        self.name2root = dict()
        self.filenames = read_textfile(names_file)
        self.name2root = dict(zip(self.filenames,
                                  [self.roots[0]] * len(self.filenames)))
        info_message = 'Created a VOC loader with %i images from file %s' \
                        % (len(self.filenames), names_file)
        log.info(info_message)

    def filter_filenames_by_cat(self):
        """Filters out filenames that don't contain cats in self.cats_include
        """

        new_filenames = []
        for name in self.filenames:
            cats = self.read_annotations(name, map_cats=False)[2]
            cats_presence = list(set(cats))
            for cat_in in cats_presence:
                if cat_in in self.cats_include:
                    new_filenames.append(name)
                    break
        message = ('Filtered by cat: After deleting {} cats, '
                   ' {} images remained out of {}').format(
                       [c for c in self.cats_exclude],
                       len(new_filenames),
                       len(self.filenames))
        print(message)
        log.info(message)
        self.filenames = new_filenames

    def get_image_path(self, name):
        root = self.name2root[name]
        path = '%sJPEGImages/%s.jpg' % (root, name)
        return path

    def get_seg_path(self, name):
        seg_file = os.path.join(self.name2root[name], 'SegmentationClass/',
                                name + '.png')
        return seg_file

    def split_filenames(self, filenames):
        """Splits all filenames in positive and negative"""

        pos_names, neg_names = [], []
        for f in filenames:
            cats = self.read_annotations(f)[2]
            pos_names.append(f) if len(cats) > 0 else neg_names.append(f)
        return pos_names, neg_names

    def get_filenames(self, _type='all'):
        """Returns requested set of filenames

        all - all, pos - positive, neg - negative
        """

        if _type == 'all':
            return self.filenames

        assert _type in ['pos', 'neg'], 'Wrong filenames type: %s' % _type
        try:
            self.pos_names
        except AttributeError:
            self.pos_names, self.neg_names \
                = self.split_filenames(self.filenames)

        if _type == 'pos':
            return self.pos_names
        if _type == 'neg':
            return self.neg_names

    def load_image(self, name, given_path=None):
        """Loads an image given it's internal name

        The format is suitable for feeding to the network
        """

        path = given_path if given_path else self.get_image_path(name)
        im = Image.open(path)
        im = np.array(im) / 255.0
        im = im.astype(np.float32)
        return im

    def read_annotations(self, name, map_cats=True):
        """Loads sample annotations from the disk.

        Args:
            name (str): internal sample name
            map_cats (bool): if the values of categories are mapped to the
            internal ones
        Returns:
            A tuple containing Object bounding boxes, Image segmentation mask,
            Object categories, Image width, Image height, Object difficulty.
        """

        bboxes = []
        cats = []

        tree = ET.parse('%sAnnotations/%s.xml' % (self.name2root[name], name))
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        difficulty = []
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            difficult = (int(obj.find('difficult').text) != 0)
            difficulty.append(difficult)
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text)-x
            h = int(bbox_tag.find('ymax').text)-y
            bboxes.append((x, y, w, h))

        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes).reshape((len(bboxes), 4))
        difficulty = np.array(difficulty)

        if map_cats:
            # throwing away boxes with exclude_cats and mapping their values
            # from global to internal
            inds = np.ones(gt_cats.shape, dtype=bool)
            if self.cats_exclude != []:
                for i, cat in enumerate(gt_cats):
                    inds[i] = cat not in self.cats_exclude
            assert np.any(inds) or not self.is_training, \
                'No positives in an image left'

            gt_cats = gt_cats[inds]
            gt_bboxes = gt_bboxes[inds]
            difficulty = difficulty[inds]

            # mapping old categories to new cats (cats_include)
            for i, gt_cat in enumerate(gt_cats):
                gt_cats[i] = self.cats_include_to_cats[gt_cat]

        seg_gt = self.read_segmentations(name, height, width)

        output = gt_bboxes, seg_gt, gt_cats, width, height, difficulty
        return output

    def read_segmentations(self, name, height=None, width=None):
        """Loads segmentation annotation from the disk"""

        seg_path = self.get_seg_path(name)
        if self.segmentation and os.path.exists(seg_path):
            seg_map = Image.open(seg_path)
            segmentation = np.array(seg_map, dtype=np.uint8)
        else:
            # if there is no segmentation for a particular image
            # we fill the mask with zeros to keep the same amount
            # of tensors but don't learn from it
            assert height is not None, ('In this case h and w '
                                        'have to be passed: %s') % name
            segmentation = np.zeros([height, width], dtype=np.uint8)

            # in case the file doesn't exist we don't know what could be inside
            # so it's better to stay not certain
            if (not os.path.exists(seg_path) and self.gt_seg):
                segmentation += 255
        return segmentation

    def get_sample(self, name):
        """Outputs a training sample with different types of annotations"""

        gt_bboxes, seg_gt, gt_cats, w, h, difficulty \
            = self.read_annotations(name)
        gt_bboxes = np.clip(gt_bboxes / np.reshape([w, h, w, h], (1, 4)), 0, 1)
        diff = np.array(difficulty, dtype=np.int32)
        image = self.load_image(name)

        assert (h, w) == image.shape[:2]
        assert len(gt_cats) == 0 or max(gt_cats) <= len(self.categories),\
            'gt_cats: {}, len(cats): {}'.format(max(gt_cats),
                                                len(self.categories))

        out = {'img': image,
               'gt_bboxes': gt_bboxes,
               'gt_cats': gt_cats,
               'seg_gt': seg_gt[:, :, None],
               'diff': diff}
        return out

    def visualize(self, name, draw=True, seg=False):
        """Makes visualization of a training sample with its annotations"""

        sample = self.get_sample(name)
        im = np.uint8(sample['img'] * 255)
        h, w = im.shape[:2]
        bboxes = sample['gt_bboxes'] * np.reshape([w, h, w, h], (1, 4))
        cats = sample['gt_cats']

        img = (draw_bbox(im, bboxes=bboxes, cats=cats) if draw
               else Image.fromarray(np.uint8(im)))

        if seg:
            from my_utils import array2palette
            seg = array2palette(np.squeeze(sample['seg_gt'])).convert('RGB')
            new_img = Image.new('RGB', (2*w, h))
            new_img.paste(img, (0, 0))
            new_img.paste(seg, (w, 0))
            img = new_img
        return img
