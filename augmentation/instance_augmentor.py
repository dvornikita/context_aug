import numpy as np
from PIL import Image
from utils.utils_bbox import draw_bbox
from augmentation.augmentation_methods import (
    InstanceEnlarger, StaticDuplicator, DynamicDuplicator, ContextPlacer)


class Augmentor(object):
    def __init__(self, loader, aug_config):
        self.loader = loader
        self.aug_prob = aug_config['aug_prob']

        self.context_names = []
        self.manipulate = False

        # Defining methods for instance data augmentation
        # and lists of filenames they apply to
        self.method_names = ['enlarger', 'duplicator', 'context_placer']
        methods = dict()
        method_filenames = dict()

        if aug_config['enlarge']:
            methods['enlarger'] = InstanceEnlarger(loader, aug_config)
            method_filenames['enlarger'] = loader.get_filenames('pos')

        if aug_config['duplicate']:
            Duplicator = (DynamicDuplicator if aug_config['dynamic']
                          else StaticDuplicator)
            methods['duplicator'] = Duplicator(loader, aug_config)
            method_filenames['duplicator'] = loader.get_filenames('pos')

        if aug_config['context_name']:
            methods['context_placer'] = ContextPlacer(loader, aug_config)
            context_names = list(methods['context_placer'].name2sample.keys())
            method_filenames['context_placer'] = context_names
            assert not (set(self.context_names) - set(loader.filenames)),\
                ('We have some context names that '
                 'don\'t belong to the training set')
        self.methods = methods
        self.method_filenames = method_filenames

    def get_filenames(self, _type='all'):
        if _type in ['all', 'pos', 'neg']:
            return self.loader.get_filenames(_type)
        else:
            assert _type in ['man', 'context'], \
                'Unknown type of filenames: %s' % _type
            if _type == 'man':
                return (self.loader.get_filenames('pos')
                        + self.method_filenames['context_placer'])
            if _type == 'context':
                return self.method_filenames['context_placer']

    filenames = property(get_filenames)

    def get_sample(self, name):
        gt_bboxes, seg_gt, gt_cats, w, h, diff \
            = self.loader.read_annotations(name)

        # Try to apply each augmentation method
        shuffled_method_names = self.method_names.copy()
        np.random.shuffle(shuffled_method_names)
        for method_name in shuffled_method_names:
            method = self.methods.get(method_name, None)
            if (method and (np.random.rand() < self.aug_prob)
                    and name in self.method_filenames[method_name]):
                im, gt_bboxes, gt_cats, diff = method.manipulate_image(
                    name, gt_bboxes, gt_cats.tolist(), diff.tolist())
                break

        # If no augmentation was applied, load the original image
        try:
            im = np.array(im.convert('RGB')) / 255.0
        except NameError:
            im = self.loader.load_image(name)
        image = im.astype(np.float32)

        gt_bboxes = np.clip(gt_bboxes / np.reshape([w, h, w, h], (1, 4)),
                            0, 0.999)

        assert gt_cats.shape[0] == gt_bboxes.shape[0] == diff.shape[0], \
            'Not equal number of cats, boxes and diffs'
        assert image.ndim == 3, 'image %s doesn\'t have 3 dims' % name
        assert image.shape[-1] == 3, 'image %s doesn\'t have 3 channels' % name

        out = {'img': image,
               'gt_bboxes': gt_bboxes,
               'gt_cats': gt_cats,
               'seg_gt': seg_gt[:, :, None],
               'diff': diff}
        return out

    def visualize(self, name, draw=True):
        sample = self.get_sample(name)
        im = np.uint8(sample['img'] * 255)
        h, w = im.shape[:2]
        bboxes = sample['gt_bboxes'] * np.reshape([w, h, w, h], (1, 4))
        cats = sample['gt_cats']

        img = (draw_bbox(im, bboxes=bboxes, cats=cats) if draw
               else Image.fromarray(np.uint8(im)))
        return img
