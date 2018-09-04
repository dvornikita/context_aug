import numpy as np
from PIL import Image
from utils.utils_bbox import batch_iou, xy2wh, wh2xy
from augmentation.instance_manipulators import (StaticInstanceManipulator,
                                                DynamicInstanceManipulator,
                                                ContextInstanceManipulator)


class InstanceEnlarger(StaticInstanceManipulator):
    """Enlarges instances and pastes them back"""

    def __init__(self, loader, aug_config):
        super(InstanceEnlarger, self).__init__(loader, aug_config)
        self.min_scale = aug_config['enlarge_min_scale']
        self.max_scale = aug_config['enlarge_max_scale']

    def enlarge_instance(self, image, mask_file, fg_image, blending_list):
        """Enlarges an instance with its mask"""
        foreground = fg_image.copy()
        if type(image) != list:
            backgrounds = [image.copy() for i in range(len(blending_list))]
        else:
            backgrounds = [i.copy() for i in image]
        width, height = backgrounds[0].size
        scale = np.random.uniform(self.min_scale, self.max_scale)
        foreground, mask, old_coord = self.extract_mask(mask_file, foreground)
        if foreground is None:
            return None, None, None
        foreground, mask, new_coord = self.rescale_instance(
            foreground, mask, scale, wh2xy(old_coord))
        xmin, ymin, xmax, ymax = new_coord
        if self.config['colorize']:
            foreground = self.color_augment(foreground)

        backgrounds = self.blend(np.array([xmin, ymin]), mask, foreground,
                                 backgrounds, blending_list)
        new_bbox, _ = self.check_bbox([xmin, ymin, xmax-xmin, ymax-ymin],
                                      width, height)
        return backgrounds, new_bbox, old_coord

    def manipulate_image(self, name, bboxes, cats, diff):
        """Perform enlargement data augmentation for an image"""

        image = Image.open(self.loader.get_image_path(name))
        init_image = image.copy()
        new_bboxes = np.array(bboxes)

        try:
            instances = self.name2instances[name]
        except KeyError:
            print('No instance segmentation gt for %s' % name)
            return image, bboxes, np.array(cats), np.array(diff)
        N_min = np.min([self.config['min_objects'], len(instances)])
        N_max = np.min([self.config['max_objects'], len(instances)]) + 1
        N = np.random.choice(np.arange(N_min, N_max))
        np.random.shuffle(instances)
        instances = instances[:N]

        random_bboxes = np.array(bboxes)
        np.random.shuffle(random_bboxes)
        random_bboxes = random_bboxes[:N]

        blendings = np.random.choice(self.blending_list, N)
        for k in range(N):
            i = instances[k]
            blending = blendings[k]
            images, new_bbox, old_bbox = self.enlarge_instance(
                image, i, init_image, [blending])
            if images is not None:
                image = images[0]
                ind = np.argmax(batch_iou(new_bboxes, old_bbox), axis=0)
                new_bboxes[ind] = new_bbox
                # new_bboxes, cats, diff = self.delete_covered_boxes(
                #     new_bbox, new_bboxes, cats, diff, self.instance2cat(i))

        return image, new_bboxes, np.array(cats), np.array(diff)


class StaticDuplicator(StaticInstanceManipulator):
    """Copy-pasting instance in one image's scope"""

    def __init__(self, loader, aug_config):
        super(StaticDuplicator, self).__init__(loader, aug_config)
        self.radius = aug_config.get('duplicate_radius', 0.25)
        self.no_spatial_constraints = aug_config.get('no_spatial_constraints')
        print('Duplicate radius is', self.radius)
        print('Spatial constraints', not self.no_spatial_constraints)

    def duplicate_instance(self, image, mask_file, fg_image, blending_list):
        """Performs copy-pasting of an instance"""

        foreground = fg_image.copy()
        width, height = image.size
        if type(image) != list:
            backgrounds = [image.copy() for i in range(len(blending_list))]
        else:
            backgrounds = [i.copy() for i in image]
        scale = np.random.uniform(1/2, 2)
        foreground, mask, old_coord = self.extract_mask(mask_file, foreground)
        if foreground is None:
            return None, None
        x_min, y_min, w, h = old_coord
        foreground, mask, new_coord = self.rescale_instance(
            foreground, mask, scale, wh2xy(old_coord))
        if self.config['colorize']:
            foreground = self.color_augment(foreground)
        xmin, ymin, xmax, ymax = new_coord
        old_bbox = xy2wh(new_coord)
        wi, hi = old_bbox[2:]
        new_wi, new_hi = xmax-xmin, ymax-ymin

        fits = False
        for i in range(50):
            new_xc = np.random.randint(xmin - self.radius*wi,
                                       xmax + self.radius*wi)
            new_yc = np.random.randint(ymin - self.radius*hi,
                                       ymax + self.radius*hi)
            xmin, ymin = new_xc - int(new_wi / 2), new_yc - int(new_hi / 2)
            new_bbox, fits = self.check_bbox([xmin, ymin, new_wi, new_hi],
                                             width, height)
            iou = batch_iou(new_bbox, old_bbox)
            if (iou > 0.2 and iou < 0.7 or self.no_spatial_constraints) and fits:
                backgrounds = self.blend(np.array([xmin, ymin]),
                                         mask, foreground,
                                         backgrounds, blending_list)
                return backgrounds, new_bbox
        return None, None

    def manipulate_image(self, name, bboxes, cats, diff):
        "Implements the full copy-pasting pipeline for an image"

        image = Image.open(self.loader.get_image_path(name))
        init_image = image.copy()
        new_bboxes = np.array(bboxes)

        try:
            instances = self.name2instances[name]
        except KeyError:
            print('No instance segmentation gt for %s' % name)
            return image, bboxes, np.array(cats), np.array(diff)
        N_min = np.min([self.config['min_objects'], len(instances)])
        N_max = np.min([self.config['max_objects'], len(instances)]) + 1
        N = np.random.choice(np.arange(N_min, N_max))
        np.random.shuffle(instances)
        instances = instances[:N]

        random_bboxes = np.array(bboxes)
        np.random.shuffle(random_bboxes)
        random_bboxes = random_bboxes[:N]

        blendings = np.random.choice(self.blending_list, N)
        for k in range(N):
            i = instances[k]
            blending = blendings[k]
            images, new_bbox = self.duplicate_instance(
                image, i, init_image, [blending])

            if images is not None:
                # deleting annotations of bboxes occluded by more than 70%
                new_bboxes, cats, diff = self.delete_covered_boxes(
                    new_bbox, new_bboxes, cats, diff, self.instance2cat(i))
                image = images[0]
        return image, new_bboxes, np.array(cats), np.array(diff)


class DynamicDuplicator(DynamicInstanceManipulator):
    """Places instances from other images in neighborhood of existing objects"""

    def __init__(self, loader, aug_config):
        self.no_spatial_constraints = aug_config.get('no_spatial_constraints')
        super(DynamicDuplicator, self).__init__(loader, aug_config)

    def dynamic_place_instance(self, name, image, bbox, blending_list):
        """Finds fitting instance given a box to place in, and copies there"""

        if type(image) != list:
            backgrounds = [image.copy() for i in range(len(blending_list))]
        else:
            backgrounds = [i.copy() for i in image]
        width, height = image.size
        xmin, ymin, xmax, ymax = wh2xy(bbox)
        mask, foreground, w, h, scale = self.find_instance(name, bbox)
        inst_bbox = [0, 0, w, h]
        if self.config['colorize']:
            foreground = self.color_augment(foreground)
        foreground, mask, inst_bbox = self.rescale_instance(
            foreground, mask, scale, wh2xy(inst_bbox))
        w, h = xy2wh(inst_bbox)[2:]

        fits = False
        for i in range(50):
            new_xc = np.random.randint(xmin - 0.25*w, xmax + 0.25*w)
            new_yc = np.random.randint(ymin - 0.25*h, ymax + 0.25*h)
            # new_xc = np.random.randint(xmin - 0.75*w, xmax + 0.75*w)
            # new_yc = np.random.randint(ymin - 0.75*h, ymax + 0.75*h)
            xmin, ymin = new_xc - int(w / 2), new_yc - int(h / 2)
            new_bbox, fits = self.check_bbox([xmin, ymin, w, h], width, height)
            iou = batch_iou(new_bbox, bbox)
            if (iou > 0.2 and iou < 0.7 or self.no_spatial_constraints) and fits:
                backgrounds = self.blend(np.array([xmin, ymin]),
                                         mask, foreground,
                                         backgrounds, blending_list)
                return backgrounds, new_bbox
        return None, None

    def manipulate_image(self, name, bboxes, cats, diff):
        """Implements full dynamic placement pipeline"""

        image = Image.open(self.loader.get_image_path(name))
        init_image = image.copy()
        new_bboxes = np.array(bboxes)

        N_min = np.min([self.config['min_objects'], len(cats)])
        N_max = np.min([self.config['max_objects'], len(cats)]) + 1
        N = np.random.choice(np.arange(N_min, N_max))

        random_bboxes = np.array(bboxes)
        np.random.shuffle(random_bboxes)
        random_bboxes = random_bboxes[:N]

        blendings = np.random.choice(self.blending_list, N)
        for k in range(N):
            blending = blendings[k]
            bbox = random_bboxes[k]
            cat = cats[k]
            images, new_bbox = self.dynamic_place_instance(
                name, init_image, bbox, [blending])

            if images is not None:
                # deleting annotations of bboxes occluded by more than 70%
                new_bboxes, cats, diff = self.delete_covered_boxes(
                    new_bbox, new_bboxes, cats, diff, cat)
                image = images[0]
        return image, new_bboxes, np.array(cats), np.array(diff)


class ContextPlacer(ContextInstanceManipulator):
    """Places with respect to context model guidance"""

    def __init__(self, loader, aug_config):
        super(ContextPlacer, self).__init__(loader, aug_config)

    def place_context_instance(self, sample, image, ind,
                               inst_path, blending_list):
        """Places a selected instance at proposed context location"""

        if type(image) != list:
            backgrounds = [image.copy() for i in range(len(blending_list))]
        else:
            backgrounds = [i.copy() for i in image]
        width, height = image.size

        anchor = sample['bboxes'][ind]
        cat = int(sample['cats'][ind])
        inst = np.arange(len(self.inst_paths))[np.array(self.inst_paths) == inst_path][0]
        scale_range = sample['scales'][ind][inst_path]

        mask, foreground, new_bbox = self.get_scaled_instance(
            anchor, inst, scale_range, width, height)

        if self.config['colorize']:
            foreground = self.color_augment(foreground)

        backgrounds = self.blend(np.array(new_bbox[:2], dtype=int), mask,
                                 foreground, backgrounds, blending_list)

        clipped_bbox, _ = self.check_bbox(new_bbox, width, height)
        return backgrounds, clipped_bbox, cat

    def manipulate_image(self, name, bboxes, cats, diff):
        """Implements the pipeline of pacing instances in right context"""
        image = Image.open(self.loader.get_image_path(name))
        init_image = image.copy()
        new_bboxes = np.array(bboxes)

        sample = self.name2sample[name]
        scores = np.array(sample['scores']).ravel()

        N_min = np.min([self.config['min_objects'], len(scores)])
        N_max = np.min([self.config['max_objects'], len(scores)]) + 1
        N = np.random.choice(np.arange(N_min, N_max))
        inds = np.random.choice(np.arange(len(scores)), size=N,
                                p=scores/scores.sum(), replace=False)

        blendings = np.random.choice(self.blending_list, N)
        for k in range(len(inds)):
            blending = blendings[k]
            ind = inds[k]
            available_instances = sample['inst_paths'][ind]
            instance_ind = np.random.randint(0, len(available_instances))
            inst_path = available_instances[instance_ind]
            print(inst_path)
            images, new_bbox, cat = self.place_context_instance(
                sample.copy(), image, ind, inst_path, [blending])

            if images is not None:
                # deleting annotations of bboxes occluded by more than 70%
                new_bboxes, cats, diff = self.delete_covered_boxes(
                    new_bbox, new_bboxes, cats, diff, cat)
                image = images[0]
        return image, new_bboxes, np.array(cats), np.array(diff)
