import cv2
import os
import numpy as np
from glob import glob
from PIL import Image, ImageEnhance
from configs.paths import DATASETS_DIR, CONTEXT_MAPPING_DIR
from utils.utils_bbox import batch_iou, xy2wh, wh2xy, wh2center, center2wh


class StaticInstanceManipulator(object):
    """Manipulates images in the scope of one image.

    Can manipulate and paste instances only on an image they were extracted
    from. Does not require building an instance database."""

    def __init__(self, loader, aug_config):
        self.loader = loader
        self.config = aug_config
        self.blending_list = ['gaussian', 'none', 'box']

        self.name2instances = dict()
        for year in loader.years:
            if year == '12':
                dataroot = os.path.join(DATASETS_DIR, 'VOC12_instances',
                                        'cut_objects/{}')
                self.instance_loader = loader
            else:
                assert False, 'Other datasets have no instance masks'

            self.name2instances.update(
                self.map_filenames_to_instances(dataroot))

    def map_filenames_to_instances(self, dataroot):
        "For each image list all instances coming from it"

        instances = []
        for cat in [self.loader.ids_to_cats[c] for c
                    in self.loader.cats_include[1:]]:
            instance_root = dataroot.format(cat)
            instances += glob(os.path.join(instance_root, '*.pbm'))

        # defining good names (that are in loader) to filter others out
        good_names = set(self.loader.filenames)
        name2instances = dict()
        for inst in instances:
            name = '_'.join(os.path.basename(inst).split('_')[:-1])
            if name not in good_names:
                continue
            try:
                name2instances[name].append(inst)
            except KeyError:
                name2instances[name] = [inst]
        return name2instances

    def get_annotation_from_mask_file(self, mask_file, scale=1.0):
        """Extracts bounding box from an instance mask"""

        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file)
            mask = 255 - mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if len(np.where(rows)[0]) > 0:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                return (int(scale*xmin), int(scale*xmax),
                        int(scale*ymin), int(scale*ymax))
            else:
                return -1, -1, -1, -1
        else:
            print("%s not found. Using empty mask instead." % mask_file)
            return -1, -1, -1, -1

    def extract_mask(self, mask_file, foreground):
        """Extracts binary mask and an image given instance's path"""

        xmin, xmax, ymin, ymax = self.get_annotation_from_mask_file(mask_file)
        old_coord = [xmin, ymin, xmax-xmin, ymax-ymin]
        foreground = foreground.crop((xmin, ymin, xmax, ymax))
        o_w, o_h = foreground.size
        mask = Image.open(mask_file)
        mask = mask.crop((xmin, ymin, xmax, ymax))
        try:
            mask = Image.fromarray(255 - np.array(mask))
        except Exception as e:
            # print('ERROR: ', e)
            return None, None, None
        return foreground, mask, old_coord

    def instance2cat(self, instance):
        "Given an instance's path outputs its class"

        cat = instance.split('/')[-2]
        id = self.loader.cats_to_ids[cat]
        final_id = self.loader.cats_include_to_cats[id]
        return final_id

    def rescale_instance(self, foreground, mask, scale, old_coord):
        """Re-scales an instance with the mask"""

        w, h = np.array(foreground.size)
        o_w, o_h = np.array(np.clip((np.array(foreground.size) * scale),
                                    1, 9999), dtype=int)
        foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
        mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
        x_min_c = -int((o_w - w) / 2)
        x_max_c = (o_w - w) + x_min_c
        y_min_c = -int((o_h - h) / 2)
        y_max_c = o_h - h + y_min_c
        new_coord = []
        for old, corr in zip(old_coord, [x_min_c, y_min_c, x_max_c, y_max_c]):
            new_coord.append(old + corr)
        return foreground, mask, new_coord

    def check_bbox(self, bbox, width, height):
        """Checks that the meets fit's image and object constraints"""

        bbox_xy = wh2xy(bbox)
        init_area = bbox[2] * bbox[3]
        xmin = np.clip(bbox_xy[0], a_min=0, a_max=width)
        ymin = np.clip(bbox_xy[1], a_min=0, a_max=height)
        xmax = np.clip(bbox_xy[2], a_min=0, a_max=width)
        ymax = np.clip(bbox_xy[3], a_min=0, a_max=height)
        new_bbox = xy2wh([xmin, ymin, xmax, ymax])
        new_area = new_bbox[2] * new_bbox[3]
        return new_bbox, new_area / init_area > 0.35

    def delete_covered_boxes(self, new_bbox, bboxes, cats, diff, given_cat):
        """Handles occlusions"""

        ious, intersections = batch_iou(bboxes, np.reshape(new_bbox, (1, 4)),
                                        return_intersection=True)
        assert bboxes.shape[0] == len(cats) == len(diff), \
            'Have a problem with number of examples'
        areas = bboxes[:, 2] * bboxes[:, 3]
        inter_fraq = (intersections.ravel() / areas)
        not_occluded = inter_fraq < 0.7
        bboxes = bboxes[not_occluded]
        cats = np.array(cats)[not_occluded].tolist()
        diff = np.array(diff)[not_occluded].tolist()

        bboxes = np.concatenate([bboxes, np.reshape(new_bbox, (1, 4))], axis=0)
        cats.append(given_cat)
        diff.append(0)
        return bboxes, cats, diff

    def color_augment(self, foreground):
        """Performs random color augmentation of an instance"""

        if np.random.rand() > 0.5:
            contrast = ImageEnhance.Contrast(foreground)
            foreground = contrast.enhance(np.random.uniform(0.5, 1.5))
        if np.random.rand() > 0.5:
            color = ImageEnhance.Color(foreground)
            foreground = color.enhance(np.random.uniform(0.5, 1.5))
        if np.random.rand() > 0.5:
            brightness = ImageEnhance.Brightness(foreground)
            foreground = brightness.enhance(np.random.uniform(0.5, 1.5))
        return foreground

    def blend(self, corner, mask, foreground, backgrounds, blending_list):
        """Blends an object inside an image.

        Uses provided blending strategy.
        """
        x, y = corner
        for i in range(len(blending_list)):
            blending = blending_list[i]
            if blending == 'none':
                backgrounds[i].paste(foreground, (x, y), mask)
            elif blending == 'gaussian':
                backgrounds[i].paste(
                    foreground, (x, y),
                    Image.fromarray(cv2.GaussianBlur(np.array(mask),
                                                     (5, 5), 2)))
            elif blending == 'box':
                backgrounds[i].paste(foreground, (x, y),
                                     Image.fromarray(cv2.blur(np.array(mask),
                                                              (3, 3))))
        return backgrounds


class DynamicInstanceManipulator(StaticInstanceManipulator):
    def __init__(self, loader, aug_config=None):
        """Manipulates instances across images.

        Builds instance database to operate all objects on-the-fly
        """

        super(DynamicInstanceManipulator, self).__init__(
            loader=loader, aug_config=aug_config)
        # collecting masks database
        self.fgs, self.masks, self.boxes, self.ars, self.scales, \
            self.names, self.inst_paths, self.cats = ([], [], [], [],
                                                      [], [], [], [])
        for name in list(self.name2instances.keys()):
            fg = Image.open(self.instance_loader.get_image_path(name))
            instances = self.name2instances.get(name)
            for inst in instances:
                fg_put, mask, box = self.extract_mask(inst, fg.copy())
                if fg_put is None:
                    continue
                self.ars.append(box[2] / box[3])
                self.scales.append(box[2] * box[3])
                self.fgs.append(fg_put)
                self.masks.append(mask)
                self.boxes.append(box)
                self.names.append(name)
                self.inst_paths.append(inst)
                self.cats.append(self.instance2cat(inst))

    def find_instance(self, name, bbox):
        """Finds an appropriate instance to put on an image.

        Given an image's name and a bbox of existing object on it,
        finds an object of identical."""
        bboxes, _, cats, wi, hi, diff = self.loader.read_annotations(name)
        ar = bbox[2] / bbox[3]
        scale = bbox[2] * bbox[3]

        good_scale_inds = self.scales > 0.5 * scale
        good_ar_inds = (self.ars > ar / 3) * (self.ars < 3 * ar)
        min_scale, max_scale = scale / 2, scale * 2

        good_inds = good_scale_inds * good_ar_inds
        if not np.any(good_inds):
            if not np.any(good_ar_inds):
                good_inds = good_scale_inds
            else:
                good_inds = good_ar_inds

        ind = np.random.choice(np.arange(len(good_inds))[good_inds])
        w, h = self.boxes[ind][2:]
        scale_mult = np.random.uniform(min_scale / self.scales[ind],
                                       max_scale / self.scales[ind])
        return self.masks[ind], self.fgs[ind], w, h, scale_mult


class ContextInstanceManipulator(DynamicInstanceManipulator):
    """Uses context guidance for instance placement"""

    def __init__(self, loader, aug_config):
        super(ContextInstanceManipulator, self).__init__(
            loader=loader, aug_config=aug_config)

        self.print_status('before load')
        self.name2sample = self.load_names2samples()
        self.print_status('after load')

        conf_thresh = aug_config['context_conf']
        if conf_thresh > 0.7:
            # filters out all proposal with confidence lower than 0.7
            print('FILTERING FOR THRESHOLD %.2f' % conf_thresh)
            self.name2sample = self.threshold_proposals(conf_thresh)
            self.print_status('after conf thresh')

        if self.config.get('constrain_instances', False):
            print('CONSTRAINING INSTANCES')
            self.name2sample = self.constrain_instances()
            self.print_status('after constraining')

        self.cat_filter_final_dict(self.name2sample)

    def print_status(self, state):
        try:
            context_n = len(self.name2sample)
        except AttributeError:
            context_n = 'None'
        print(('{}| instance sources: {}, pos_filenames:'
               '{}, context positive images: {}').format(
                   state.upper(), len(self.name2instances),
                   len(self.loader.get_filenames('pos')),
                   context_n))

    def constrain_instances(self):
        """ When some positive images are missing for some reason, deletes
            from context mapping candidates instances that came from these
            images.
        """
        new_dict = dict()
        available_instances = np.unique(self.inst_paths).tolist()
        print(len(self.name2instances),
              len(self.loader.get_filenames('pos')),
              len(self.name2sample))

        for name in self.name2sample:
            sample = self.name2sample[name]
            good_inds = []
            for ind in range(len(sample['scores'])):
                good_insts_paths = [i for i in sample['inst_paths'][ind]
                                    if i in available_instances]
                if len(good_insts_paths) == 0:
                    continue
                good_inds.append(ind)
                sample['inst_paths'][ind] = good_insts_paths
                sample['scales'][ind] = {path: sample['scales'][ind][path]
                                         for path in good_insts_paths}
            if len(good_inds) > 0:
                for key in sample:
                    sample[key] = [sample[key][i] for i in good_inds]
                new_dict[name] = sample

        print(len(self.name2instances),
              len(self.loader.get_filenames('pos')),
              len(new_dict))
        return new_dict

    def threshold_proposals(self, thresh):
        """Filters context location candidates.

        Removes the proposals that were scored by the context model lower than
        a threshold
        """

        new_dict = dict()
        for im_name in self.name2sample.keys():
            sample = self.name2sample[im_name]
            new_sample = {key: [] for key in sample}
            to_add = False
            for i in range(len(sample['scores'])):
                if sample['scores'][i] > thresh:
                    for key in new_sample:
                        new_sample[key].append(sample[key][i])
                        to_add = True
            if to_add:
                new_dict[im_name] = new_sample
        return new_dict

    def load_names2samples(self):
        """ Loads name2sample dict into memory.

        Also applies filtering and checks
        """

        def load(name, suff=''):
            full_proposals_dict = {}
            n_datasets = len(self.loader.years)
            for i in range(n_datasets):
                year, split = self.loader.years[i], self.loader.splits[i]
                datasetname = self.loader.dataset + year + split

                # because self.segmentation is equivalent to args.small_data
                datasetname = datasetname + '_small' * self.loader.segmentation
                path = os.path.join(CONTEXT_MAPPING_DIR, datasetname,
                                    '%s%s.npy' % (name, suff))
                di = np.load(path)[()]
                print('interm_context_load: len {}, loaded from {}'
                      .format(len(di), path))
                full_proposals_dict.update(di)
            print('The context file of total len {} is loaded from {} dataset(s)'
                  .format(len(full_proposals_dict), n_datasets))
            return full_proposals_dict

        try:
            final_dict = load(self.config['context_name'])
            final_dict = {key: value for key, value in final_dict.items()
                          if key in self.loader.filenames}
            print('Full dict was loaded from file %s' % self.config['context_name'])
        except FileNotFoundError:
            assert False, 'Full dict was\'t loaded from file %s' % self.config['context_name']

        return final_dict

    def get_scaled_instance(self, anchor, inst, scale_range,
                            width, height, jitter=True):
        """Re-scales an instance before placing in an image.

        The scaling factor is chosen from the ones that ensure tight fitting in
        the context candidate box.
        """

        if jitter:
            for i in range(40):
                anchor_ = np.array(anchor, dtype=np.float32)
                random_shift_mult = np.random.uniform(-0.2, 0.2, 2)
                anchor_[:2] += random_shift_mult * anchor_[2:]
                anchor_xy = np.array(wh2xy(anchor_), dtype=int)
                if (np.all(anchor_xy[:2] > [0, 0])
                        and np.all(anchor_xy[2:] < [width, height])):
                    anchor = anchor_
                    break

        cx, cy = wh2center(anchor)[:2]
        wi, hi = np.array(self.boxes[inst])[2:]
        inst_bbox = center2wh(np.array([cx, cy, wi, hi]))
        mask = self.masks[inst]
        foreground = self.fgs[inst]
        scale = np.random.uniform(scale_range[0], scale_range[1] + 0.03)
        foreground, mask, inst_bbox = self.rescale_instance(
            foreground, mask, scale, wh2xy(inst_bbox))
        return mask, foreground, xy2wh(inst_bbox)

    def cat_filter_final_dict(self, final_dict):
        """Filters out context matches for classes not supported by the loader"""

        if not self.loader.cats_exclude:
            # all cats are supported, nothing to filter out
            return final_dict

        cat_incl_ids = list(map(int, self.loader.cats_include))
        cat_incl_names = [self.loader.ids_to_cats[id] for id in cat_incl_ids]

        # going over images
        good_keys = []
        for key in final_dict:
            img_dict = final_dict[key]
            is_good = []
            # go over anochor boxes on an image
            for i in range(len(img_dict['cats'])):
                c = img_dict['cats'][i]
                is_good.append(c in cat_incl_ids)
                if not is_good[-1]:
                    continue
                else:
                    # mapping the cat to internal cat
                    img_dict['cats'][i] = self.loader.cats_include_to_cats[c]

                # go over instances that matched the anchor and check their cat
                inst_paths_matched = img_dict['inst_paths'][i]
                new_inst_paths_matched = []
                for inst_path in inst_paths_matched:
                    if any([cat == inst_path.split('/')[-2]
                            for cat in cat_incl_names]):
                        new_inst_paths_matched.append(inst_path)
                if new_inst_paths_matched:
                    img_dict['inst_paths'][i] = new_inst_paths_matched
                else:
                    is_good[-1] = False

            # for each key ('scores', 'bboxes', ...) in an img_dict throwing
            # away entries that are not good
            for l_key in img_dict:
                final_dict[key][l_key] = [v for j, v in enumerate(img_dict[l_key])
                                          if is_good[j]]

            # if no good anchors is left for an image, delete the mapping
            if len(final_dict[key]['scores']) > 0:
                good_keys.append(key)

        print('LEN GOOD KEYS', len(good_keys))
        self.name2sample = {gkey: final_dict[gkey] for gkey in good_keys}
        print('After cats filtering we have')
        self.print_status('after load')
        return self.name2sample
