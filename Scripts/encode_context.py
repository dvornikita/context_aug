import os
import sys
import numpy as np
import argparse
import progressbar
from glob import glob

sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))

from configs.paths import EVAL_DIR, CONTEXT_MAPPING_DIR, RAW_CONTEXT_DIR, check_dir
from dataset.voc_loader import VOCLoader
from utils.utils_general import write_file
from utils.utils_bbox import batch_iou, xy2wh, wh2xy, wh2center, center2wh
from augmentation.instance_manipulators import DynamicInstanceManipulator


parser = argparse.ArgumentParser(description='local stuff')
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--dataset", default='voc12', type=str)
parser.add_argument("--split", default='train', type=str)
parser.add_argument("--names_file", default='', type=str)
parser.add_argument("--small_data", default=False, action='store_true')
parser.add_argument("--excluded", default='', type=str)
parser.add_argument("--conf_thresh", default=0.7, type=float)
parser.add_argument("--area_lower_thresh", default=0.007, type=float)
parser.add_argument("--area_upper_thresh", default=0.5, type=float)
parser.add_argument("--n_neighbors", default=1, type=int)
args = parser.parse_args()


def process_raw_context(loader, folder):
    """ Puts in correspondence images and estimated context for them

    Args:
        loader - voc dataset loader
        folder - the folder that contains unprocessed output of the context model
    Returns:
        A dict of format img_name: arr[Nx6], a row is bbox coordinates, object
    category and the corresponding score: [x, y, w, h] + [class] + [class_score]
    """

    full_dict = dict()
    for name in loader.filenames:
        _, _, _, w, h, _ = loader.read_annotations(name)
        cont_dict = np.load(os.path.join(folder, name + '.npy'))[()]
        bboxes = cont_dict['bboxes']
        mask = bboxes[:, 2] * bboxes[:, 3] < w * h * args.area_upper_thresh
        mask = mask * (bboxes[:, 2] * bboxes[:, 3] > w * h * args.area_lower_thresh)
        bboxes = bboxes[mask]
        probs = cont_dict['probs'][mask]

        best_cat = np.argmax(probs[:, 1:], axis=1) + 1
        best_score = probs[np.arange(probs.shape[0]), best_cat]
        mask = best_score > args.conf_thresh
        if mask.sum() == 0:
            continue

        my_scores = best_score[mask].reshape([-1, 1])
        my_cats = best_cat[mask].reshape([-1, 1])
        my_bboxes = bboxes[mask]
        full_dict[name] = np.concatenate([my_bboxes, my_cats, my_scores], axis=1)

    pos_samples = [n for n in list(full_dict.keys()) if n in pos_filenames]
    neg_samples = [n for n in list(full_dict.keys()) if n in neg_filenames]
    print('neg len is {} out of {}'.format(len(neg_samples), len(neg_filenames)))
    print('pos len is {} out of {}'.format(len(pos_samples), len(pos_filenames)))
    return full_dict


def match_anchors_to_instances(anchor_dict, excluded, loader):
    """for each candidate box finds a set of instances that could
    be inserted inside

    Creates a dict structure that for each image retrieves -> a set of
    confident context candidate boxes and for each of them -> all possible
    instance (with corresponding scaling factor) fitting there
    """

    keys = ['bboxes', 'cats', 'scores', 'inst_paths', 'scales']
    man = DynamicInstanceManipulator(loader)    # contains instance dataset
    n_scales = 16
    all_scales = np.linspace(0.5, 2.0, n_scales)
    n_inst = len(man.ars)

    # extracting width and height for all instance boxes in the instance dataset
    all_whs = []
    for i in range(n_inst):
        wh = np.array(man.boxes[i][2:])
        whs = np.array([wh * scale for scale in all_scales])
        all_whs.append(whs)
    all_whs = np.array(all_whs)

    box2instance = dict()
    bar = progressbar.ProgressBar()
    for name in bar(anchor_dict):
        # iterating over the candidates for one image
        for i in range(anchor_dict[name].shape[0]):
            bbox = anchor_dict[name][i, :4]
            cat = anchor_dict[name][i, 4]
            score = anchor_dict[name][i, 5]

            # picking [w, h] of instances belonging to the current category
            cat_inds = np.arange(len(man.cats))[np.array(man.cats) == cat]
            cat_whs = all_whs[cat_inds]
            n_inst = cat_whs.shape[0]

            # trying to fit instances into the candidate box by re-scaling them
            # by n_scales different factors
            init_bboxes = np.tile(bbox.reshape(1, 1, 4), (n_inst, n_scales, 1))
            new_bboxes = np.concatenate([init_bboxes[:, :, :2], cat_whs], axis=-1)
            ious = batch_iou(bbox.reshape((1, 4)), new_bboxes.reshape([-1, 4])).ravel()
            # picking the ones with high IoI
            good_ones = (ious > 0.7).reshape([n_inst, n_scales])
            good_instances = good_ones.sum(axis=1) > 0
            instances = np.arange(good_instances.shape[0])[good_instances]
            inst_paths = [man.inst_paths[cat_inds[i]] for i in instances]
            scales = [all_scales[m] for m in good_ones[instances]]
            scales = {inst_paths[i]: np.array([np.min(scales[i]), np.max(scales[i])])
                      for i in np.arange(len(scales))}
            if np.any(good_ones):
                storage = box2instance.get(name, {k: list() for k in keys})
                vals = {'bboxes': bbox.astype(int), 'cats': cat, 'scores': score,
                        'inst_paths': inst_paths, 'scales': scales}
                for key in keys:
                    storage[key].append(vals[key])
                box2instance[name] = storage
    return box2instance


if __name__ == '__main__':
    excluded = ([int(c) for c in args.excluded.split('_')]
                if args.excluded != "" else [])
    if not args.names_file:
        if '0712' in args.dataset:
            loader = VOCLoader(['07', '12'], ['trainval'] * 2,
                               cats_exclude=excluded,
                               is_training=False)
        elif '12' in args.dataset:
            loader = VOCLoader('12', args.split, args.small_data,
                               cats_exclude=excluded, is_training=False,
                               cut_bad_names=True)
        elif '7' in args.dataset:
            loader = VOCLoader('07', args.split, cats_exclude=excluded,
                               is_training=False)
        dataset = args.dataset
        split = args.split
    else:
        train_names = os.path.join(EVAL_DIR, 'Extra', args.names_file + '.txt')
        loader = VOCLoader(args.dataset[3:], 'custom', names_file=train_names)
        dataset = 'voc07'
        split = 'custom'

    pos_filenames = loader.get_filenames('pos')
    neg_filenames = loader.get_filenames('neg')

    suffix = '_small' * args.small_data + ('-%dneib' % args.n_neighbors)
    context_folder = os.path.join(RAW_CONTEXT_DIR, "-".join(
        [args.run_name, args.dataset + args.split + suffix]))
    files = glob(os.path.join(context_folder, '*.npy'))
    files_striped = [os.path.basename(f).strip(' _.npy') for f in files]

    print('Extracting raw context info from files')
    anchor_dict = process_raw_context(loader, context_folder)
    print('Processing images')
    anchor2instance = match_anchors_to_instances(anchor_dict, excluded, loader)
    write_file(list(anchor2instance.keys()), os.path.join(context_folder, 'filenames.txt'))
    save_folder_name = args.dataset + split + '_small' * args.small_data
    check_dir(os.path.join(CONTEXT_MAPPING_DIR, save_folder_name))
    print('Saving the file with mapping')
    np.save(os.path.join(CONTEXT_MAPPING_DIR, save_folder_name, args.run_name),
            anchor2instance)
    print('DONE')
