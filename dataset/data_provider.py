import tensorflow as tf
import matplotlib.pyplot as plt

from configs.config import args
from utils.utils_tf import data_augmentation
from dataset.voc_loader import VOCLoader
from dataset.instance_sampler import InstanceSampler

Iterator = tf.data.Iterator
slim = tf.contrib.slim


class DataHandler():
    """Serves data from hard drive to the model

    Uses a loader to get samples, feeds them to tf data pipeline,
    implements asynchronous loading and augmentation with feedable iterators
    """

    def __init__(self, sess, encoding_type='simple'):
        self.sess = sess
        self.handle_ph = tf.placeholder(dtype=tf.string, shape=())
        self.keys = ['img', 'bbox', 'label', 'w', 'h', 'frame']
        self.shapes = [[None, None, 3], [4], [], [], [], [4]]
        self.types = [tf.float32] * 2 + [tf.int32] + [tf.float32] * 3
        self.encode_dataset = self.encode_dataset_generator

    def get_samplers(self, dataset, split):
        """For a dataset, gives a pair of context image samplers (train/val)"""

        # categories that don't participate in the training
        excluded = ([int(c) for c in args.excluded.split('_')]
                    if args.excluded != "" else [])

        if dataset == 'voc12':
            train_loader = VOCLoader('12', split, args.small_data,
                                     cats_exclude=excluded)
            val_loader = VOCLoader('12', 'val', cats_exclude=excluded)
        if dataset == 'voc07':
            train_loader = VOCLoader('07', split, cats_exclude=excluded)
            val_loader = VOCLoader('07', 'test', cats_exclude=excluded)
        if '-' in dataset:
            # To create one loader encapsulating several dataset
            years = [y[-2:] for y in args.dataset.split('-')]
            splits = [s for s in args.split.split('-')]
            train_loader = VOCLoader(years, splits, args.segment,
                                     cats_exclude=excluded,
                                     cut_dataset=args.cut_dataset,
                                     random_seed=args.random_seed,
                                     gt_seg=args.segment)
            val_loader = VOCLoader('07', 'test', args.segment,
                                   cats_exclude=excluded)

        train_sampler = InstanceSampler(loader=train_loader,
                                        random_box=args.random_box,
                                        neg_bias=args.neg_bias)
        val_sampler = InstanceSampler(loader=val_loader,
                                      random_box=args.random_box,
                                      neg_bias=args.neg_bias)
        self.num_classes = train_loader.num_classes
        return train_sampler, val_sampler

    def encode_dataset_generator(self, sampler, is_training=True):
        """Creates a dataset in a feedable tf.iterator"""

        def generator():
            while True:
                yield sampler.get_sample()

        type_d = dict(zip(self.keys, self.types))
        shape_d = dict(zip(self.keys, self.shapes))
        dataset = tf.data.Dataset.from_generator(generator, type_d, shape_d)
        # plugging standard data augmentation in the graph
        dataset = dataset.map(lambda s: dict(zip(['img', 'label'], data_augmentation(
            s, is_training))), num_parallel_calls=args.batch_size)
        dataset = dataset.batch(args.batch_size).prefetch(8)

        self.final_shapes = dataset.output_shapes
        self.final_types = dataset.output_types
        iterator = dataset.make_one_shot_iterator()
        handle = self.sess.run(iterator.string_handle())
        return iterator, handle

    def get_feedable_iterator(self, dataset, split):
        """Creates iterators for train and val set, given a dataset name"""

        assert dataset in ['voc07', 'voc12', 'voc0712'], 'Wrong dataset'
        train_sampler, val_sampler = self.get_samplers(dataset, split)
        train_iterator, self.train_handle = self.encode_dataset(train_sampler)
        val_iterator, self.val_handle = self.encode_dataset(val_sampler,
                                                            is_training=False)
        iterator = Iterator.from_string_handle(self.handle_ph,
                                               self.final_types,
                                               self.final_shapes)
        self.sample = iterator.get_next()

        # defining tf feed_dicts here to use them later for switching between
        # train and val sets (in training)
        self.train_fd = {self.handle_ph: self.train_handle}
        self.val_fd = {self.handle_ph: self.val_handle}
        return self.sample


if __name__ == '__main__':
    with tf.Session() as sess:
        datahand = DataHandler(sess, 'generator')
        sample = datahand.get_feedable_iterator(args.dataset, args.split)
        print('!!!!! TRAIN !!!!!!!')
        for i in range(1):
            sam = sess.run(sample, datahand.train_fd)
            for i in range(len(sam['img'])):
                img = sam['img'][i]
                label = sam['label'][i]
                fig = plt.gcf()
                plt.imshow(img)
                if label > 0:
                    fig.canvas.set_window_title('Pos')
                else:
                    fig.canvas.set_window_title('Neg')
                plt.show()

        # print('\n!!!!! VAL !!!!!!!')
        # for i in range(3):
        #     print(sess.run(sample, datahand.val_fd))
