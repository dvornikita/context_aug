import os
import argparse
from configs.paths import CKPT_DIR, LOGS

# Mean color to subtract before propagating an image through a DNN
MEAN_COLOR = [103.062623801, 115.902882574, 123.151630838]

parser = argparse.ArgumentParser(description='Train or eval SSD model with goodies.')


parser.add_argument("--run_name", type=str, required=True,
                    help="The name of your experiment")

parser.add_argument("--ckpt", default=0, type=int,
                    help="The number of checkpoint (in thousands) you want to restore from")

parser.add_argument("--dataset", default='voc12', choices=['voc07', 'voc12'],
                    help="The dataset you want to train/test the model on")

parser.add_argument("--split", default='train', choices=['train', 'test', 'val', 'trainval',
                    'valminusminival2014', 'minival2014', 'test-dev2015', 'test2015'],
                    help="The split of the dataset you want to train/test on")

parser.add_argument("--subsets", default=[], nargs='+', type=str,
                    help="""Which subset (subsets[i]) of data is taken from a
                    dataset. subsets[i] in ['all', 'pos', 'neg']""")

parser.add_argument("--image_size", default=300, type=int,
                    help="Which image size to chose for training")

# TRAINING FLAGS
parser.add_argument("--max_iterations", default=1000000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--bn_decay", default=0.9, type=float)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--warmup_step", default=0, type=int,
                    help="For training with warmup, chose the number of steps")
parser.add_argument("--warmup_lr", default=1e-5, type=float,
                    help="For training with warmup, chose the starting learning rate")
parser.add_argument("--optimizer", default='adam', choices=['adam', 'nesterov'],
                    help="Optimizer of choice")
parser.add_argument("--lr_decay", default=[], nargs='+', type=int,
                    help="A list of steps where after each a learning rate is multiplied by 1e-1")
parser.add_argument("--random_trunk_init", default=False, action='store_true',
                    help="Random initialization of a base network")
parser.add_argument("--summarize_gradients", default=False, action='store_true',
                    help="Writing gradients stats in summaries")
parser.add_argument("--excluded", default='', type=str,
                    help="Which cats to exclude from training")


# DATA AUGMENTATION OPTIONS

parser.add_argument("--small_data", default=False, action='store_true',
                    help="If it's on, constrains the training set to VOC12*-seg images")

parser.add_argument("--names_file", default='', type=str,
                    help='This named splits are used instead of standard VOC filenames splits')

parser.add_argument("--random_box", default=False, action='store_true',
                    help='If the candidate boxes are sampled randomly with no regard to distribution')

parser.add_argument("--neg_bias", default=1, type=int, help='How many times more often a negative is sampled during training')

# Standard data augmentation for training the context model
parser.add_argument("--zoomout_prob", default=0.5, type=float)
parser.add_argument("--no_scale_distort", default=False, action='store_true')
parser.add_argument("--brightness_prob", default=0.5, type=float)
parser.add_argument("--contrast_prob", default=0.5, type=float)
parser.add_argument("--hue_prob", default=0.5, type=float)
parser.add_argument("--saturation_prob", default=0.5, type=float)
parser.add_argument("--flip_prob", default=0.5, type=float)


# Inference options
parser.add_argument("--test_n", default=200, type=int,
                    help='How many candidate boxes to sample for one image in test')

parser.add_argument("--test_batch_size", default=50, type=int,
                    help='How many context bboxes I evaluate in a batch')

parser.add_argument("--n_neighborhoods", default=1, type=int,
                    help='How many context images for one bboxes I sample for evaluation')

args = parser.parse_args()
train_dir = os.path.join(CKPT_DIR, args.run_name)

# Configurations for standard data augmentation
std_data_augmentation_config = {
    'X_out': 4,
    'brightness_prob': args.brightness_prob,
    'brightness_delta': 0.125,
    'contrast_prob': args.contrast_prob,
    'contrast_delta': 0.5,
    'hue_prob': args.hue_prob,
    'hue_delta': 0.07,
    'saturation_prob': args.saturation_prob,
    'saturation_delta': 0.5,
    'flip_prob': args.flip_prob,
    'crop_max_tries': 50,
    'fill_color': [x/255.0 for x in reversed(MEAN_COLOR)],
}

eval_log = '1evaluations'
evaluation_logfile = eval_log + '.txt'

def get_logging_config(run):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
            'short': {
                'format': '[%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'short',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': LOGS+run+'.log'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }
