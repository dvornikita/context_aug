import os

def check_dir(dirname):
    """This function creates a directory
    in case it doesn't exist"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# TODO: change EVAL_DIR to ROOT_DIR

# The project directory
EVAL_DIR = os.path.dirname('/'.join(os.path.realpath(__file__).split('/')[:-1]))

# Folder with datasets
DATASETS_DIR = check_dir(os.path.join(EVAL_DIR, 'Data/'))

# Where the checkpoints are stored
# CKPT_DIR = check_dir(os.path.join(EVAL_DIR, 'old_archive_cat/'))
CKPT_DIR = check_dir(os.path.join(EVAL_DIR, 'Weights/'))

# Where the logs are stored
LOGS = check_dir(os.path.join(EVAL_DIR, 'Logs/Experiments/'))

# Where the context estimated boxes are stored
RAW_CONTEXT_DIR = check_dir(os.path.join(DATASETS_DIR, 'context_probs/'))

# Where the context estimated boxes are stored
CONTEXT_MAPPING_DIR = check_dir(os.path.join(DATASETS_DIR, 'context_mapping/'))

# Where the imagenet weights are located
INIT_WEIGHTS_DIR = check_dir(os.path.join(EVAL_DIR, 'Weights/Imagenet/'))

# Where the demo images are located
DEMO_DIR = check_dir(os.path.join(EVAL_DIR, 'Demo/'))
