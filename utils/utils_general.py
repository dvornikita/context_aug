import subprocess
import os
import numpy as np
from configs.paths import EVAL_DIR
from PIL import Image

palette = np.load(os.path.join(EVAL_DIR, 'Extra/palette.npy')).tolist()


def read_textfile(filename, skip_last_line=True):
    with open(filename, 'r') as f:
        container = f.read().split('\n')
        if skip_last_line:
            container = container[:-1]
    return container


def write_file(content, filename):
    with open(filename, 'w') as f:
        if type(content) == list:
            content = '\n'.join(content) + '\n'
        f.write(content)


def make_list(item):
    if isinstance(item, str):
        item = [item]
    return item


def array2palette(arr):
    im = Image.fromarray(np.uint8(arr))
    im.putpalette(palette)
    return im


def softlink(source, dest):
    if not os.path.exists(dest):
        bashCommand = 'ln -s %s %s' % (source, dest)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


def mask_vectors(vectors, mask):
    new_vectors = []
    for v in vectors:
        assert len(v) == len(mask)
        new_vectors.append(v[mask])
    return new_vectors
