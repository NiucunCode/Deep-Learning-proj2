"""
Preprocess the dataset:

Using download_dataset.sh
or
Download dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

Then preprocess dataset...
"""

import os
import glob

from PIL import Image
from tqdm import tqdm

load = ['monet2photo', 'vangogh2photo', 'ukiyoe2photo', 'cezanne2photo']
save = ['Monet', 'Vangogh', 'Ukiyoe', 'Cezanne', 'Photo']
modes = ['train', 'test']

print('Start build dataset...')
for index in range(len(load)):
    for mode in modes:
        path = glob.glob(os.path.join("%s/" % load[index], "%sA/" % mode) + "*.*")
        save_path = os.path.join('data/', "%s/" % save[index], "%s/" % mode)
        os.makedirs(save_path, exist_ok=True)
        cur_attr_file = sorted(path)
        for filepath in tqdm(cur_attr_file):
            img = Image.open(filepath)
            basename = os.path.basename(filepath)
            img.save(os.path.join(save_path, basename))
        # for class 'Photo'
        if index == 3:
            path = glob.glob(os.path.join("%s/" % load[index], "%sB/" % mode) + "*.*")
            save_path = os.path.join('data/', "%s/" % save[index + 1], "%s/" % mode)
            os.makedirs(save_path, exist_ok=True)
            cur_attr_file = sorted(path)
            for filepath in tqdm(cur_attr_file):
                img = Image.open(filepath)
                basename = os.path.basename(filepath)
                img.save(os.path.join(save_path, basename))

print('Done')
