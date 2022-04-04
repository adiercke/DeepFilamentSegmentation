import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from skimage.io import imsave
from sklearn.utils import shuffle
from tqdm import tqdm

from dfs.data.data_set import get_data

img_path = '/gpfs/gpfs0/robert.jarolim/data/filament/gong_img'
os.makedirs(img_path, exist_ok=True)

qualities = pd.read_csv('/gpfs/gpfs0/robert.jarolim/siqa/version1/evaluation/qualities.csv', index_col=0)
qualities = qualities[qualities.quality < 0.25]
files = list(qualities.file)
files = shuffle(files)
def convert(f):
    save_file = os.path.join(img_path, os.path.basename(f).replace('fits', 'jpg'))
    if os.path.exists(save_file):
        return
    try:
        data = get_data(f)
        data = ((data + 1) / 2 * 255).astype('uint8')
        imsave(save_file, data[0])
    except Exception as ex:
        print(ex)
        return

with Pool(8) as p:
    [None for _ in tqdm(p.imap_unordered(convert, files), total=len(files))]