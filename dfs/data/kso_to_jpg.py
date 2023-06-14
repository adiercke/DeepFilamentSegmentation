import glob
import os
from multiprocessing import Pool

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm

from dfs.data.data_set import get_data

img_path = '/gpfs/gpfs0/robert.jarolim/data/filament/kso_img'
os.makedirs(img_path, exist_ok=True)
files = glob.glob('/gpfs/gpfs0/robert.jarolim/data/filament/kso/*.fts.gz')

def convert(f):
    save_file = os.path.join(img_path, os.path.basename(f).replace('fts.gz', 'jpg'))
    if os.path.exists(save_file):
        return
    try:
        data = get_data(f)
        plt.imsave(save_file, data[0], vmin=-1, vmax=1, cmap='gray', origin='lower')
    except Exception as ex:
        print(ex)
        return


with Pool(12) as p:
    [None for _ in tqdm(p.imap_unordered(convert, files), total=len(files))]