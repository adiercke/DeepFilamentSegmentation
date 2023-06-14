import glob
import shutil

import numpy as np
import pandas
from matplotlib import pyplot as plt
import os

for f in glob.glob('/gpfs/gpfs0/robert.jarolim/data/gregor/labels_0/*.txt'):
    d = pandas.read_csv(f, header=None, names=['class', 'cx', 'cy', 'w', 'h'], delim_whitespace=True)
    if len(d) == 0 or d['class'][0] == 'Bad':
        continue
    #
    img = plt.imread(f'/gpfs/gpfs0/robert.jarolim/data/gregor/images/{os.path.basename(f).replace(".txt", ".jpg")}')
    #
    d['cx'] = d['cx'].divide(img.shape[1])
    d['cy'] = d['cy'].divide(img.shape[0])
    #
    d['w'] = d['w'].divide(img.shape[1])
    d['h'] = d['h'].divide(img.shape[0])
    #
    arr = np.array(d)
    np.savetxt(f'/gpfs/gpfs0/robert.jarolim/data/gregor/labels/{os.path.basename(f)}',
               arr, fmt="%i %8.2f %8.2f %8.2f %8.2f")


files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/gregor/labels/*.txt'))
os.makedirs('/gpfs/gpfs0/robert.jarolim/data/gregor/labels/train', exist_ok=True)
os.makedirs('/gpfs/gpfs0/robert.jarolim/data/gregor/labels/test', exist_ok=True)
[shutil.move(f, f.replace('labels/', 'labels/train/')) for f in files[:-100]]
[shutil.move(f, f.replace('labels/', 'labels/test/')) for f in files[-100:]]

files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/gregor/images/*.jpg'))
os.makedirs('/gpfs/gpfs0/robert.jarolim/data/gregor/images/train', exist_ok=True)
os.makedirs('/gpfs/gpfs0/robert.jarolim/data/gregor/images/test', exist_ok=True)
[shutil.move(f, f.replace('images/', 'images/train/')) for f in files[:-100]]
[shutil.move(f, f.replace('images/', 'images/test/')) for f in files[-100:]]