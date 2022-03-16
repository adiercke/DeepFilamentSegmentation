import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/filament/unet_v1/evaluation/*.jpg'))

timeline = []
for f in tqdm(files):
    img = plt.imread(f)
    slice = np.sum(img, 0)
    timeline += [slice]


plt.figure(figsize=(12, 2))
plt.imshow(np.stack(timeline, 1), norm=LogNorm(vmin=100, vmax=10000))
plt.savefig('/gpfs/gpfs0/robert.jarolim/filament/unet_v1/timeline.jpg')
plt.close()