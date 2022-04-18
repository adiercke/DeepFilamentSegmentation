from glob import glob

import matplotlib.pyplot as plt
from tqdm import tqdm

for f in tqdm(glob('/gpfs/gpfs0/robert.jarolim/data/filament/yolov5_data/images/**/*.jpg', recursive=True)):
    im_array = plt.imread(f)
    plt.imsave(f, im_array, origin='lower')