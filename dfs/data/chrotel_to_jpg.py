import os
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy import io as sio
from skimage.io import imsave
from skimage.transform import resize
import arrow


dir0 = '/home/adiercke/Documents/data/chrotel/'
out = dir0 + 'new_jpeg/'


fchro = [{'filename':f, 'ts': arrow.get(f[18:26]+'T'+f[27:33])} for f in os.listdir(dir0) if f.startswith('chrotel_ha_lev2.0')]
nf = len(fchro)

for ifile in range(0, nf):
    chrotel = sio.readsav(dir0+fchro[ifile]['filename'])
    img = chrotel.pic
    hdr = chrotel.hdr
    date = chrotel.date
    pic = np.nan_to_num(img, posinf=0)
    pic0 = resize(pic, (1024, 1024))
    vv0 = 0.8
    vv1 = 1.3

    data = Normalize(vv0, vv1, clip=True)(pic0) * 2 - 1
    save_file = os.path.join(out, os.path.basename(fchro[ifile]['filename']).replace('sav', 'jpg'))
    plt.imsave(save_file, data, vmin=-1, vmax=1, cmap='gray')

