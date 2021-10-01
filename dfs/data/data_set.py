import glob
import logging
import os
import random

from skimage.measure import label, regionprops
from skimage.morphology import disk
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import readsav

class ChroTelDataSet(Dataset):

    def __init__(self, data, patch_size=(512, 512)):
        self.data = data
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_path, img_path = self.data[idx]
        img = (plt.imread(img_path) / 255) * 2 - 1

        #original label-files
        labels = pd.read_csv(label_path, sep=' ', header=None)
        labels.columns = ['cl', 'n1', 'xx1', 'n2', 'yy1', 'n3', 'w', 'n4', 'h']
        xc = labels.xx1 * img.shape[0]
        yc = labels.yy1 * img.shape[0]
        w = labels.w * img.shape[0]
        h = labels.h * img.shape[0]

        segmentation_img = np.zeros(img.shape)

        for i in range(len(labels)):
            # for i in range(1):
            ww = w.values[i]
            hh = h.values[i]
            xx = xc.values[i] - ww / 2.
            yy = yc.values[i] - hh / 2.

            patch = img[int(yy):int(yy + hh), int(xx):int(xx + ww)]
            threshold = np.mean(patch) - np.std(patch)
            masked_patch = (patch <= threshold)

            patch_label = label(masked_patch, connectivity=1)
            try:
                regions = regionprops(patch_label)
            except Exception as ex:
                logging.error(ex)
                logging.error(img_path)
                logging.error(str(patch_label.shape))
                raise Exception('Invalid')
            small_labels = np.array([prop.label for prop in regions if prop.area / np.product(patch_label.shape) < 0.01])
            masked_patch[np.isin(patch_label, small_labels)] = 0

            segmentation_img[int(yy):int(yy + hh), int(xx):int(xx + ww)] = masked_patch

        segmentation_img *= disk(img.shape[0] // 2)[1:, 1:]

        # load from storage
        img, segmentation_img = random_patch(img, segmentation_img, self.patch_size)
        img = np.expand_dims(img, 0).astype(np.float32)
        segmentation_img = np.expand_dims(segmentation_img, 0).astype(np.float32)
        return img, segmentation_img

def random_patch(img, segmentation_img, patch_size):
    patch_size = patch_size[0]
    img_w, img_h = img.shape[:2]
    # select random patch
    patch_x, patch_y = random.randint(0, img_w - patch_size), random.randint(0, img_h - patch_size)
    patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    patch_segmentation = segmentation_img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

    return patch, patch_segmentation

