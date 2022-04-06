import logging
import random

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from skimage.measure import label, regionprops
from skimage.morphology import disk
from skimage.transform import resize
from sunpy.map import Map, all_coordinates_from_map
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from torch.utils.data import Dataset
import albumentations as A

### new libs
from aotools.functions import zernike
from skimage import morphology
from scipy import ndimage

class ImageDataSet(Dataset):

    def __init__(self, images, labels, patch_size=(512, 512), augmentation=True):
        self.data = list(zip(images, labels))
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.transform = A.Compose([
                A.Blur(p=0.1),
                A.MedianBlur(p=0.1),
                A.CLAHE(p=0.1),
                A.RandomBrightnessContrast(p=0.1),
                A.RandomGamma(p=0.1),])
        self.image_transform = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True,
                                                  shear_range=0.2, fill_mode='constant', cval=-1, data_format='channels_first')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        img = plt.imread(img_path)
        if self.augmentation:
            img = self.transform(image=img)['image']
        img = (img / 255) * 2 - 1
        img = img.astype(np.float32)
        # original label-files
        labels = pd.read_csv(label_path, delim_whitespace=True, header=None)
        labels.columns = ['cl', 'xx1', 'yy1', 'w', 'h']
        xc = labels.xx1 * img.shape[0]
        yc = labels.yy1 * img.shape[0]
        w = labels.w * img.shape[0]
        h = labels.h * img.shape[0]

        segmentation_img = np.zeros(img.shape)
        pst = zernike.zernike_noll(1, img.shape[0])
        img[pst < 1] = np.NaN
        for i in range(len(labels)):
            # for i in range(1):
            ww = w.values[i]
            hh = h.values[i]
            xx = xc.values[i] - ww / 2.
            yy = yc.values[i] - hh / 2.

            patch = img[int(yy):int(yy + hh), int(xx):int(xx + ww)]
            threshold = np.nanmean(patch) - np.nanstd(patch)
            masked_patch = (patch <= threshold)

            patch_label = label(masked_patch, connectivity=1)
            try:
                regions = regionprops(patch_label)
            except Exception as ex:
                logging.error(ex)
                logging.error(img_path)
                logging.error(str(patch_label.shape))
                raise Exception('Invalid')
            small_labels = np.array(
                [prop.label for prop in regions if prop.area / np.product(patch_label.shape) < 0.01])
            masked_patch[np.isin(patch_label, small_labels)] = 0

            segmentation_img[int(yy):int(yy + hh), int(xx):int(xx + ww)] += masked_patch

        # set overlapping label to [0, 1] and crop off-limb
        segmentation_img[segmentation_img > 1] = 1
        segmentation_img *= disk(img.shape[0] // 2)[1:, 1:]

        # TODO remove padding
        img = np.nan_to_num(img, nan=-1)
        img = to_shape(img, (1024, 1024), -1)
        segmentation_img = to_shape(segmentation_img, (1024, 1024), 0)
        #
        img = np.expand_dims(img, 0).astype(np.float32)
        segmentation_img = np.expand_dims(segmentation_img, 0).astype(np.float32)
        if self.augmentation:
            params = self.image_transform.get_random_transform(img.shape)
            img = self.image_transform.apply_transform(img, params)
            segmentation_img = self.image_transform.apply_transform(segmentation_img, params)
            segmentation_img = segmentation_img >= 0.5 # back to boolean
        img, segmentation_img = random_patch(img, segmentation_img, self.patch_size)
        return np.array(img, dtype=np.float32), np.array(segmentation_img, dtype=np.float32)


class EvaluationDataSet(Dataset):
    def __init__(self, images,):
        self.data = list(images)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = plt.imread(img_path)
        img = (img / 255) * 2 - 1
        img = np.expand_dims(img, 0).astype(np.float32)
        return img


class FITSDataSet(Dataset):

    def __init__(self, files):
        self.data = files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return get_data(self.data[idx])

def random_patch(img, segmentation_img, patch_size):
    img_w, img_h = img.shape[1:]
    # select random patch
    patch_x, patch_y = random.randint(0, img_w - patch_size[0]), random.randint(0, img_h - patch_size[1])
    patch = img[:, patch_y:patch_y + patch_size[0], patch_x:patch_x + patch_size[1]]
    patch_segmentation = segmentation_img[:, patch_y:patch_y + patch_size[0], patch_x:patch_x + patch_size[1]]

    return patch, patch_segmentation

def to_shape(a, shape, c):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant', constant_values=c)

def polarint(img):
    ### adapted from IDL STOOLS Library stools_polarint.pro (Kuckein et al. 2017, IAU Sym. 327, 20)
    s = img.shape
    rmax = np.max(s) / 2
    r = np.arange(rmax)
    nt = np.round(2 * np.pi * rmax + 1)
    theta = 2 * np.pi * np.arange(nt) / (nt - 1)

    rm = np.matrix(r)
    ctm = np.matrix(np.cos(theta))
    stm = np.matrix(np.sin(theta))
    xpolar = s[0] / 2. + np.dot(rm.T, ctm)
    ypolar = s[1] / 2. + np.dot(rm.T, stm)

    pp = np.sum(ndimage.map_coordinates(img, [xpolar, ypolar], order=1), axis=1) / nt
    return (pp)


def dist(n):
    axis = np.linspace(-n / 2 + 1, n / 2, n)
    result = np.sqrt(axis ** 2 + axis[:, np.newaxis] ** 2)
    nroll = np.roll(result, int(n / 2 + 1), axis=(0, 1))
    return (nroll)


def polarfit(prof):
    ### adapted from IDL STOOLS Library stools_polarfit.pro (Kuckein et al. 2017, IAU Sym. 327, 20)
    nr = int(prof.shape[0])
    index = np.round(dist(2 * nr) * 1000.)
    newnr = 1000 * nr
    prof0 = np.repeat(prof, newnr / nr, axis=0)

    ind1 = index.reshape(index.shape[0] * index.shape[1])
    ll = list(ind1)

    pp = np.zeros(len(ll))
    for i in range(len(ll)):
        inn = int(ind1[i])
        if inn >= prof0.shape[0]:
            inn = -1
        pp[i] = prof0[inn]

    pp = pp.reshape(index.shape[0], index.shape[1]) * (index < nr * 1000.)
    img = np.roll(pp, int(index.shape[0] / 2.), axis=(0, 1))
    return (img)


def get_data(fits_file):
    resolution = 1024
    s_map = Map(fits_file)
    r_obs_pix = s_map.rsun_obs / s_map.scale[0]
    scale_factor = resolution / (2 * r_obs_pix.value)
    s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
    arcs_frame = (resolution / 2) * s_map.scale[0].value
    s_map = s_map.submap(bottom_left=SkyCoord(- arcs_frame * u.arcsec, - arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                         top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
    pad_x = s_map.data.shape[0] - resolution
    pad_y = s_map.data.shape[1] - resolution
    s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                         top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)
    #
    # s_map.data /=  np.nanmedian(s_map.data)
    # LDC
    coords = all_coordinates_from_map(s_map)
    radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value
    radial_distance[radial_distance >= 1] = np.NaN
    ideal_correction = np.cos(radial_distance * np.pi / 2)

    condition = np.logical_not(np.isnan(np.ravel(ideal_correction)))
    map_list = np.ravel(s_map.data)[condition]
    correction_list = np.ravel(ideal_correction)[condition]

    fit = np.polyfit(correction_list, map_list, 6)
    poly_fit = np.poly1d(fit)

    map_correction = poly_fit(ideal_correction)
    #
    data0 = s_map.data / map_correction
    # intensity correction with Zernike polynomials
    nz = 36  # number of Zernike polynomials
    zk = np.array([zernike.zernike_noll(i, resolution) for i in range(1, nz + 1)])
    pst = zernike.zernike_noll(1, resolution)
    data = data0.copy()
    # normalize image
    i, j = np.where(pst)
    data = np.nan_to_num(data, nan=-1)
    # compute mask
    thres1 = np.median(data[i, j]) - 3 * np.std(data[i, j])
    thres2 = np.median(data[i, j]) + 3 * np.std(data[i, j])
    mask2 = (data < thres2)
    mask1 = (data > thres1)
    mask = mask1 - (1 - mask2)
    masko = morphology.binary_opening(mask, morphology.disk(15))
    zkmask = morphology.binary_opening(masko, morphology.disk(2))
    selection = np.where(zkmask, data, -1)
    # solving linear equation
    wf = selection[selection > -1]
    zk0 = zk[:, selection > -1]
    zkt = np.transpose(zk0)
    lsolv = np.linalg.lstsq(zkt, wf, rcond=None)
    cf = lsolv[0]
    zkfit = np.sum(np.array([((zernike.zernike_noll(i, resolution)) * cf[i - 1]) for i in range(1, nz - 1)]), axis=0)
    zkfit0 = zkfit.copy()
    zkfit0[pst < 0] = 1.

    ##background:
    img_filled = data.copy()
    mask = np.logical_and(selection <= -1, pst > 0, where=True)
    i, j = np.where(mask)
    img_filled[i, j] = np.median(data)
    bg0 = img_filled / zkfit0
    bgp = polarint(bg0)
    bg = polarfit(bgp)
    cor = zkfit0 * bg
    res = data / cor

    ##################################################################################
    data = Normalize(0.8, 1.3, clip=True)(res) * 2 - 1
    data = np.nan_to_num(data, nan=-1)
    #
    return np.array(data[None, :, :], dtype=np.float32)
