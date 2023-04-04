import glob
import os
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader
from torchvision.transforms import Pad
from tqdm import tqdm

from dfs.data.data_set import EvaluationDataSet

base_path = '/gpfs/gpfs0/robert.jarolim/filament/unet_v3'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=False)
model.to(device)
cp = torch.load(os.path.join(base_path, 'checkpoint.pt'), map_location=device)
model.load_state_dict(cp['m'])
model.eval()

def evaluate(files, result_path, batch_size = 8):
    ds = EvaluationDataSet(files)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    file_batches = [files[x:x + batch_size] for x in range(0, len(files), batch_size)]

    with torch.no_grad():
        for file_batch, x in tqdm(zip(file_batches, loader), total=len(file_batches)):
            pred = model(x.to(device))
            result = (pred >= 0.5).cpu().numpy().astype(np.bool)
            for f, res in zip(file_batch, result):
                plt.imsave(os.path.join(result_path, os.path.basename(f)), res[0], cmap='gray', vmin=0, vmax=1)

    shutil.make_archive(result_path, 'zip', result_path)


ct_result_path = os.path.join(base_path, 'evaluation_chrotel_test')
os.makedirs(ct_result_path, exist_ok=True)
ct_files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/filament/yolov5_data/images/test/*.jpg'))
evaluate(ct_files, ct_result_path)

gong_result_path = os.path.join(base_path, 'evaluation_gong')
os.makedirs(gong_result_path, exist_ok=True)
gong_files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/filament/gong_img/*.jpg'))
evaluate(gong_files, gong_result_path)
