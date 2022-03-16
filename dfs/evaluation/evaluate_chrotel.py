import glob
import os
import shutil

import numpy as np
import torch
from skimage.io import imsave
from torch.utils.data import DataLoader
from torchvision.transforms import Pad
from tqdm import tqdm

from dfs.data.data_set import EvaluationDataSet

result_path = '/gpfs/gpfs0/robert.jarolim/filament/unet_v1/evaluation_chrotel_test'
os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=False)
model.to(device)
cp = torch.load('/gpfs/gpfs0/robert.jarolim/filament/unet_v1/checkpoint.pt', map_location=device)
model.load_state_dict(cp['m'])
model.eval()

files = sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/filament/yolov5_data/images/test/*.jpg'))

ds = EvaluationDataSet(files)
batch_size = 8
loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
file_batches = [files[x:x+batch_size] for x in range(0, len(files), batch_size)]

pad = Pad([12, 12, 12, 12], fill=-1)
with torch.no_grad():
    result = []
    for file_batch, x in tqdm(zip(file_batches, loader), total=len(file_batches)):
        x = pad(x)
        pred = model(x.to(device))
        result = (pred >= 0.5).cpu().numpy().astype(np.bool)[:, :, 12:-12, 12:-12]
        for f, res in zip(file_batch, result):
            imsave(os.path.join(result_path, os.path.basename(f)), res[0], check_contrast=False)


shutil.make_archive('/gpfs/gpfs0/robert.jarolim/filament/unet_v1/chrotel_test_evaluation', 'zip', result_path)



