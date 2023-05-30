import argparse
import glob
import os
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dfs.data.data_set import EvaluationDataSet

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to the model weights')
parser.add_argument('--data_path', type=str, required=True, help='path to the images')
parser.add_argument('--result_path', type=str, required=True, help='path to the result directory')

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
data_path = args.data_path
result_path = args.result_path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1,
                       init_features=32, pretrained=False)
model.to(device)
cp = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(cp['m'])
model.eval()


def evaluate(files, result_path, batch_size=8):
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


os.makedirs(result_path, exist_ok=True)
ct_files = sorted(glob.glob(data_path))
evaluate(ct_files, result_path)
