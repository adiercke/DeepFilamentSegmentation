import os

import glob
import torch
from torch.utils.data import DataLoader
import numpy as np
from dfs.data.data_set import ChroTelDataSet
from dfs.model.util import iou, precision, recall

base_path = '/work2/adiercke/ml/DeepFilamentSegmentation/runs/004'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load(os.path.join(base_path, 'last.pt'))
model.to(device)
model.eval()

labels = sorted(glob.glob(os.path.join('/work2/adiercke/ml/yolov5_data/labels/test', 'chrotel*.txt')))
images = [label.replace('.txt', '.jpg').replace('labels', 'images') for label in labels]
test_data = list(zip(labels, images))

test_ds = ChroTelDataSet(test_data, patch_size=(1024, 1024))
test_loader = DataLoader(test_ds, batch_size=8, shuffle=True, num_workers=2)

test_epoch_accuracy = []
test_epoch_iou = []
test_epoch_precision = []
test_epoch_recall = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)['out']
        y_pred = torch.round(torch.sigmoid(y_pred.detach())).float()
        accuracy = torch.mean((y == y_pred).float(), [1, 2, 3])
        iou_loss = iou(y_pred, y)
        precision_loss = precision(y_pred, y)
        recall_loss = recall(y_pred, y)
        test_epoch_accuracy.extend(accuracy.cpu().numpy())
        test_epoch_iou.extend(iou_loss.cpu().numpy())
        test_epoch_precision.extend(precision_loss.cpu().numpy())
        test_epoch_recall.extend(recall_loss.cpu().numpy())

print('[test: acc %.02f; iou %.03f, prec: %.03f, rec: %.03f]' % (
    np.mean(test_epoch_accuracy), np.mean(test_epoch_iou), np.mean(test_epoch_precision),
    np.mean(test_epoch_recall)))
