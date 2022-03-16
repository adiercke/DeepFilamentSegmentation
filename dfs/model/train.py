import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dfs.data.data_set import ImageDataSet
from dfs.evaluation.callback import PlotCallback
from dfs.model.util import iou, precision, recall, plot_results


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=8)
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

epochs = int(args.epochs)
base_path = args.base_path
os.makedirs(base_path, exist_ok=True)
checkpoint_file = base_path + '/checkpoint.pt'

labels_train_path = args.labels_train_path
images_train_path = args.images_train_path
labels_valid_path = args.labels_valid_path
images_valid_path = args.images_valid_path

train_labels_gong = glob.glob('/beegfs/home/robert.jarolim/projects/yolov5/runs/detect/exp3/labels/????-0[2-9]-*.txt')
train_images_gong = [os.path.join('/gpfs/gpfs0/robert.jarolim/data/filament/gong_img', os.path.basename(f).replace('txt', 'jpg')) for f in train_labels_gong]

train_labels_ct = sorted(glob.glob(os.path.join(labels_train_path, 'chrotel_*.txt')))
train_images_ct = [os.path.join(images_train_path, os.path.basename(f).replace('txt', 'jpg')) for f in train_labels_ct]

valid_labels = sorted(glob.glob(os.path.join(labels_valid_path, 'chrotel_*.txt')))
valid_images = [os.path.join(images_valid_path, os.path.basename(f).replace('txt', 'jpg')) for f in valid_labels]

train_ds = ImageDataSet(train_images_gong + train_images_ct, train_labels_gong + train_labels_ct, patch_size=(512, 512), augmentation=False)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=args.num_workers)
valid_ds = ImageDataSet(valid_images, valid_labels, patch_size=(1024, 1024), augmentation=False)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=True, num_workers=args.num_workers)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=False)
#torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=False, num_classes=1)
# model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.to(device)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)#torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
criterion = nn.BCELoss(weight=torch.cuda.FloatTensor([10]))

plot_callback = PlotCallback(*next(valid_loader.__iter__()), model, base_path)

init_epoch = 0
if os.path.exists(checkpoint_file):
    cp = torch.load(checkpoint_file)
    init_epoch = cp['epoch'] + 1
    model.load_state_dict(cp['m'])
    opt.load_state_dict(cp['o'])
    # scheduler.load_state_dict(cp['s'])

plot_callback.call(0)
for epoch in range(init_epoch, epochs):
    epoch_loss = []
    train_epoch_accuracy = []
    train_epoch_iou = []
    train_epoch_precision = []
    train_epoch_recall = []
    for x, y in tqdm(train_loader, total=len(train_loader)):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        y_pred = model(x)#['out']

        loss = criterion(y_pred, y)
        loss = torch.mean(loss)
        loss.backward()

        y_pred = torch.round(y_pred.detach()).float()
        accuracy = torch.mean((y == y_pred).float(), [1, 2, 3])
        iou_loss = iou(y_pred, y)
        precision_loss = precision(y_pred, y)
        recall_loss = recall(y_pred, y)

        epoch_loss.append(loss.detach().cpu().numpy())
        train_epoch_accuracy.extend(accuracy.cpu().numpy())
        train_epoch_iou.extend(iou_loss.cpu().numpy())
        train_epoch_precision.extend(precision_loss.cpu().numpy())
        train_epoch_recall.extend(recall_loss.cpu().numpy())

        opt.step()
    # do evaluation
    valid_epoch_accuracy = []
    valid_epoch_iou = []
    valid_epoch_precision = []
    valid_epoch_recall = []
    with torch.no_grad():
        model.eval()
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            y_pred = torch.round(y_pred.detach()).float()
            accuracy = torch.mean((y == y_pred).float(), [1, 2, 3])
            iou_loss = iou(y_pred, y)
            precision_loss = precision(y_pred, y)
            recall_loss = recall(y_pred, y)

            valid_epoch_accuracy.extend(accuracy.cpu().numpy())
            valid_epoch_iou.extend(iou_loss.cpu().numpy())
            valid_epoch_precision.extend(precision_loss.cpu().numpy())
            valid_epoch_recall.extend(recall_loss.cpu().numpy())

        plot_callback.call(epoch)
        model.train()
    # scheduler.step()

    print(
        '(%03d/%03d) [train: bce: %.03f; acc %.02f; iou %.03f, prec: %.03f, rec: %.03f] [valid: acc %.02f; iou %.03f, prec: %.03f, rec: %.03f]' % (
            epoch, epochs, np.mean(epoch_loss), np.mean(train_epoch_accuracy), np.mean(train_epoch_iou),
            np.mean(train_epoch_precision), np.mean(train_epoch_recall),
            np.mean(valid_epoch_accuracy), np.mean(valid_epoch_iou), np.mean(valid_epoch_precision),
            np.mean(valid_epoch_recall)))
    torch.save(model, os.path.join(base_path, 'last.pt'))
    torch.save({'epoch': epoch, 'm': model.state_dict(), 'o':opt.state_dict()}, checkpoint_file)

    dict = {'epoch': [epoch], 'loss': [np.mean(epoch_loss)], 'train_accuracy': [np.mean(train_epoch_accuracy)],
            'train_iou': [np.mean(train_epoch_iou)], 'train_precision': [np.mean(train_epoch_precision)],
            'train_recall': [np.mean(train_epoch_recall)], 'valid_accuracy': [np.mean(valid_epoch_accuracy)],
            'valid_iou': [np.mean(valid_epoch_iou)], 'valid_precision': [np.mean(valid_epoch_precision)],
            'valid_recall': [np.mean(valid_epoch_recall)]}
    df = pd.DataFrame.from_dict(dict)
    if epoch == 0:
        df.to_csv(os.path.join(base_path, 'results.csv'), header=True)
    else:
        df.to_csv(os.path.join(base_path, 'results.csv'), mode='a', header=False)

    plot_results(base_path)

# plot_results(base_path)