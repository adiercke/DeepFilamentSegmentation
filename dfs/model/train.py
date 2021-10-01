import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import pandas as pd

from dfs.data.data_set import ChroTelDataSet
import numpy as np

from dfs.evaluation.callback import PlotCallback
from dfs.model.util import iou, precision, recall, plot_results

epochs = 500
base_path = '/work2/adiercke/ml/DeepFilamentSegmentation/runs/006'
os.makedirs(base_path, exist_ok=True)
results_file = base_path + '/results.txt'

# images = sorted(glob.glob(os.path.join('/net/reko/work1/soe/adiercke/ChroTel/ml', '*_full.jpg')))  # 1024x1024 jpg
labels = sorted(glob.glob(os.path.join('/net/reko/work1/soe/adiercke/ChroTel/ml', '*yolov5.txt')))
images = [label[:-10] + 'full.jpg' for label in labels]
data = list(zip(labels, images))

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=0)
train_ds = ChroTelDataSet(train_data)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8)
valid_ds = ChroTelDataSet(valid_data, patch_size=(1024, 1024))
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=True, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=False, num_classes=1)
model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.to(device)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor([10]))

plot_callback = PlotCallback(*next(valid_loader.__iter__()), model, base_path)

for epoch in range(epochs):
    epoch_loss = []
    train_epoch_accuracy = []
    train_epoch_iou = []
    train_epoch_precision = []
    train_epoch_recall = []
    for x, y in tqdm(train_loader, total=len(train_loader)):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        y_pred = model(x)['out']

        loss = criterion(y_pred, y)
        loss = torch.mean(loss)
        loss.backward()

        y_pred = torch.round(torch.sigmoid(y_pred.detach())).float()
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
            y_pred = model(x)['out']

            y_pred = torch.round(torch.sigmoid(y_pred.detach())).float()
            accuracy = torch.mean((y == y_pred).float(), [1, 2, 3])
            iou_loss = iou(y_pred, y)
            precision_loss = precision(y_pred, y)
            recall_loss = recall(y_pred, y)

            valid_epoch_accuracy.extend(accuracy.cpu().numpy())
            valid_epoch_iou.extend(iou_loss.cpu().numpy())
            valid_epoch_precision.extend(precision_loss.cpu().numpy())
            valid_epoch_recall.extend(recall_loss.cpu().numpy())

    model.train()
    scheduler.step()

    plot_callback.call(epoch)
    print(
        '(%03d/%03d) [train: bce: %.03f; acc %.02f; iou %.03f, prec: %.03f, rec: %.03f] [valid: acc %.02f; iou %.03f, prec: %.03f, rec: %.03f]' % (
            epoch, epochs, np.mean(epoch_loss), np.mean(train_epoch_accuracy), np.mean(train_epoch_iou),
            np.mean(train_epoch_precision), np.mean(train_epoch_recall),
            np.mean(valid_epoch_accuracy), np.mean(valid_epoch_iou), np.mean(valid_epoch_precision),
        np.mean(valid_epoch_recall)))
    torch.save(model, os.path.join(base_path, 'last.pt'))

    # Write
    # with open(os.path.join(base_path, 'results.csv'), 'a') as f:
    #    f.write('(%03d/%03d) %5.03f %5.02f %5.03f' % (epoch, epochs, np.mean(epoch_loss), np.mean(train_epoch_accuracy), np.mean(train_epoch_iou)) + '\n')  # epoch, loss, accuracy, iou

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

#plot_results(base_path)