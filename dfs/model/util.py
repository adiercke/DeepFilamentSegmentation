import torch
import matplotlib.pyplot as plt
import os
import pandas as pd


def iou(y_pred, y, eps=1e-9):
    labels = y == 1
    pred = y_pred == 1

    intersection = (pred & labels).float().sum((1, 2, 3))
    union = (pred | labels).float().sum((1, 2, 3))

    iou = (intersection + eps) / (union + eps)  # eps to avoid division by 0
    # print('iou: ', intersection, union, iou)
    return iou


def precision(y_pred, y, eps=1e-9):
    labels = y == 1
    pred = y_pred == 1

    TP = ((pred == 1) & (labels == 1)).float().sum((1, 2, 3))
    FP = ((pred == 1) & (labels == 0)).float().sum((1, 2, 3))
    Prec = TP / ((TP + FP) + eps)
    # print('precision: ', TP, FP, Prec)
    return Prec


def recall(y_pred, y, eps=1e-9):
    labels = y == 1
    pred = y_pred == 1

    TP = ((pred == 1) & (labels == 1)).float().sum((1, 2, 3))
    FN = ((pred == 0) & (labels == 1)).float().sum((1, 2, 3))
    Rec = TP / ((TP + FN) + eps)
    # print('recall: ', TP, FN, Rec)
    return Rec


def plot_results(base_path):
    results = pd.read_csv(os.path.join(base_path, 'results.csv'))
    plt.figure(figsize=(2 * 3, 5*3))

    e = results.epoch.values
    loss = results.loss.values
    train_acc = results.train_accuracy.values
    train_iou = results.train_iou.values
    train_recall = results.train_recall.values
    train_precision = results.train_precision.values
    valid_acc = results.valid_accuracy.values
    valid_iou = results.valid_iou.values
    valid_recall = results.valid_recall.values
    valid_precision = results.valid_precision.values

    plt.subplot(511)
    plt.plot(e, loss, 'rx')
    plt.title('BCE')

    plt.subplot(512)
    plt.plot(e, train_acc, 'rx', e, valid_acc, 'bx')
    plt.title('Accuracy')

    plt.subplot(513)
    plt.plot(e, train_iou, 'rx', e, valid_iou, 'bx')
    plt.title('IoU')

    plt.subplot(514)
    plt.plot(e, train_precision, 'rx', e, valid_precision, 'bx')
    plt.title('Precision')

    plt.subplot(515)
    plt.plot(e, train_recall, 'rx', e, valid_recall, 'bx')
    plt.title('Recall')

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'results.png'), dpi=200)
    plt.close()
