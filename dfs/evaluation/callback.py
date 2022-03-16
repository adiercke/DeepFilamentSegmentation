import os

import torch
from matplotlib import pyplot as plt

import numpy as np

class PlotCallback:

    def __init__(self, x, y, model, path):
        self.x, self.y = x, y
        self.path = path
        self.model = model

    def call(self, epoch):
        img_batch, mask_batch = self.x, self.y
        pred_batch = self.predict()
        fig, axs = plt.subplots(len(img_batch), 5, figsize=(5 * 3, len(img_batch) * 3))
        [ax.set_axis_off() for ax in np.ravel(axs)]
        for row, img, mask, pred in zip(axs, img_batch, mask_batch, pred_batch):
            row[0].imshow(img[0], cmap='gray')
            row[0].set_title('Input')
            row[1].imshow(mask[0], cmap='gray', vmin=0, vmax=1)
            row[1].set_title('Ground Truth Labels')
            row[2].imshow(img[0], cmap='gray')
            row[2].contour(mask[0], levels=[0.5], colors=['red'], linewidths=0.5)
            row[2].set_title('Ground Truth Labels')
            row[3].imshow(pred[0], cmap='gray', vmin=0, vmax=1)
            row[3].set_title('Prediction')
            row[4].imshow(img[0], cmap='gray')
            row[4].contour(pred[0], levels=[0.5], colors=['red'], linewidths=0.5)
            row[4].set_title('Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, '%04d.jpg' % epoch), dpi=100)
        plt.close()

    def predict(self):
        with torch.no_grad():
            y_pred = self.model(self.x.cuda()).detach().cpu().numpy()
        return  y_pred
