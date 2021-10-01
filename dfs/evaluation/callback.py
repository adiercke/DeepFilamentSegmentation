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
        fig, axs = plt.subplots(len(img_batch), 3, figsize=(3 * 3, len(img_batch) * 3))
        [ax.set_axis_off() for ax in np.ravel(axs)]
        for row, img, mask, pred in zip(axs, img_batch, mask_batch, pred_batch):
            row[0].imshow(img[0], cmap='gray')
            row[0].set_title('Input')
            row[1].imshow(mask[0], cmap='gray', vmin=0, vmax=1)
            row[1].set_title('Ground Truth Labels')
            row[2].imshow(pred[0], cmap='gray', vmin=0, vmax=1)
            row[2].set_title('Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, '%04d.jpg' % epoch), dpi=300)
        plt.close()

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.x.cuda())['out'].detach().cpu().numpy()
        self.model.train()
        return  y_pred
