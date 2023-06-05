# Training script adapted from https://www.youtube.com/watch?v=IHq1t7NxS8k

from UNet import UNet

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Hyperparams
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 36
num_epochs = 100
num_workers = 8
i_width = 256 * 6
i_height = 256 * 6
pin_memory = True
load_model = False

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Forward
        preds = model(data)
        loss = loss_fn(preds, targets)

        # Backward
        optimizer.zero_grad()
        optimizer.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimzer = optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    main()


import os
import numpy as np
import skimage.io as io

directory = "../panda_data/train_label_masks"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    image = io.imread(f)
    if np.sum(image) != 0:
        image = np.asarray(image)
        x, y, z = image.shape
        image = image.reshape((z, x, y))
        print(image.shape)
        print(np.max(image))
        print(np.max(image[0, :, :]))
        print(np.max(image[1, :, :]))
        print(np.max(image[2, :, :]))