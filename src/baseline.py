import numpy as np
import os
import sys
import glob
import cv2
import PIL
import random
import openslide
import skimage.io
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#Misc imports
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import albumentations
import time

#Torch imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

device = torch.device('cuda')

from transformers import AutoImageProcessor, Swinv2Config, Swinv2Model
from transformers import ViTConfig, ViTModel
BASE_PATH = "/home/ubuntu/cs231nFinalProject/panda_data"
data_folder = os.path.join(BASE_PATH, "all_images")
masks_path = os.path.join(BASE_PATH, "train_label_masks")

all_df = pd.read_csv(os.path.join(BASE_PATH, "filtered_images.csv"))

print(f"Size of all images: {len(all_df)}")

images = list(all_df['image_id'])
labels = list(all_df['isup_grade'])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_df['fold'] = -1
for i, (train_idx, test_idx) in enumerate(skf.split(all_df, all_df['isup_grade'])):
    # Set up testing fold
    all_df.loc[test_idx, 'fold'] = i

# Create patches from WSI image
#If > 95% of tile is white, flag that tile

def create_patches(img):
    patch_size, image_size = 256, 256
    n_patches = 36
    
    patches = []
    H, W, _ = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size 
    pad_w = (patch_size - W % patch_size) % patch_size  
    
    # Pad with white
    padded_img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0,0]], constant_values=255)
    padded_img = padded_img.reshape(padded_img.shape[0] // patch_size, # number of tiles in h-dimension
            patch_size,
            padded_img.shape[1] // patch_size, # number of tiles in width dimension
            patch_size,
            3)

    padded_img = padded_img.transpose(0,2,1,3,4).reshape(-1, patch_size, patch_size, 3)
    useful_patches = (padded_img.reshape(padded_img.shape[0], -1).sum(1) < (0.95 * (patch_size ** 2) * 3 * 255)).sum()
    if len(padded_img) < n_patches:
        padded_img = np.pad(padded_img, [[0, n_tiles - len(padded_img)], [0,0], [0,0], [0,0]], constant_values=255)
    # select most representative tiles
    indices = np.argsort(padded_img.reshape(padded_img.shape[0],-1).sum(-1))[:n_patches]
    filtered_img = padded_img[indices]
    for i in range(len(filtered_img)):
        patches.append({'patches': filtered_img[i], 'idx': i})
    return patches, useful_patches >= n_patches



class PandaDataset(Dataset):
    def __init__(self, df, image_size, n_tiles = 36, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.transform = transforms
    def __len__(self):
        return len(self.df)
    # Custom get item function
    def __getitem__(self, idx):
        """
        if idx > len(self.df):
            idx = len(self.df) - 1
        img_path = os.path.join(data_folder, f"{self.df.iloc[idx]['image_id']}.tiff")
        indices = list(range(self.n_tiles))
        img = skimage.io.MultiImage(img_path)[-1]
        patches, keep_slide = create_patches(img)

        rows = int(np.sqrt(self.n_tiles))
        imgs = np.zeros((self.image_size * rows, self.image_size * rows, 3))
        for h in range(rows):
            for w in range(rows):
                curr_idx = h * rows + w
                if indices[curr_idx] < len(patches):
                    curr_img = patches[indices[curr_idx]]['patches']
                else:
                    # White tile
                    curr_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                # Remove white background
                curr_img = 255 - curr_img
                if self.transform is not None:
                    curr_img = self.transform(image=curr_img)['image']
                h_scaled = h * self.image_size
                w_scaled = w * self.image_size
                imgs[h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size] = curr_img

        if self.transform is not None:
            imgs = self.transform(image=imgs)['image']
        imgs = imgs.astype(np.float64)
        # Normalize back to 0-255
        imgs /= 255
        imgs = imgs.transpose(2, 0, 1)

        # 5 classes w/ binning label
        labels = np.zeros(5).astype(np.float32)
        labels[: self.df.iloc[idx]["isup_grade"]] = 1.
        return torch.tensor(imgs), torch.tensor(labels)
        """

# Create transforms of patches
transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])

# Load tiles into train/test dataloaders
df_train = all_df.loc[np.where((all_df["fold"] != 0))[0]]
df_test = all_df.loc[np.where((all_df["fold"] == 0))[0]]

train_loader = PandaDataset(df_train, 256, 36, transforms=transforms_train)
test_loader = PandaDataset(df_test, 256, 36, transforms=None)

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=50, sampler=RandomSampler(train_loader),
                                            num_workers = 1)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=50, sampler=RandomSampler(test_loader),
                                            num_workers = 1)

print(len(train_loader))
print(len(test_loader))

densenet = torchvision.models.densenet121(pretrained=True).to(device)
num_features = densenet.classifier.in_features
# 5 grading classes
num_classes = 5 
densenet.classifier = nn.Linear(num_features, num_classes).to(device)

for param in densenet.parameters():
    param.requires_grad = False

#model = densenet.to(device)
#model.cuda()

# 256x256 pre-trained
configuration = ViTConfig(image_size=1536)
model = ViTModel(configuration).to(device)
loss_func = nn.BCEWithLogitsLoss()

def train_loop(loader, opt):
    model.train()
    train_loss = []
    #preds, targets = [], []
    for (data, label) in tqdm(loader):
        data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        opt.zero_grad()
        logits = model(data)
        loss = loss_func(logits, label)
        #pred = logits.sigmoid().sum(1).detach().round()
        #preds.append(pred)
        #targets.append(label.sum(1))

        loss.backward()
        opt.step()
        train_loss.append(loss.detach().cpu().numpy())
    return train_loss

def test_loop(loader):
    test_loss = []
    preds, targets = [], []
    with torch.no_grad():
        for (data, label) in tqdm(loader):
            data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
            logits = densenet(data)
            loss = loss_func(logits, label)
            # Get pred by taking sigmoid of logits
            pred = logits.sigmoid().sum(1).detach().round()
            preds.append(pred)
            # Sum over bins
            targets.append(label.sum(1))
            
            test_loss.append(loss.detach().cpu().numpy())
        final_test_loss = np.mean(test_loss)

    preds = torch.cat(preds).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    acc = np.mean(preds == targets) * 100
    return final_test_loss, acc

#Zero-shot Evaluation
test_loop, acc = test_loop(test_loader)
print(acc)
