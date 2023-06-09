import numpy as np
import os
import sys
import glob
import cv2
import PIL
import random
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
import glob
import copy

#Torch imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

sys.path.append("/home/ubuntu/cs231nFinalProject/src/mtl_exps/mtl_models")
from hiptMTL import MTLHIPT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_PATH = "/home/ubuntu/cs231nFinalProject/panda_data"
data_folder = os.path.join(BASE_PATH, "image_patches")
masks_path = os.path.join(BASE_PATH, "mask_patches")

model = MTLHIPT().to(device)
model.load_state_dict(torch.load("/home/ubuntu/cs231nFinalProject/src/mtl_exps/hipt_mtl4_final_params.pth"))

def get_item():
        img_dir_path = os.path.join(data_folder, self.df.iloc[idx]['image_id'])
        mask_dir_path = os.path.join(masks_path, self.df.iloc[idx]['image_id'])

        # Collect
        image_tensors = []
        mask_tensors = []

        for i in range(self.n_tiles):
            # Get respective patch and mask
            subscript = "_" + str(i) + ".png"
            ID = self.df.iloc[idx]['image_id'] + subscript
            patch = os.path.join(img_dir_path, ID)
            mask = os.path.join(mask_dir_path, ID)

            # Create Image -> (R, G, B), H, W
            curr_img = cv2.imread(patch)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            curr_img = curr_img.transpose(2,0,1)

            # Create Mask -> 1, H, W
            curr_mask = cv2.imread(mask)
            curr_mask = curr_mask.transpose(2, 0, 1)
            curr_mask = np.max(curr_mask, axis=0)

            # Apply Augmentations
            if self.transform is not None:
                   augmentations = self.transform(image=curr_img, mask=curr_mask)
                   curr_img = augmentations["image"]
                   curr_mask = augmentations["mask"]

            image_tensors.append(torch.tensor(curr_img).float())
            mask_tensors.append(torch.tensor(curr_mask))

        # Create Big Tensor -> C x 36 x H x W
        imgs = torch.zeros((3, 1536, 1536)).float()
        masks = torch.zeros((1536, 1536))

        for i in range(5):
            for j in range(5):
                imgs[:, 256 * i : 256 * (i + 1), 256 * j : 256 * (j + 1)] = image_tensors[i * 6 + j]
                masks[256 * i : 256 * (i + 1), 256 * j : 256 * (j + 1)] = mask_tensors[i * 6 + j]

        assert torch.equal(masks[256:512, 768:1024], mask_tensors[9])

        # Rescale mask if needed
        if self.df.iloc[idx]["data_provider"] == "radboud":
            masks = torch.where(masks <= 1, masks, torch.where(masks <= 2, 1, 2))

        return imgs, masks
