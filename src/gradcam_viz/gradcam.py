import os
import numpy as np 
import pandas as pd 

import sys
import cv2
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn.functional as F


from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BASE_PATH = "/home/ubuntu/cs231nFinalProject/panda_data"
data_folder = os.path.join(BASE_PATH, "image_patches")
masks_path = os.path.join(BASE_PATH, "mask_patches")

all_df = pd.read_csv(os.path.join(BASE_PATH, "filtered_image_patches.csv"))

print(f"Total number of image patches: {len(all_df)}")

images = list(all_df['image_id'])
labels = list(all_df['isup_grade'])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_df['fold'] = -1
for i, (train_idx, test_idx) in enumerate(skf.split(all_df, all_df['isup_grade'])):
    # Set up testing fold
    all_df.loc[test_idx, 'fold'] = i


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
        if idx > len(self.df):
            idx = len(self.df) - 1
        img_dir_path = os.path.join(data_folder, self.df.iloc[idx]['image_id'])
        image_tensors = []
        for patch in glob.glob(img_dir_path + "/*"):
            curr_img = cv2.imread(patch)
            curr_img_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            # Transpose from 1, H, W, C to 1, C, H, W
            curr_img_tensor = curr_img_rgb.transpose(2,0,1)
            image_tensors.append(curr_img_tensor)

        rows = int(np.sqrt(self.n_tiles))
        # C x H x W
        imgs = np.zeros((3, self.image_size * rows, self.image_size * rows))
        for h in range(rows):
            for w in range(rows):
                curr_idx = h * rows + w
                curr_img = image_tensors[curr_idx]
                if self.transform is not None:
                   curr_img = self.transform(image=curr_img)['image']
                h_scaled = h * self.image_size
                w_scaled = w * self.image_size
                imgs[:, h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size] = curr_img

        #if self.transform is not None:
        #   imgs = self.transform(image=imgs)['image']
        imgs = imgs.astype(np.float64)
        #imgs /= 255

        return torch.tensor(imgs), torch.tensor(self.df.iloc[idx]["isup_grade"])

# Create transforms of patches
transforms_train = albumentations.Compose([
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(scale_limit=0.15, rotate_limit=20, p=0.5)
])

# Load tiles into train/test dataloaders
df_train = all_df.loc[np.where((all_df["fold"] != 0))[0]]
df_test = all_df.loc[np.where((all_df["fold"] == 0))[0]]

# Valid = 0.3 * 0.2 = 0.06%
df_valid = df_test.sample(frac=0.3, random_state=42)
df_test = df_test.drop(df_valid.index)

train_loader = PandaDataset(df_train, 256, 36, transforms=transforms_train)
valid_loader = PandaDataset(df_valid, 256, 36, transforms=None)
test_loader = PandaDataset(df_test, 256, 36, transforms=None)

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=2, sampler=RandomSampler(train_loader),
                                            num_workers = 8)
valid_loader = torch.utils.data.DataLoader(valid_loader, batch_size=1, sampler=RandomSampler(valid_loader),
                                            num_workers = 8)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=16, sampler=RandomSampler(test_loader),
                                            num_workers = 8)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of valid batches: {len(valid_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# Current Model
resnet = torchvision.models.resnet50(pretrained=True).to(device)
num_features = resnet.fc.in_features
# 6 grading classes
num_classes = 6
resnet.fc = nn.Linear(num_features, num_classes).to(device)

# Current model to be visualized
model_weights_path = "/home/ubuntu/cs231nFinalProject/src/cnn_baseline/resnet_final_last_params.pth"
resnet.load_state_dict(torch.load(model_weights_path))

final_conv = resnet.layer4[2]._modules.get('conv3')
fc_params = list(resnet._modules.get('fc').parameters())

# Code adapted from Grad-CAM implementation
class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, C, H, W = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((C, H*W)))
    cam = cam.reshape(H, W)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

def plotGradCAM(model, final_conv, fc_params, train_loader, rows=2, cols=2, img_size=256, device=device, original=True):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)
    # save weight from fc
    fc_weight = np.squeeze(fc_params[0].cpu().data.numpy())
    # Plot original image
    fig = plt.figure(figsize=(20, 15))
    for i, (img, target) in enumerate(test_loader):
        output = model(img.to(device))
        pred_idx = output.to('cpu').numpy().argmax(1)
        cur_images = org_img.numpy().transpose((0, 2, 3, 1))
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        #plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
        ax.set_title('Original grading:%d, Grading prediction:%d' % (target, pred_idx), fontsize=14)

        if i == col - 1:
           break 
