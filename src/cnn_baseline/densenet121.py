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
df_valid = df_test.sample(frac=0.3)
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

densenet = torchvision.models.densenet121(pretrained=True).to(device)
num_features = densenet.classifier.in_features
# 6 grading classes
num_classes = 6 
densenet.classifier = nn.Linear(num_features, num_classes).to(device)


# Train just last dense block
for name, param in densenet.named_parameters():
    print(name)
    param.requires_grad = False

for param in densenet.features.denseblock4.parameters():
    param.requires_grad = True

loss_func = nn.CrossEntropyLoss()

#HYPERPARAMS
LR = 0.0035
EPOCHS = 10
opt = optim.Adam(densenet.parameters(), LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)

def train_loop(train_loader, valid_loader, opt):
    densenet.train()
    train_loss = []
    for epochs in tqdm(range(EPOCHS)):
        print(f"Training Epoch {epochs}")
        preds, targets = [], []
        for (data, label) in tqdm(train_loader):
            data, label = data.to(device, dtype=torch.float), label.type(torch.LongTensor).to(device)
            opt.zero_grad()
            logits = densenet(data)
            loss = loss_func(logits, label)

            loss.backward()
            opt.step()
            train_loss.append(loss.detach().cpu().numpy())
        lr_scheduler.step(loss.item())

        # Calculate val accuracy after each epoch
        best_acc = 0.0
        for (data, label) in tqdm(valid_loader):
            data, label = data.to(device, dtype=torch.float), label.type(torch.LongTensor).to(device)
            val_logits = densenet(data)
            _, pred = torch.max(val_logits.data, axis=1)
            preds.append(pred)
            targets.append(label)
        preds = torch.cat(preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        val_acc = (preds == targets).mean() * 100
        print(f"Epoch [{epochs + 1}/{EPOCHS}], Loss: {loss.item()}, Validation accuracy: {val_acc}%")
        if best_acc < val_acc:
            torch.save(densenet.state_dict(), os.path.join(f"/home/ubuntu/cs231nFinalProject/src/cnn_baseline/densenet_params_last_{epochs+1}.pth"))
            best_acc = val_acc  

    # Save params after training
    torch.save(densenet.state_dict(), os.path.join("/home/ubuntu/cs231nFinalProject/src/cnn_baseline/densenet_final_last_params.pth"))
    return train_loss

def test_loop(loader):
    test_loss = []
    preds, targets = [], []
    correct = 0
    densenet.eval()
    with torch.no_grad():
        for (data, label) in tqdm(loader):
            data, label = data.to(device, dtype=torch.float), label.type(torch.LongTensor).to(device)
            logits = densenet(data)
            loss = loss_func(logits, label)
            _, pred = torch.max(logits.data, axis=1)
            preds.append(pred)
            targets.append(label)
            
            test_loss.append(loss.detach().cpu().numpy())
        final_test_loss = np.mean(test_loss)

    preds = torch.cat(preds).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    test_acc = (preds == targets).mean() * 100
    return final_test_loss, test_acc

#Training
train_loss = train_loop(train_loader, valid_loader, opt)

#Evaluation

#densenet = torchvision.models.densenet121(pretrained=True).to(device)
#num_features = densenet.classifier.in_features
# 6 grading classes
#num_classes = 6
#densenet.classifier = nn.Linear(num_features, num_classes).to(device)
#densenet.load_state_dict(torch.load(os.path.join(f"/home/ubuntu/cs231nFinalProject/src/cnn_baseline/densenet_final_params.pth")))
#densenet.eval()

test_loss, test_acc = test_loop(test_loader)
print(f"Final test set classification accuracy: {test_acc}%") 
