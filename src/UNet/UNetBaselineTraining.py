from UNet import UNet

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
import pandas as pd
import matplotlib.pyplot as plt
import time

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

# DATA LOADING
print("Starting Data Loading")

# Set Paths
BASE_PATH = "/home/ubuntu/cs231nFinalProject/panda_data"
data_folder = os.path.join(BASE_PATH, "image_patches")
masks_path = os.path.join(BASE_PATH, "mask_patches")

# Create Data Frame
all_df = pd.read_csv(os.path.join(BASE_PATH, "filtered_image_patches.csv"))
images = list(all_df['image_id'])

# Set Train/Test Split
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_df['fold'] = -1
for i, (train_idx, test_idx) in enumerate(skf.split(all_df, all_df['isup_grade'])):
    # Set up testing fold
    all_df.loc[test_idx, 'fold'] = i

print("Train/Test Split Done")

# Create Class
class UNetDataLoader(Dataset):
    def __init__(self, df, image_size, n_tiles=36, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.transform = transforms

    def __len__(self):
        return len(self.df)
    '''
    # Custom get item function
    def __getitem__(self, idx):
        if idx > len(self.df):
            idx = len(self.df) - 1

        # Get Image Patches
        img_dir_path = os.path.join(data_folder, self.df.iloc[idx]['image_id'])
        image_tensors = []
        for patch in glob.glob(img_dir_path + "/*"):
            curr_img = cv2.imread(patch)
            curr_img_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            # Transpose from 1, H, W, C to 1, C, H, W
            curr_img_tensor = curr_img_rgb
            # if curr_img_tensor.shape != (3, 256, 256):
            curr_img_tensor = curr_img_rgb.transpose(2,0,1)
            # print(np.asarray(curr_img_tensor).shape)
            # assert curr_img_tensor.shape == (3, 256, 256), print(curr_img_tensor.shape)
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
                # print(imgs[:, h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size].shape)
            
                imgs[:, h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size] = curr_img

        imgs = imgs.astype(np.float64)
        image_tensors = []

        # Get Mask Patches
        mask_dir_path = os.path.join(masks_path, self.df.iloc[idx]['image_id'])
        mask_tensors = []
        for patch in glob.glob(mask_dir_path + "/*"):
            curr_mask = cv2.imread(patch)
            curr_mask_rgb = cv2.cvtColor(curr_mask, cv2.COLOR_BGR2RGB)
            # Transpose from H, W, C to C, H, W
            curr_mask_tensor = curr_mask_rgb.transpose(2,0,1)
            # Get Channel 1 -> H, W
            curr_mask_tensor = torch.tensor(curr_img_tensor)[0].squeeze(0)
            # Transform to scale 0, 1, 2
            if self.df.iloc[idx]["data_provider"] == "radboud":
                curr_mask_tensor = torch.where(curr_mask_tensor <= 1, curr_mask_tensor, torch.where(curr_mask_tensor <= 2, 1, 2))
            # Add to list
            mask_tensors.append(curr_mask_tensor)
        
        # H x W
        masks = torch.zeros((self.image_size * rows, self.image_size * rows))
        for h in range(rows):
            for w in range(rows):
                curr_idx = h * rows + w
                curr_mask = mask_tensors[curr_idx]
                h_scaled = h * self.image_size
                w_scaled = w * self.image_size
                masks[h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size] = curr_mask
        
        return torch.tensor(imgs), masks
    '''
    # Custom get item function
    def __getitem__(self, idx):
        if idx > len(self.df):
            idx = len(self.df) - 1
        
        # print("Get Item Called")
        # Get Image and Mask Directories
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

train_loader = UNetDataLoader(df_train, 256, 36, transforms=transforms_train)
valid_loader = UNetDataLoader(df_valid, 256, 36, transforms=None)
test_loader = UNetDataLoader(df_test, 256, 36, transforms=None)

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, sampler=RandomSampler(train_loader),
                                            num_workers = 8)
valid_loader = torch.utils.data.DataLoader(valid_loader, batch_size=1, sampler=RandomSampler(valid_loader),
                                            num_workers = 8)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=1, sampler=RandomSampler(test_loader),
                                            num_workers = 8)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of valid batches: {len(valid_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# TRAINING
print("Done Loading, Starting Training")

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda')

# Set hyperparameters
batch_size = 16
learning_rate = 0.0001
num_epochs = 10

# Initialize Model
model = UNet().to(device)
if os.path.exists('custom_unet_model_epoch_4.pth'):
    # Load saved model parameters
    model.load_state_dict(torch.load('custom_unet_model_epoch_4.pth'))
    print('Loaded saved model parameters')

# Intialize Loss and Optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)


# Function for Dice Score and IOU
def calculate_dice_iou_score(predictions, mask, num_classes=3):
    dice_scores = []
    iou_scores = []

    for class_id in range(num_classes):
        # Flatten the predictions and mask tensors for the specific class
        predictions_flat = (predictions == class_id).view(-1)
        mask_flat = (mask == class_id).view(-1)

        # Compute intersection and union
        intersection = (predictions_flat & mask_flat).sum().item()
        union = (predictions_flat | mask_flat).sum().item()

        # Calculate Dice score for the class
        dice = (2.0 * intersection) / (union + intersection + 1e-8)
        dice_scores.append(dice)

        # Calculate IOU score for the class
        iou = intersection / (union + 1e-8)
        iou_scores.append(iou)

    return np.mean(dice_scores), np.mean(iou_scores)

# Check Accuracy Function
def check_accuracy(loader, model, loss_fn=None):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    loss = 0
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            # To Device
            if torch.cuda.is_available():
                images = images.type(torch.float).to(device)
                masks = masks.type(torch.long).to(device)

            # Predictions
            preds = model(images)

            # Calculate Loss for Validation
            if loss_fn is not None:
                loss += loss_fn(preds, masks)

            # Convert to Classes
            preds = torch.argmax(preds, dim=1).squeeze(1).type(torch.long)

            # General Accuracy
            num_correct += (masks == preds).sum()
            num_pixels += torch.numel(preds)

            # Dice & IOU Score
            dice, iou = calculate_dice_iou_score(preds, masks, num_classes=3)
            dice_score += dice
            iou_score += iou

    accuracy = float(num_correct / num_pixels)
    dice_score /= len(loader)
    iou_score /= len(loader)
    loss /= len(loader)

    return accuracy, dice_score, iou_score, loss

# Train Loop
def train(num_epochs, train_loader, model, loss_fn, optimizer, log):
    # Training loop
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        train_loss = 0
        train_loader = tqdm(train_loader)
        for i, (images, masks) in enumerate(train_loader):
            # Move images and labels to device
            if torch.cuda.is_available():
                images = images.type(torch.float).to(device)
                masks = masks.type(torch.long).to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            train_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            train_loader.set_postfix({'Batch Loss': loss.item()})
            #if (i + 1) % 10 == 0:
             #   print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')

        # Save the trained model
        torch.save(model.state_dict(), f'custom_unet_model_epoch_{epoch+1}.pth')
       
        # Schedule learning rate
        lr_scheduler.step(loss.item())

        # Check Accuracy after each epoch
        train_loss /= total_step
        train_accuracy, train_dice, train_iou, _ = check_accuracy(train_loader, model)
        valid_accuracy, valid_dice, valid_iou, valid_loss = check_accuracy(valid_loader, model, loss_fn)
      
        # Create strings
        ep = "Epoch: " + str(epoch) + "\n"
        train_out = f"Training: Loss: {train_loss}, Accuracy: {train_accuracy}, Dice: {train_dice}, IOU: {train_iou}\n"
        valid_out = f"Valiation: Loss: {valid_loss}, Accuracy: {valid_accuracy}, Dice: {valid_dice}, IOU: {valid_iou}\n"
        
        # Print values
        print(train_out[:-2])
        print(valid_out[:-2])
        
        # Log values
        log.write(ep)
        log.write(train_out)
        log.write(valid_out)
        
# Create Logging File
log = open('UNetlogTest.txt', 'w')

# Train
# train(num_epochs, train_loader, model, loss, optimizer, log)

#Evaluate
test_accuracy, test_dice, test_iou, test_loss = check_accuracy(test_loader, model, loss)

log.write("Final Test: ")
test_out = f"Test: Loss: {test_loss}, Accuracy: {test_accuracy}, Dice: {test_dice}, IOU: {test_iou}\n"
print(test_out[:-2])
log.write(test_out)
