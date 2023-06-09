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
        mask_dir_path = os.path.join(masks_path, self.df.iloc[idx]['image_id'])
        image_tensors, mask_tensors = [], []
        
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
            if curr_mask is None:
                curr_mask = np.zeros((1, self.image_size, self.image_size)).astype(np.uint8)
            curr_mask = curr_mask.transpose(2, 0, 1)
            curr_mask = np.max(curr_mask, axis=0)

            # Apply Augmentations
            if self.transform is not None:
                   augmentations = self.transform(image=curr_img, mask=curr_mask)
                   curr_img = augmentations["image"]
                   curr_mask = augmentations["mask"]

            image_tensors.append(curr_img)
            mask_tensors.append(curr_mask)


        rows = int(np.sqrt(self.n_tiles))
        # C x H x W
        imgs = np.zeros((3, self.image_size * rows, self.image_size * rows))
        # H x W
        masks = np.zeros((self.image_size * rows, self.image_size * rows))
        for h in range(rows):
            for w in range(rows):
                curr_idx = h * rows + w
                curr_img = image_tensors[curr_idx]
                curr_mask = None
                if len(mask_tensors) < curr_idx:
                   curr_mask = np.zeros((self.image_size, self.image_size)).astype(np.uint8)
                else:
                   curr_mask = mask_tensors[curr_idx]
                h_scaled = h * self.image_size
                w_scaled = w * self.image_size
                imgs[:, h_scaled : h_scaled + self.image_size, w_scaled : w_scaled + self.image_size] = curr_img
                masks[h_scaled: h_scaled + self.image_size, w_scaled: w_scaled + self.image_size] = curr_mask
                
        imgs = imgs.astype(np.float64)
        imgs /= 255
        #masks /= 255
        masks = torch.tensor(masks, dtype=torch.float64)

        # Rescale mask if needed
        if self.df.iloc[idx]["data_provider"] == "radboud":
            masks = torch.where(masks <= 1, masks, torch.where(masks <= 2, 1, 2)) 
        return torch.tensor(imgs), masks, torch.tensor(self.df.iloc[idx]["isup_grade"])

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

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, sampler=RandomSampler(train_loader),
                                            num_workers = 6)
valid_loader = torch.utils.data.DataLoader(valid_loader, batch_size=1, sampler=RandomSampler(valid_loader),
                                            num_workers = 6)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=1, sampler=RandomSampler(test_loader),
                                            num_workers = 6)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of valid batches: {len(valid_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# Multi-Task Learning Model
mtl = MTLHIPT().to(device)

# Freeze encoder
'''
for name, param in mtl.named_parameters():
    if name.startswith('hipt'):
        param.requires_grad = False
'''
# Train full architecture
#for param in mtl.parameters():
#    param.requires_grad = False


def dice_loss(preds, mask, num_classes=3):
    dice_losses = []
    preds = torch.argmax(preds, dim=1).squeeze(1).type(torch.long)
    for class_id in range(num_classes):
        predictions_flat = (preds == class_id).view(-1)
        mask_flat = (mask == class_id).view(-1)

        # Compute intersection and union
        intersection = (predictions_flat & mask_flat).sum().item()
        union = (predictions_flat | mask_flat).sum().item()

        # Calculate Dice score for the class
        dice = (2.0 * intersection) / (union + intersection + 1e-8)
        dice_losses.append(1. - dice)

    return np.mean(dice_losses)


class_loss_func = nn.CrossEntropyLoss()
segment_loss_func = nn.CrossEntropyLoss()

#HYPERPARAMS
LR = 1e-4
EPOCHS = 20
opt = optim.Adam(mtl.parameters(), LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-06)
log = open("hipt_train_log.txt", "w")

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

def train_loop(train_loader, valid_loader, opt, log):
    mtl.train()
    train_loss = []
    for epochs in tqdm(range(EPOCHS)):
        print(f"Training Epoch {epochs + 1}")
        class_preds, segment_preds, targets = [], [], []
        num_pixels, num_correct, dice_score, iou_score = 0, 0, 0.0, 0.0
        for (data, masks, label) in tqdm(train_loader):
            if torch.cuda.is_available():
                data, masks, label = data.to(device, dtype=torch.float), masks.to(device, dtype=torch.long), label.type(torch.LongTensor).to(device)
            opt.zero_grad()

            # with torch.cuda.amp.autocast():
            class_logits, mask_logits = mtl(data)
            mask_logits += 1e-6
            class_logits += 1e-6
            # MTL loss is sum of losses
            loss = class_loss_func(class_logits, label) + segment_loss_func(mask_logits, masks) + 1e-6
            #loss = class_loss_func(class_logits, label) + dice_loss(mask_logits, masks)
            _, class_pred = torch.max(class_logits.data, axis=1)
            class_preds.append(class_pred)
            targets.append(label)

            # Segment evaluation
            segment_preds = torch.argmax(mask_logits, dim=1).squeeze(1).type(torch.long)
            num_correct += (masks == segment_preds).sum()
            num_pixels += torch.numel(segment_preds)

            # Dice & IOU Score
            dice, iou = calculate_dice_iou_score(segment_preds, masks, num_classes=3)
            dice_score += dice
            iou_score += iou

            train_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            opt.step()

        class_preds = torch.cat(class_preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        val_class_acc = (class_preds == targets).mean() * 100

        pixel_acc = 100 * float(num_correct / num_pixels)
        dice_score /= len(train_loader)
        iou_score /= len(train_loader)

        print(f"Epoch [{epochs + 1}/{EPOCHS}], Training Loss: {loss.item()}, Training class accuracy: {val_class_acc}%, Training pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}")
        report = f"Epoch [{epochs + 1}/{EPOCHS}], Training Loss: {loss.item()}, Training class accuracy: {val_class_acc}%, Training pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}\n"
        log.write(report)
        lr_scheduler.step(loss.item())
        
        torch.cuda.empty_cache()
        
        class_preds, segment_preds, targets = [], [], []
        # Calculate val accuracy after each epoch
        best_acc, best_dice = 0.0, 0.0
        num_pixels, num_correct, dice_score, iou_score = 0, 0, 0.0, 0.0
        for (data, masks, label) in tqdm(valid_loader):
            if torch.cuda.is_available():
                data, masks, label = data.to(device, dtype=torch.float), masks.to(device, dtype=torch.long), label.type(torch.LongTensor).to(device)
            
            with torch.cuda.amp.autocast():
                val_class_logits, val_mask_logits = mtl(data)
                val_class_logits += 1e-6
                val_mask_logits += 1e-6
            # Class evaluation
            _, class_pred = torch.max(val_class_logits.data, axis=1)
            class_preds.append(class_pred)
            targets.append(label)

            # Segment evaluation
            segment_preds = torch.argmax(val_mask_logits, dim=1).squeeze(1).type(torch.long)
            num_correct += (masks == segment_preds).sum()
            num_pixels += torch.numel(segment_preds)

            # Dice & IOU Score
            dice, iou = calculate_dice_iou_score(segment_preds, masks, num_classes=3)
            dice_score += dice
            iou_score += iou
        
        class_preds = torch.cat(class_preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        val_class_acc = (class_preds == targets).mean() * 100
        
        pixel_acc = 100 * float(num_correct / num_pixels)
        dice_score /= len(valid_loader)
        iou_score /= len(valid_loader)
        

        print(f"Epoch [{epochs + 1}/{EPOCHS}], Val class accuracy: {val_class_acc}%, Val pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}")
        report = f"Epoch [{epochs + 1}/{EPOCHS}], Training Loss: {loss.item()}, Training class accuracy: {val_class_acc}%, Training pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}\n"
        log.write(report)
        if best_acc < val_class_acc and best_dice < dice_score:
            torch.save(mtl.state_dict(), os.path.join(f"/home/ubuntu/cs231nFinalProject/src/mtl_exps/hipt__mtl4_{epochs+1}.pth"))
            best_acc = val_class_acc
            best_dice = dice_score

    # Save params after training
    torch.save(mtl.state_dict(), os.path.join("/home/ubuntu/cs231nFinalProject/src/mtl_exps/hipt_mtl4_final_params.pth"))
    return train_loss

def test_loop(loader):
    test_loss = []
    class_preds, segment_preds, targets = [], [], []
    mtl.eval()
    num_pixels, num_correct, dice_score, iou_score = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for (data, masks, label) in tqdm(loader):
            # Move images and labels to device
            if torch.cuda.is_available():
                data, masks, label = data.to(device, dtype=torch.float), masks.to(device, dtype=torch.long), label.type(torch.LongTensor).to(device)
            test_class_logits, test_mask_logits = mtl(data)
            test_class_logits += 1e-6
            test_mask_logits += 1e-6
            # MTL loss function
            loss = 1.2*class_loss_func(test_class_logits, label) + segment_loss_func(test_mask_logits, masks) + 1e-6 
            #loss = class_loss_func(test_class_logits, label) + dice_loss(test_mask_logits, masks)

            _, class_pred = torch.max(test_class_logits.data, axis=1)
            class_preds.append(class_pred)
            targets.append(label)

            # Segment evaluation
            segment_preds = torch.argmax(test_mask_logits, dim=1).squeeze(1).type(torch.long)
            num_correct += (masks == segment_preds).sum()
            num_pixels += torch.numel(segment_preds)

            # Dice & IOU Score
            dice, iou = calculate_dice_iou_score(segment_preds, masks, num_classes=3)
            dice_score += dice
            iou_score += iou

            test_loss.append(loss.detach().cpu().numpy())

        class_preds = torch.cat(class_preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        test_class_acc = (class_preds == targets).mean() * 100
        
        pixel_acc = 100 * float(num_correct / num_pixels)
        dice_score /= len(valid_loader)
        iou_score /= len(valid_loader)

        final_test_loss = np.mean(test_loss)

    return final_test_loss, test_class_acc, pixel_acc, dice_score, iou_score

#Training
train_loss = train_loop(train_loader, valid_loader, opt, log)

#Evaluation
#mtl = MTLWithDensenet121Encoder().to(device)
#mtl.load_state_dict(torch.load(os.path.join("/home/ubuntu/cs231nFinalProject/src/mtl_exps/densenet_mtl_final_params.pth")))
#mtl.eval()

test_loss, test_acc, pixel_acc, dice_score, iou_score  = test_loop(test_loader)
print(f"Final test set metrics, Class accuracy: {test_acc}%, Pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}") 
report = f"Final test set metrics, Class accuracy: {test_acc}%, Pixel accuracy: {pixel_acc}%, Dice score: {dice_score}, IOU score: {iou_score}"
log.write(report)
