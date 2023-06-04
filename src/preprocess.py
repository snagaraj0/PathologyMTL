import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np

imgs = "/home/ubuntu/cs231nFinalProject/panda_data/all_images/"
masks = "/home/ubuntu/cs231nFinalProject/panda_data/train_label_masks/"

out_patch_zip = "/home/ubuntu/cs231nFinalProject/panda_data/image_patches.zip"
out_mask_zip = "/home/ubuntu/cs231nFinalProject/panda_data/mask_patches.zip"

# Create patches from WSI image
#If > 95% of tile is white, flag that tile
def create_patches(img, mask):
    patch_size, image_size = 256, 256
    n_patches = 36
    
    
    patches = []
    H, W, _ = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # Pad with white
    padded_img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0,0]], constant_values=255)
    padded_mask = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0,0]], constant_values=0)

    padded_img = padded_img.reshape(padded_img.shape[0] // patch_size, # number of tiles in h-dimension
            patch_size,
            padded_img.shape[1] // patch_size, # number of tiles in width dimension
            patch_size,
            3)
    padded_img = padded_img.transpose(0,2,1,3,4).reshape(-1, patch_size, patch_size, 3)
    
    if len(padded_img) < n_patches:
        padded_img = np.pad(padded_img, [[0, n_tiles - len(padded_img)], [0,0], [0,0], [0,0]], constant_values=255)
    # select most representative tiles
    indices = np.argsort(padded_img.reshape(padded_img.shape[0],-1).sum(-1))[:n_patches]
    filtered_img = padded_img[indices]
    for i in range(len(filtered_img)):
        patches.append({'patches': filtered_img[i], 'idx': i})
    return patches

