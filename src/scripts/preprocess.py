import os
import cv2
import skimage.io
from tqdm import *
from tqdm.contrib.concurrent import process_map
import zipfile
import numpy as np
from multiprocessing import Pool
import pandas as pd

# Create patches from WSI image
#If > 95% of tile is white, flag that tile
def create_patches(name):
    imgs = "/home/ubuntu/cs231nFinalProject/panda_data/all_images"
    masks = "/home/ubuntu/cs231nFinalProject/panda_data/train_label_masks"

    out_patch_dir = "/home/ubuntu/cs231nFinalProject/panda_data/image_patches"
    out_mask_dir = "/home/ubuntu/cs231nFinalProject/panda_data/mask_patches"
    
    try:
        img = skimage.io.MultiImage(os.path.join(imgs, name + '.tiff'))[-1]
    except IndexError:
        print(f"Error with {os.path.join(imgs, name + '.tiff')}")
        return
    else:
        img = skimage.io.MultiImage(os.path.join(imgs, name + '.tiff'))[-1]
    

    try:
        mask = skimage.io.MultiImage(os.path.join(masks, name + '_mask.tiff'))[-1]
    except IndexError:
        print(f"Error with {os.path.join(masks, name + '_mask.tiff')}")
        return
    else:
        mask = skimage.io.MultiImage(os.path.join(masks, name + '_mask.tiff'))[-1]
    patch_size, image_size = 256, 256
    n_patches = 36

    patches = []
    H, W, _ = img.shape
    #H, W, _ = mask.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # Pad with white
    padded_img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0,0]], constant_values=255)
    padded_mask = np.pad(mask, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0,0]], constant_values=0)

    padded_img = padded_img.reshape(padded_img.shape[0] // patch_size, # number of tiles in h-dimension
            patch_size,
            padded_img.shape[1] // patch_size, # number of tiles in width dimension
            patch_size,
            3)
    padded_img = padded_img.transpose(0,2,1,3,4).reshape(-1, patch_size, patch_size, 3)
    
    padded_mask = padded_mask.reshape(padded_mask.shape[0] // patch_size, # number of tiles in h-dimension
            patch_size,
            padded_mask.shape[1] // patch_size, # number of tiles in width dimension
            patch_size,
            3)
    padded_mask = padded_mask.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, 3)
    if len(padded_img) < n_patches:
        padded_img = np.pad(padded_img, [[0, n_tiles - len(padded_img)], [0,0], [0,0], [0,0]], constant_values=255)
        padded_mask = np.pad(padded_mask, [[0, n_tiles - len(padded_mask)], [0, 0], [0,0], [0,0]], constant_values=0)
    # select most representative tiles
    indices = np.argsort(padded_img.reshape(padded_img.shape[0],-1).sum(-1))[:n_patches]
    #filtered_img = padded_img[indices]
    filtered_mask = padded_mask[indices]
    for i in range(len(filtered_mask)):
        #patches.append({'img_patches': filtered_img[i], 'mask_patches': filtered_mask[i], 'idx': i})
        patches.append({'mask_patches': filtered_mask[i], 'idx': i})
    os.makedirs(os.path.join(out_patch_dir, name), exist_ok=True)
    os.makedirs(os.path.join(out_mask_dir, name), exist_ok=True)
    for tile in patches:
        #tiled_img, tiled_mask, idx = tile['img_patches'], tile['mask_patches'], tile["idx"]
        tiled_mask, idx = tile['mask_patches'], tile["idx"]
        #out_img = open(os.path.join(out_patch_dir, name, f'{name}_{idx}.png'), 'wb')
        out_mask = open(os.path.join(out_mask_dir, name,  f'{name}_{idx}.png'), 'wb')
        
        #write_img = cv2.imencode('.png', cv2.cvtColor(tiled_img, cv2.COLOR_RGB2BGR))[1]
        #out_img.writestr(f'{name}_{idx}.png', write_img)
        #out_img.write(write_img)
        write_mask = cv2.imencode('.png', tiled_mask[:, :, 0])[1]
        #out_mask.writestr(f'{name}_{idx}.png', write_mask)
        out_mask.write(write_mask)
 
if __name__ == '__main__':
   #imgs = "/home/ubuntu/cs231nFinalProject/panda_data/all_images"
   #image_names = [name.split(".")[0] for name in os.listdir(imgs)]
   df = pd.read_csv("/home/ubuntu/cs231nFinalProject/panda_data/filtered_image_patches.csv")
   image_names = list(df["image_id"])
   pool = Pool(os.cpu_count())
   with pool as p:
        with tqdm(total=len(image_names)) as pbar:
             for _ in p.imap_unordered(create_patches, image_names):
                 pbar.update()
