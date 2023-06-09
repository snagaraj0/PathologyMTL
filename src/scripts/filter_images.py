import numpy as np
import pandas as pd
import os

DATA_PATH = "/home/ubuntu/cs231nFinalProject/panda_data/images.csv"
df = pd.read_csv(DATA_PATH)

file_names = [name for name in os.listdir("/home/ubuntu/cs231nFinalProject/panda_data/image_patches")]

print(f"Number of processed tiffs: {len(file_names)}")

filtered_files = []
count = 0
for file_name in file_names:
    num_patches = [name for name in os.listdir(os.path.join("/home/ubuntu/cs231nFinalProject/panda_data/image_patches", file_name))]
    if len(num_patches) < 36:
       count += 1
    else:
        filtered_files.append(file_name)

print(f"Number of files found: {len(filtered_files)}")
count = 0
for idx, row in df.iterrows():
    if row["image_id"] not in filtered_files:
        count += 1
        df.drop(idx, inplace=True)

df.to_csv("/home/ubuntu/cs231nFinalProject/panda_data/filtered_image_patches.csv")



