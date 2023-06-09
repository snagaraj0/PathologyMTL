import numpy as np
import pandas as pd
import glob
DATA_PATH = "/home/ubuntu/cs231nFinalProject/panda_data/images.csv"
df = pd.read_csv(DATA_PATH)
file_names = []
for file_name in glob.glob("/home/ubuntu/cs231nFinalProject/panda_data/all_images/*"):
    idx_file = file_name.split("/")[-1].split(".")[0]
    file_names.append(idx_file)

count = 0
for idx, row in df.iterrows():
    if row["image_id"] not in file_names:
        count += 1
        df.drop(idx, inplace=True)
print(len(df))
print(count)
df.to_csv("/home/ubuntu/cs231nFinalProject/panda_data/filtered_images.csv")
