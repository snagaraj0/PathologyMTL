import os
import numpy as np
import skimage.io as io
from tqdm import tqdm

directory1 = "../panda_data/all_images"
directory = "../panda_data/images.csv"

for filename in tqdm(os.listdir(directory2)):
    # id, ftype = tuple(filename.split("."))
    # id += "_mask"
    # filename = id + "." + ftype
    f = os.path.join(directory2, filename)

    try:
        image = io.imread(f)
    except FileNotFoundError:
        continue

    if np.sum(image) != 0:
        image = np.asarray(image)
        image = image.transpose((2, 0, 1))
        print(np.max(image[0]))
        print(np.max(image[1]))
        print(np.max(image[2]))
        
