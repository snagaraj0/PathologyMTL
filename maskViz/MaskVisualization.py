import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os
import numpy as np
import skimage.io as io

# Generate mask:  0 - Background  |  1 - Class 1  |  2 - Class 2, and so on.

Image.MAX_IMAGE_PIXELS = None
img = np.asarray(Image.open("../Downloads/003046e27c8ead3e3db155780dc5498e_mask.tiff"))
plt.savefig(img)