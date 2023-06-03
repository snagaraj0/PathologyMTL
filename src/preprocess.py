import numpy as np
import os
import sys
import glob
import boto3
import multiprocessing
# Make module accessible
sys.path.append('/home/ubuntu/softwares/WSITools')
from wsitools.tissue_detection.tissue_detector import TissueDetector 
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor



def create_patches(input_paths: list, output_dir):
    num_processors = 200                     # Number of processes that can be running at once
    #wsi_fn = "/path/2/file.tff"             # Define a sample image that can be read by OpenSlide

    # Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
    # tissue.
    parameters = ExtractorParameters(output_dir, # Where the patches should be extracted to
        save_format = '.png',                      # Can be '.jpg', '.png', or '.tfrecord'
        sample_cnt = -1,                           # Limit the number of patches to extract (-1 == all patches)
        patch_size = 256,                          # Size of patches to extract (Height & Width)
        rescale_rate = 128,                        # Fold size to scale the thumbnail to (for faster processing)
        patch_filter_by_area = 0.5,                # Amount of tissue that should be present in a patch
        with_anno = True,                          # If true, you need to supply an additional XML file
        extract_layer = 0                          # OpenSlide Level
    )

    # Choose a method for detecting tissue in thumbnail image
    tissue_detector = TissueDetector("LAB_Threshold",   # Can be LAB_Threshold or GNB
        threshold = 85,                                   # Number from 1-255, anything less than this number means there is tissue
        training_files = None                             # Training file for GNB-based detection
    )

    # Create the extractor object
    patch_extractor = PatchExtractor(tissue_detector,
        parameters,
        feature_map = None,                       # See note below
        annotations = None                        # Object of Annotation Class (see other note below)
    )

    # Run the extraction process
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes = num_processors)
    pool.map(patch_extractor.extract, input_paths)

if __name__ == '__main__':
   data_folder = "/home/ubuntu/data/images"
   tiffs = []
   for file in glob.glob(data_folder + "/*"):
       tiffs.append(file)

   output_dir = "/home/ubuntu/data/image_patches"    # Define an output directory
   create_patches(tiffs, output_dir)
