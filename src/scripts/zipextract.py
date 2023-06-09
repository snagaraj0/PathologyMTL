import pandas as pd
import zipfile

csv_file = "/home/ubuntu/cs231nFinalProject/panda_data/filtered_image_patches.csv"
zip_file = "/home/ubuntu/cs231nFinalProject/panda_data/prostate-cancer-grade-assessment.zip"
output_dir = "/home/ubuntu/cs231nFinalProject/panda_data"

df = pd.read_csv(csv_file)
filenames = list(df["image_id"])

filenames = [f"train_images/{filename}.tiff" for filename in filenames] 
# Extract only the files present in the CSV from the zip archive
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    for name in zip_ref.namelist():
        if name in filenames:
            zip_ref.extract(name, output_dir)

