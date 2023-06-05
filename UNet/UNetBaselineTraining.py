from UNet import UNet
from UN

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

BASE_PATH = "/home/ubuntu/cs231nFinalProject/panda_data"
data_folder = os.path.join(BASE_PATH, "image_patches")
masks_path = os.path.join(BASE_PATH, "mask_patches")

all_df = pd.read_csv(os.path.join(BASE_PATH, "filtered_image_patches.csv"))

print(f"Size of all images: {len(all_df)}")

images = list(all_df['image_id'])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_df['fold'] = -1
for i, (train_idx, test_idx) in enumerate(skf.split(all_df, all_df['isup_grade'])):
    # Set up testing fold
    all_df.loc[test_idx, 'fold'] = i

# Hyperparams
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 36
num_epochs = 5
num_workers = 8
i_width = 256 * 6
i_height = 256 * 6
pin_memory = True
load_model = False

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Forward
        preds = model(data)
        loss = loss_fn(preds, targets)

        # Backward
        optimizer.zero_grad()
        optimizer.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimzer = optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    main()