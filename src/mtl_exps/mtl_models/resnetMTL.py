
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys

sys.path.append("/home/ubuntu/cs231nFinalProject/src/UNet")
from UNet import UNet
device = torch.device('cuda')

class UpBlockForUNetWithResNet50(nn.Module):

    def __init__(self, up, conv):
        super().__init__()
        self.upsample = up
        self.conv_block = conv

    def forward(self, x, bridge):
        x = self.upsample(x)
        x = torch.cat([x, bridge], 1)
        x = self.conv_block(x)
        return x

class MTLWithResnetEncoder(nn.Module):
    def __init__(self, num_classes=6, weights_path="/home/ubuntu/cs231nFinalProject/src/cnn_baseline/resnet_final_last_params.pth",
            unet_path = "/home/ubuntu/cs231nFinalProject/src/UNet/custom_unet_model_epoch_4.pth", embed_dim = 768):
        super().__init__()
        
        # Get models
        resnet = models.resnet50(pretrained=True).to(device)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes).to(device)
        resnet.load_state_dict(torch.load(weights_path))

        unet = UNet().to(device)
        unet.load_state_dict(torch.load(unet_path))

        # Define Input Block and Pool
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]

        # Get Blocks
        down_blocks = []
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)

        # Create Down
        self.down_blocks = nn.ModuleList(down_blocks)

        # Initialize Upsampling Convs for Bridge
        self.res0up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.res1up = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)

        # Create Classification Decoder
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.class_head_fc1 = nn.Linear(2048, embed_dim)
        self.class_head_fc2 = nn.Linear(embed_dim, num_classes)

        # Create Segmentation Decoder
        unet_up = unet.up_path
        segementation = []

        for block in unet_up[1:]:
            segementation.append(UpBlockForUNetWithResNet50(block.up, block.conv_block))

        self.up_blocks = nn.ModuleList(segementation)

        self.out = unet.last

    def forward(self, x, with_output_feature_map=False):
        # Track
        bridges = dict()

        # First Block
        x = self.input_block(x)
        bridges[0] = self.res0up(x)

        # Second Block
        x = self.input_pool(x)
        bridges[1] = self.res1up(x)

        # Rest of the Boxes
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            bridges[i] = x

        # Classification
        x_classification = torch.squeeze(self.global_pool(x), dim=(2, 3))
        x_classification = F.relu(self.class_head_fc1(x_classification))
        classification_pred = self.class_head_fc2(x_classification)

        # Segmentation
        for i, block in enumerate(self.up_blocks):
            x = block(x, bridges[4 - i])
        segmentation_pred = self.out(x)
        del bridges

        return classification_pred, segmentation_pred
