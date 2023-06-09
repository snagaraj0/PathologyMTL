import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys

sys.path.append("/home/ubuntu/cs231nFinalProject/src/UNet")
from UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UpBlockForUNetWithDense121(nn.Module):

    def __init__(self, up, conv):
        super().__init__()
        self.upsample = up
        self.conv_block = conv

    def forward(self, x, bridge):
        x = self.upsample(x)
        x = torch.cat([x, bridge], 1)
        x = self.conv_block(x)
        return x

class MTLWithDensenet121Encoder(nn.Module):

    def __init__(self, num_classes=6, weights_path="/home/ubuntu/cs231nFinalProject/src/cnn_baseline/densenet_final_last_params.pth",
            unet_path = "/home/ubuntu/cs231nFinalProject/src/UNet/custom_unet_model_epoch_4.pth", embed_dim = 768):
        super().__init__()
        
        # Get models
        densenet = models.densenet121(pretrained=True).to(device)
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_features, num_classes).to(device)
        densenet.load_state_dict(torch.load(weights_path))

        unet = UNet().to(device)
        unet.load_state_dict(torch.load(unet_path))

        # Remove Final Linear Layer
        components = list(densenet.children())[:-1]
        
        # Define Input Block and Pool
        self.input_block = nn.Sequential(*list(components[0].children())[:3])
        self.input_pool = list(components[0].children())[3]
        
        # Get Blocks 
        down_blocks = []
        for item in list(components[0].children())[4:]:
            down_blocks.append(item)
        
        # Create Down
        self.down_blocks = nn.ModuleList(down_blocks)

        # Initialize Upsampling Convs for Bridge
        self.dense0up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dense1up = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        
        # Create 1 x 1 Conv for Densenet output to Unet
        self.dout = nn.Conv2d(1024, 2048, kernel_size=1, stride=1)

        # Create Classification Decoder
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.class_head_fc1 = nn.Linear(1024, embed_dim)
        self.class_head_fc2 = nn.Linear(embed_dim, num_classes)

        # Create Segmentation Decoder
        unet_up = unet.up_path
        segmentation = []

        for block in unet_up[1:]:
            segmentation.append(UpBlockForUNetWithDense121(block.up, block.conv_block))

        self.up_blocks = nn.ModuleList(segmentation)
        self.out = unet.last

    def forward(self, x, with_output_feature_map=False):
        # Track bridging elements
        bridges = dict()

        # First Block
        x = self.input_block(x)
        bridges[0] = self.dense0up(x)

        # Second Block
        x = self.input_pool(x)
        bridges[1] = self.dense1up(x)

        # Rest of the Boxes
        idx = 2
        for block in self.down_blocks:
            x = block(x)
            if isinstance(block, models.densenet._DenseBlock):
                bridges[idx] = x
                idx += 1
        bridges[idx - 1] = x

        x_classification = torch.squeeze(self.global_pool(x), dim=(2, 3))
        x_classification = F.leaky_relu(self.class_head_fc1(x_classification))
        classification_pred = self.class_head_fc2(x_classification) 

        # Run bridging between down and up blocks
        x = self.dout(x)
        for i, block in enumerate(self.up_blocks):
            x = block(x, bridges[4 - i])
        segmentation_pred = self.out(x)
        del bridges

        return classification_pred, segmentation_pred


