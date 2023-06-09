import torch
import torch.nn as nn
import torchvision.models as models
from nt_xent import NTXentLoss
import sys
import os

sys.path.append("/home/ubuntu/cs231nFinalProject/src/UNet")
from UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def adj_matrix():

    return 


class BaseResnet(nn.Module):
    def __init__(self, n_classes=t6, weights_path="/home/ubuntu/cs231nFinalProject/src/cnn_baseline/resnet_final_last_params.pth"):
        super(BaseResnet, self).__init__()

        # Get models
        resnet = models.resnet50(pretrained=True).to(device)
        resenet.load_state_dict(torch.load(weights_path))

        # Remove Final Linear Layer
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        # projection MLP
        num_features = resnet.fc.in_features
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
    
    def forward(self, x):
        h = self.resnet_features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x

# Contrastive Learning Class
class CustomSimCLR(nn.Module):
    def __init__(self):
        super(CustomSimCLR, self).__init()
    
    def _step(self, model, x_is, x_js):
        r_is, proj_x_is = model(x_is)
        r_js, proj_r

    def train(self):
class GraphMTL(nn.Module):

    def __init__(self, n_classes=6, weights_path="/home/ubuntu/cs231nFinalProject/src/cnn_baseline/resnet_final_last_params.pth",
            unet_path = "/home/ubuntu/cs231nFinalProject/src/UNet/custom_unet_model_epoch_4.pth", embed_dim = 1024):
        super(GraphMTL, self).__init__()
        
        
        # Get Blocks 
        down_blocks = []
        for item in list(components.children())[4:]:
            down_blocks.append(item)
        
        # Create Down
        self.down_blocks = nn.ModuleList(down_blocks)

        # Initialize Upsampling Convs for Bridge
        self.dense0up = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.dense1up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Create Classification Decoder
        self.flatten = nn.Flatten()
        self.class_head_fc1 = nn.Linear(2048 * 48 * 48,embed_dim)
        self.class_head_fc2 = nn.Linear(embed_dim, num_classes)

        # Create Segmentation Decoder
        unet_up = unet.up_path
        segmentation = []

        for block in unet_up[1:]:
            segmentation.append(UpBlockForUNetWithDense121(block.up, block.conv_block))

        self.up_blocks = nn.ModuleList(segmentation)
        self.out = unet.last

    def forward(self, x):

        h = self.features(x)
        h = h.squeeze()

        x = F.relu(self.l1(h))
        x = self.l2(h)
        

        x_classification = self.flatten(x)
        x_classification = self.class_head_fc1(x_classification)
        classification_pred = self.class_head_fc2(x_classification)
        
        assert bridges[idx - 1].shape == (1, 1024, 48, 48)

        # Run bridging between down and up blocks
        for i, block in enumerate(self.up_blocks):
            x = block(x, bridges[4 - i])
        segmentation_pred = self.out(x)
        del bridges

        return classification_pred, segmentation_pred


