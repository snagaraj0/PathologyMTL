import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import sys
sys.path.append("/home/ubuntu/cs231nFinalProject/src/mtl_exps/mtl_models/HIPT/HIPT_4K")
from hipt_4k import HIPT_4K

class Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fcout4K = nn.Linear(192, 192 * 2)
        
        # self.correct = nn.Conv2d(768, 768, kernel_size=1)        
    
        self.correct = nn.Sequential(nn.Conv2d(768, 768, kernel_size=1), nn.LeakyReLU())

        self.grow = nn.UpsamplingBilinear2d(scale_factor=16)

        self.go_up = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64))

        self.last = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, out4k, out256):
        b, c, w, h = out256.shape
        fc_conv = self.fcout4K(out4k.squeeze()).view(b, 384, 1, 1).repeat(b, 1, w, h)
        new_embd = torch.cat((out256, fc_conv), dim=1)
        # new_embd = out256
        x = self.correct(new_embd)
        x = self.grow(x)
        x = self.go_up(x)
        return self.last(x)


class MTLHIPT(nn.Module):
    def __init__(self, embed_dim=192, num_classes=6):
        super().__init__()

        # Encoder
        self.hipt = HIPT_4K()

        # Classification Decoder
        self.classify = nn.Sequential(
                nn.Linear(192, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, num_classes)
        )

        # Segmentation Decoder
        self.segment = Segmentation()
        # self.segment = nn.Sequential(nn.Conv2d(384, 3, kernel_size=1), nn.UpsamplingBilinear2d(scale_factor=256))
    def forward(self, x):

        # Encode
        out4k, out256 = self.hipt(x)

        # Classify
        class_preds = self.classify(torch.squeeze(out4k)).unsqueeze(0)

        # Segment
        segment_preds = self.segment(out4k, out256)

        return class_preds, segment_preds

