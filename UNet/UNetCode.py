import torch
from torch import nn
import torch.nn.functional as F

# Architecture based of https://arxiv.org/pdf/1505.04597.pdf 
# Utilized tutorial found at https://www.youtube.com/watch?v=IHq1t7NxS8k 

# Double Convolution (BatchNorm Layer Added)
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_c=3, out_c=1, wf=6):
        super(UNET, self).__init__()
        


