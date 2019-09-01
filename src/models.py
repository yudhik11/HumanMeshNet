import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from src.SMPL_pytorch import SMPL

from torchvision.models import resnet50, resnet18

class MyResnet18(nn.Module):
    def __init__(self, num_out, pretrained=False):
        super(MyResnet18, self).__init__()

        self.num_out = num_out

        self.resnet = resnet18(pretrained=pretrained)
        self.lrelu = nn.LeakyReLU()
        self.myfc = nn.Linear(1000, self.num_out)

    def forward(self, x):
        x = self.resnet(x)
        x = self.lrelu(x)
        x = self.myfc(x)
        return x

class MyFCNet2(nn.Module):
    def __init__(self, num_inp, num_out):
        super(MyFCNet2, self).__init__()
        self.fc1 = nn.Linear(num_inp, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_out)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
