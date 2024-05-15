import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, input_channels=2):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv15 = nn.Conv2d(64, input_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = relu(self.conv1(x))
        x2 = relu(self.conv2(x1))
        x3 = self.pool1(x2)
        
        x4 = relu(self.conv3(x3))
        x5 = relu(self.conv4(x4))
        x6 = self.pool2(x5)
        
        x7 = relu(self.conv5(x6))
        x8 = relu(self.conv6(x7))
        x9 = self.pool3(x8)
        
        # Bottleneck
        x10 = relu(self.conv7(x9))
        x11 = relu(self.conv8(x10))
        
        # Decoder
        x12 = self.upconv1(x11)
        x12 = torch.cat([x12, x8], dim=1)
        x13 = relu(self.conv9(x12))
        x14 = relu(self.conv10(x13))
        
        x15 = self.upconv2(x14)
        x15 = torch.cat([x15, x5], dim=1)
        x16 = relu(self.conv11(x15))
        x17 = relu(self.conv12(x16))
        
        x18 = self.upconv3(x17)
        x18 = torch.cat([x18, x2], dim=1)
        x19 = relu(self.conv13(x18))
        x20 = relu(self.conv14(x19))
        
        x21 = self.conv15(x20)
        return x21