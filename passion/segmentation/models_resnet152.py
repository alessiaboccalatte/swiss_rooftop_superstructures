import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
import torchvision.models as models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class=3):  
        super().__init__()

        self.base_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer1_1x1 = convrelu(256, 64, 1, 0)
        self.layer2 = self.base_layers[5]  
        self.layer2_1x1 = convrelu(512, 128, 1, 0)
        self.layer3 = self.base_layers[6]  
        self.layer3_1x1 = convrelu(1024, 256, 1, 0)
        self.layer4 = self.base_layers[7]  
        self.layer4_1x1 = convrelu(2048, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = convrelu(512 + 256, 256, 3, 1)
        self.conv_up2 = convrelu(256 + 128, 128, 3, 1)
        self.conv_up1 = convrelu(128 + 64, 64, 3, 1)
        self.conv_up0 = convrelu(64 + 64, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 64, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return out
