# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""

import torch
import torch.nn as nn
from torchsummary import summary


class VGG11_ConfigA(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv3_64 = self.make_block1(in_channels=3, out_channels=64)
        self.conv3_128 = self.make_block1(in_channels=64, out_channels=128)
        self.conv3_256 = self.make_block2(in_channels=128, out_channels=256)
        self.first_conv3_512 = self.make_block2(in_channels=256, out_channels=512)
        self.second_conv3_512 = self.make_block2(in_channels=512, out_channels=512)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(),
        )

    def make_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def make_block2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        x = self.conv3_64(x)
        x = self.conv3_128(x)
        x = self.conv3_256(x)
        x = self.first_conv3_512(x)
        x = self.second_conv3_512(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class VGG16_ConfigC(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv3_64 = self.make_block1(in_channels=3, out_channels=64)
        self.conv3_128 = self.make_block1(in_channels=64, out_channels=128)
        self.conv3_1_256 = self.make_block2(in_channels=128, out_channels=256)
        self.first_conv3_1_512 = self.make_block2(in_channels=256, out_channels=512)
        self.second_conv3_1_512 = self.make_block2(in_channels=512, out_channels=512)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(),
        )


    def make_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def make_block2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv3_64(x)
        x = self.conv3_128(x)
        x = self.conv3_1_256(x)
        x = self.first_conv3_1_512(x)
        x = self.second_conv3_1_512(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = VGG16_ConfigC()
    model = VGG11_ConfigA()
    model.train()
    model.to(device)
    summary(model, (3, 224, 224))
    # sample_input = torch.rand(2, 3, 128, 128)
    # result = model(sample_input)
    # print(result.shape)
