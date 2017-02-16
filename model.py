import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.relu,
            self.maxpool,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.relu,
            self.maxpool,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.relu,
            self.maxpool
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            self.relu,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.relu,
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

