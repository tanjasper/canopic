"""A network consisting only of fully connected layers"""

import torch.nn as nn
from models.discriminators.base_discriminator import BaseDiscriminator


class FCNet64(BaseDiscriminator):

    def __init__(self, num_classes=10):

        super(FCNet64, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(64*64*3, 2500),
            nn.ReLU(inplace=True),
            nn.Linear(2500, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1500),
            nn.ReLU(inplace=True),
            nn.Linear(1500, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x