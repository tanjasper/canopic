import torch.nn as nn
from models.discriminators.base_discriminator import BaseDiscriminator

class AllConvNet224(BaseDiscriminator):

    def __init__(self, num_classes=10):

        super(AllConvNet224, self).__init__()

        self.features = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=5), # 224 --> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0), # 56 --> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1), # 56 --> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 28 --> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0), # 28 --> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 28 --> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 14 --> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),  # 14 --> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),  # 14 --> 7
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(384, 1024, kernel_size=3, stride=1, padding=1), # 7 --> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),  # 7 --> 7
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze()
        return x
