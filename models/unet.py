"""TrackNetV2 encoder-decoder (VGG-style U-Net), as used by TrackNetV3/V4/V5."""
import torch
import torch.nn as nn


def conv_block(cin, cout, n):
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class TrackNetUNet(nn.Module):
    def __init__(self, in_ch, out_ch, width_mult=1.0):
        super().__init__()
        c1, c2, c3, c4 = (int(c * width_mult) for c in (64, 128, 256, 512))
        self.down1 = conv_block(in_ch, c1, 2)
        self.down2 = conv_block(c1, c2, 2)
        self.down3 = conv_block(c2, c3, 3)
        self.bottleneck = conv_block(c3, c4, 3)
        self.up1 = conv_block(c4 + c3, c3, 3)
        self.up2 = conv_block(c3 + c2, c2, 2)
        self.up3 = conv_block(c2 + c1, c1, 2)
        self.head = nn.Conv2d(c1, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.up1(torch.cat([self.upsample(x), x3], dim=1))
        x = self.up2(torch.cat([self.upsample(x), x2], dim=1))
        x = self.up3(torch.cat([self.upsample(x), x1], dim=1))
        return self.head(x)
