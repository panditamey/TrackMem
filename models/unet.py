"""TrackNetV2 encoder-decoder (VGG-style U-Net), shared by TrackNetV2-V5.

conv_order and upsample default to the TrackNetV5 SDK (Conv -> ReLU -> BN, bilinear with
align_corners=True). The 1x1 `head` produces one draft logit map per input frame.
"""
import torch
import torch.nn as nn


def conv_block(cin, cout, n, order):
    layers = []
    for i in range(n):
        conv = nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False)
        if order == "conv_relu_bn":
            layers += [conv, nn.ReLU(inplace=True), nn.BatchNorm2d(cout)]
        else:
            layers += [conv, nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class TrackNetUNet(nn.Module):
    def __init__(self, in_ch, out_ch, width_mult=1.0, conv_order="conv_relu_bn",
                 upsample="bilinear", head_bias=False):
        super().__init__()
        c1, c2, c3, c4 = (int(c * width_mult) for c in (64, 128, 256, 512))
        self.down1 = conv_block(in_ch, c1, 2, conv_order)
        self.down2 = conv_block(c1, c2, 2, conv_order)
        self.down3 = conv_block(c2, c3, 3, conv_order)
        self.bottleneck = conv_block(c3, c4, 3, conv_order)
        self.up1 = conv_block(c4 + c3, c3, 3, conv_order)
        self.up2 = conv_block(c3 + c2, c2, 2, conv_order)
        self.up3 = conv_block(c2 + c1, c1, 2, conv_order)
        self.head = nn.Conv2d(c1, out_ch, 1, bias=head_bias)
        self.pool = nn.MaxPool2d(2)
        if upsample == "bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def features(self, x):
        """Decoder output features (B, c1, H, W)."""
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.up1(torch.cat([self.upsample(x), x3], dim=1))
        x = self.up2(torch.cat([self.upsample(x), x2], dim=1))
        return self.up3(torch.cat([self.upsample(x), x1], dim=1))

    def forward(self, x):
        return self.head(self.features(x))
