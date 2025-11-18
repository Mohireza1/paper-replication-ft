import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample = downsample  # optional projection (1x1 conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # apply projection if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---- 2. Configurable ResNet ----
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, base_width=16, dropout=0.0):
        super().__init__()
        self.in_channels = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # make the residual stages
        self.layer1 = self._make_layer(
            block, base_width, layers[0], stride=1, dropout=dropout
        )
        self.layer2 = self._make_layer(
            block, base_width * 2, layers[1], stride=2, dropout=dropout
        )
        self.layer3 = self._make_layer(
            block, base_width * 4, layers[2], stride=2, dropout=dropout
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 4 * block.expansion, num_classes)

        # weight initialization (useful for research consistency)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride, dropout):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    1,
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample, dropout)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---- 3. Factory function for ResNet-20 ----
def ResNet20(num_classes=10, dropout=0.0):
    # 3 blocks per layer group = 20 layers total
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, dropout=dropout)


# ---- 4. Example ----
if __name__ == "__main__":
    model = ResNet20(num_classes=10, dropout=0.1)
    x = torch.randn(1, 3, 32, 32)
    print(model(x).shape)
