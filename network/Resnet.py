import torch
import torch.nn as nn
from Senet import SELayer

cfgs = {
    "resnet-18": [2, 2, 2, 2],
    "resnet-34": [3, 4, 6, 3],
    "resnet-50": [3, 4, 6, 3],
    "resnet-101": [3, 4, 23, 3]
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_se=False):
        super(BasicBlock, self).__init__()
        self.layes = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if use_se:
            self.se = SELayer(out_channel)
        else:
            self.se = None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.layes(x)
        if self.se:
            out = self.se(out)
        x = x if not self.downsample else self.downsample(x)
        out += x

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_se=False):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel*self.expansion)
        )
        if use_se:
            self.se = SELayer(out_channel*self.expansion)
        else:
            self.se = None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.layers(x)
        if self.se:
            out = self.se(out)
        x = x if not self.downsample else self.downsample(x)
        out += x

        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, num_class=10, use_se=False):
        super(Resnet, self).__init__()
        self.use_se = use_se
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channel, out_channels, stride, downsample, self.use_se)]
        self.in_channel = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channels, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def get_resnet(name:str, use_se=False):
    """
    get resnet as your choice
    """
    if int(name.split('-')[1]) < 50:
        block = BasicBlock
    else:
        block = Bottleneck
    layers = cfgs.get(name)
    model = Resnet(block, layers, use_se=use_se)

    return model


if __name__ == '__main__':
    model = get_resnet("resnet-18", use_se=True)
    print(model)
    sample = torch.rand((10, 3, 32, 32))

    output = model(sample)
    print(output.shape)