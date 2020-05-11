import torch
import torch.nn as nn
from Senet import SELayer


cfgs = {
    'vgg-11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg-13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg-19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # make sample shape from different shape of
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, use_se=False):
    """
    make feature layers
    """
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            if use_se:
                channels = cfg[i-1]
                layers += [SELayer(channels)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_vgg(cfg, batch_norm: bool, num_classes: int, use_se=False):
    """
    Make vgg's feature layers
    """
    return VGG(make_layers(cfgs.get(cfg), batch_norm, use_se=use_se), num_classes)


if __name__ == '__main__':
    model = get_vgg('vgg-11', True, 10, True)
    imags = torch.rand((10, 3, 32, 32))
    out = model(imags)
    print(model)
    print(out.shape)