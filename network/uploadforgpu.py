import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """
    make feature layers
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_vgg(cfg, batch_norm: bool, num_classes: int, init_weight: bool):
    """
    Make vgg's feature layers
    """
    return VGG(make_layers(cfgs.get(cfg), batch_norm), num_classes, init_weight)


def train(model: nn.Module, optimizer, lossfuc, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # clear historical gradient
        optimizer.zero_grad()
        output = model(data)
        loss = lossfuc(output, target)
        loss.backward()

        # next step
        optimizer.step()
        if idx % args.interval == 0:
            print('Model: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}'.format(
                args.model, epoch, idx * len(data), len(train_loader.dataset), 100 * idx / len(train_loader), loss.item()))
            Writer.add_scalar("{}/Train loss".format(args.model), loss.item(), (epoch-1)*len(train_loader)+idx)


def test(model: nn.Module, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # get prediction batch x 1
            prediction = torch.argmax(output, dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Epoch:[{}]/[{}]\tAccuracy: {:.2f}%'.format(epoch, args.epochs, accuracy))
    Writer.add_scalar("Accuracy", accuracy, epoch)
    return accuracy


if __name__ == '__main__':
    cfgs = {
        'vgg-11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg-13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg=19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                   'M'],
    }
    # add argument
    parser = argparse.ArgumentParser(description="model test")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=30, help='train times')
    parser.add_argument('--lr', type=int, default=0.0001, help="Learning rate")
    parser.add_argument('--interval', type=int, default=100, help='Record the loss')
    parser.add_argument('--save_dir', default='../checkpoints/model-alexnet.pt', help='Directory save the model')
    parser.add_argument('--cuda', type=bool, default=True, help="if use gpu")
    parser.add_argument('--vi', default='../logs', help="visualize the model training")
    parser.add_argument('--model', default='vgg-11', help='chose model')

    args = parser.parse_args()

    # using tensorboard
    Writer = SummaryWriter(args.vi)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_data = CIFAR10('data', train=True, download=True, transform=transform)
    test_data = CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch)
    test_loader = DataLoader(test_data, batch_size=args.batch)

    # using gpu or not
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if use_cuda else 'cpu')
    model = get_vgg(args.model, True, 10, True).to(device)

    best_score = 0

    # initial optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfuc = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

    for i in range(1, args.epochs+1):
        train(model, optimizer, lossfuc, i)
        score = test(model, i)
        scheduler.step(args.epochs)

        # save best model
        if best_score < score:
            best_score = score
            torch.save(model.state_dict(), args.save_dir)
    images, _ = next(iter(train_loader))
    Writer.add_graph(model, (images.cuda(),))
    Writer.close()