import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from main import MYDATA, get_cifar10


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset), 100 * idx / len(train_loader), loss.item()))
            Writer.add_scalar("Train loss", loss.item(), (epoch-1)*len(train_loader)+idx)


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
    # add argument
    parser = argparse.ArgumentParser(description="Alexnet test")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=30, help='train times')
    parser.add_argument('--lr', type=int, default=0.0001, help="Learning rate")
    parser.add_argument('--interval', type=int, default=100, help='Record the loss')
    parser.add_argument('--save_dir', default='../checkpoints/model-alexnet.pt', help='Directory save the model')
    parser.add_argument('--cuda', type=bool, default=True, help="if use gpu")
    parser.add_argument('--vi', default='../logs', help="visualize the model training")

    args = parser.parse_args()

    # using tensorboard
    Writer = SummaryWriter(args.vi)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_data, train_target, test_data, test_target = get_cifar10('../dataset')
    train_dataset = MYDATA(train_data, train_target, transform=transform)
    test_dataset = MYDATA(test_data, test_target, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)

    # using gpu or not
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if use_cuda else 'cpu')
    model = AlexNet().to(device)

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
