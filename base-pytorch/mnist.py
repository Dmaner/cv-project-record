import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

EPOCHS = 10
BARCH_SIZE = 28
LR = 0.01

transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='../dataset', train=True, download=True, transform=transf)
test_data = datasets.MNIST(root='../dataset', train=False, download=True, transform=transf)
train_loader = DataLoader(train_data, batch_size=BARCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BARCH_SIZE)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28*28)
        output = self.net(x)
        return F.log_softmax(output, dim=1)


def train(model: nn.Module, epoch):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    lossfuc = nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device),target.to(device)

        # 清除历史梯度
        optimizer.zero_grad()
        output = model(data)
        loss = lossfuc(output, target)
        loss.backward()

        # 下一步梯度
        optimizer.step()
        if idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset), 100 * idx / len(train_loader), loss.item()))


def test(model: nn.Module, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 得到batch*1的预测结果
            prediction = torch.argmax(output, dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    print('Epoch:[{}]/[{}]\tAccuracy: {:.2f}%'.format(epoch, EPOCHS, 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = MLP().to(device)
    for i in range(EPOCHS):
        train(model, i)
        test(model, i)