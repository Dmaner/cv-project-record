import argparse
import torch.nn as nn
import torch
import pickle
import os
import numpy as np
from PIL import Image
from VGG import get_vgg
from Resnet import get_resnet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class MYDATA(Dataset):
    def __init__(self, datas, labels, transform=None):
        super(MYDATA, self).__init__()
        self.datas = datas
        self.targets = labels
        self.transform = transform

    def __getitem__(self, item):
        data, target = self.datas[item], self.targets[item]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.datas)


def get_cifar10(root):
    """
    Get cifar10 data from cifar-10-batches-py
    """
    base_folder = "cifar-10-batches-py"
    train_file = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_file = "test_batch"

    train_datas = []
    train_targets = []
    test_datas = []
    test_targets = []
    for file in train_file:
        path = os.path.join(root, base_folder, file)
        with open(path, 'rb') as f:
            datas = pickle.load(f, encoding='bytes')

        train_targets += datas['labels'.encode('utf8')]
        for i in range(len(datas['data'.encode('utf8')])):
            train_datas.append(
                Image.fromarray(
                    np.rot90(
                        datas["data".encode("utf8")][i].reshape((32, 32, 3), order="F"), -1
                    )
                )
            )
    test_path = os.path.join(root, base_folder, test_file)
    with open(test_path, "rb") as f:
        datas = pickle.load(f, encoding="bytes")

    test_targets += datas["labels".encode("utf8")]
    for i in range(len(datas["data".encode("utf8")])):
        test_datas.append(
            Image.fromarray(
                np.rot90(
                    datas["data".encode("utf8")][i].reshape((32, 32, 3), order="F"), -1
                )
            )
        )

    return train_datas, train_targets, test_datas, test_targets


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
                args.model, epoch, idx * len(data), len(train_loader.dataset), 100 * idx / len(train_loader),
                loss.item()))
            Writer.add_scalar("Train loss", loss.item(), (epoch - 1) * len(train_loader) + idx)
            # Writer.add_scalars("Train", {"Train Loss": loss.item()}, global_step=(epoch-1)*len(train_loader)+idx)


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
    # Writer.add_scalars("Test", {"Accuracy:": accuracy}, global_step=epoch)
    return accuracy


def pre_args():
    """
    information of args
    """
    print("Model: ", args.model)
    print("Optimizer: ", args.optm)
    print("Learning rate: ", args.lr)
    print("Batch size: ", args.batch)
    print("Use cuda: ", args.cuda)
    print("Epochs", args.epochs)
    print("Learning rate change interval:", args.lr_step)
    print("Learning rate change rate: ", args.lr_rate)


if __name__ == '__main__':
    # add argument
    parser = argparse.ArgumentParser(description="model test")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help='train times')
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--interval', type=int, default=100, help='Record the loss')
    parser.add_argument("--optm", type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
    parser.add_argument('--save_dir', default='../checkpoints', help='Directory save the model')
    parser.add_argument('--cuda', type=bool, default=True, help="if use gpu")
    parser.add_argument('--vi', default='../logs', help="visualize the model training")
    parser.add_argument('--model', default='vgg-19', help='chose model')
    parser.add_argument("--lr_step", default=50, help='choose lr step')
    parser.add_argument("--lr_rate", default=0.1, help="learning rate change factor")
    parser.add_argument("--add_se", default=False, type=bool, help='if add se')
    parser.add_argument("--use_data", default=False, type=bool, help="if use data augmentation")
    parser.add_argument("--use_fp16", default=False, type=bool, help="if use fp16")

    args = parser.parse_args()

    # using tensorboard
    if not os.path.exists(args.vi):
        os.mkdir(args.vi)

    line_name = "/{}_{}_{}_{}_{}/"
    s_model = args.model.split('-')[0]
    s_optm = "Adam" if args.optim == "adam" else "SGD"
    s_se = "NoSE" if args.add_se else "SE"
    s_transf = "Transform" if args.use_data else "NoTransform"
    s_lr = "lr"+str(args.lr)
    Writer = SummaryWriter(args.vi+line_name.format(s_model, s_optm, s_se, s_transf, s_lr))

    if args.use_data:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.ToTensor()

    train_data, train_target, test_data, test_target = get_cifar10('../dataset')
    train_dataset = MYDATA(train_data, train_target, transform=transform)
    test_dataset = MYDATA(test_data, test_target, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)

    # using gpu or not

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if use_cuda else 'cpu')
    if args.model.startswith("vgg"):
        model = get_vgg(args.model, True, 10, args.add_se).to(device)
    else:
        model = get_resnet(args.model, args.add_se).to(device)

    best_score = 0

    # initial optimizer and loss function
    if args.optm == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    lossfuc = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_rate)

    # print information
    pre_args()

    for i in range(1, args.epochs + 1):
        train(model, optimizer, lossfuc, i)
        score = test(model, i)
        scheduler.step(args.epochs)

        # save best model
        if best_score < score:
            best_score = score
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), args.save_dir+"/{}.pt".format(args.model))
    images, _ = next(iter(train_loader))
    Writer.add_graph(model, (images.cuda(),))
    Writer.close()
