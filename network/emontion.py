import torch
import numpy as np
import random
import torch.nn as nn
import MyCell
from torch import optim
from tqdm import tqdm
from time import sleep
from torch.utils.data import DataLoader, Dataset


class MyData(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return len(self.data)


class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size=1):
        super(MyLSTM, self).__init__()
        self.hidden = hidden_size
        self.features = feature_size
        # (feature length, hidden size, layers)
        self.lstm1 = MyCell.LSTMCell(feature_size, hidden_size)
        self.lstm2 = MyCell.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        output = None
        h_t = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        h_t2 = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        c_t2 = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)

        # split by time step
        for x_t in x.chunk(x.size(1), dim=1):
            h_t, c_t = self.lstm1(x_t.squeeze(1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)

        return output


def wordtovec(PATH):
    w2v = {}
    with open(PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            key, value = line[0], [float(i) for i in line[1:]]
            w2v[key] = np.array(value)

    print("Finish word to vector")
    return w2v


def analysis(sentences, w2v: dict, length=40):
    """
    get vector from the sentence
    length: max length of sentence
    """
    s_vector = np.zeros((length, 50), dtype=np.float32)
    for i in range(length):
        if i >= len(sentences) or sentences[i] not in w2v:
            continue
        else:
            s_vector[i] = w2v.get(sentences[i])

    return s_vector


def read_comment(pos_path, neg_path, w2v:dict):
    pos_comments = []
    neg_comments = []

    # read positive comments
    with open(pos_path, 'r', encoding='windows-1252') as f:
        for line in tqdm(f.readlines()):
            pos_v = analysis(line.split(), w2v)
            pos_comments.append(pos_v)

    sleep(0.01)
    print("Finish positive comments transform.")

    # read negative comments
    with open(neg_path, 'r', encoding='windows-1252') as f:
        for line in tqdm(f.readlines()):
            neg_v = analysis(line.split(), w2v)
            neg_comments.append(neg_v)

    sleep(0.01)
    print("Finish negative comments transform.")

    pos_targets = [1] * len(pos_comments)
    neg_targets = [0] * len(neg_comments)
    train_data = pos_comments[:-1000] + neg_comments[:-1000]
    train_target = pos_targets[:-1000] + neg_targets[:-1000]
    test_data = pos_comments[-1000:] + neg_comments[-1000:]
    test_target = pos_targets[-1000:] + neg_targets[-1000:]

    # shuffle data
    train_data, train_target = shuffle(train_data, train_target)
    test_data, test_target = shuffle(test_data, test_target)
    print("Shuffle finish!")

    return train_data, train_target, test_data, test_target


def shuffle(data: list, target: list):
    """
    shuffle data
    """
    dataset = list(zip(data, target))
    random.shuffle(dataset)
    data, target = zip(*dataset)

    return torch.tensor(data), torch.tensor(target)


def train(model: nn.Module, epoch, criterion, optimizer):
    """
    train model
    """
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # clear historical gradient
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # next step
        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset), 100 * idx / len(train_loader), loss.item()))
            # Writer.add_scalar("Train loss", loss.item(), (epoch - 1) * len(train_loader) + idx)


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
    print('Epoch:[{}]/[{}]\tAccuracy: {:.2f}%'.format(epoch, EPOCH, accuracy))
    # Writer.add_scalar("Accuracy", accuracy, epoch)


if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.01
    EPOCH = 100

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # load data
    word2vector = wordtovec('../dataset/glove.6B.50d.txt')
    x_data, x_target, y_data, y_target = read_comment('../dataset/rt-polarity.pos', '../dataset/rt-polarity.neg', word2vector)
    trainset = MyData(x_data, x_target)
    testset = MyData(y_data, y_target)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

    model = MyLSTM(50, 50, 2).to(device)
    # model = Rnn(50, 50, 2, 2).to(device)
    lossfuc = nn.CrossEntropyLoss()
    optm = optim.Adam(model.parameters(), lr=LR)

    for i in range(EPOCH):
        train(model, i, lossfuc, optm)
        test(model, i)
