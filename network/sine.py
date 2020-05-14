import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import MyCell


class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size=1):
        super(MyLSTM, self).__init__()
        self.hidden = hidden_size
        # (feature length, hidden size, layers)
        self.lstm1 = MyCell.LSTMCell(feature_size, hidden_size)
        self.lstm2 = MyCell.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, future=0):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        h_t2 = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)
        c_t2 = torch.zeros(x.size(0), self.hidden, dtype=x.dtype, device=x.device)

        # split by batch input_t shape: ( batch size, 1)
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # predict the next sequence
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def draw(y, epoch):
    plt.figure(figsize=(30, 10))
    plt.title('Sine prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    for i, color in enumerate(['r', 'g', 'b']):
        x = y.shape[1] - future
        plt.plot(np.arange(x), y[i][:x], color, linestyle=':', linewidth=2)
        plt.plot(np.arange(x, x + future), y[i][x:], color, linestyle=':', linewidth=2)
    plt.savefig('../results/Predction-{}.png'.format(epoch))
    plt.close()


def train(model, optimizer, epoch, criterion, data, target):
    model.train()
    print('Epoch: [{}]/[{}]'.format(epoch, EPOCH))

    def closure():
        """
        定义闭包函数用于LBFGS优化器
        """
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target).to(device)
        print('Loss: {:5e}'.format(loss.item()))
        loss.backward()
        return loss

    optimizer.step(closure)


def test(model: nn.Module, epoch, test_data, test_target):
    model.eval()
    with torch.no_grad():
        prediction = model(test_data, future=future)
        loss = criterion(prediction[:, :-future], test_target)
        prediction = prediction.detach().cpu().numpy()
        print('Test loss: {:5e}'.format(loss.item()))

    # show predicition
    draw(prediction, epoch)


if __name__ == '__main__':

    LR = 0.8
    EPOCH = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    future = 1000
    h_state = None

    # load dataset
    data = torch.load('../dataset/traindata.pt')

    # train data shape (time step, batch size, feature length)
    train_data = torch.from_numpy(data[3:, :-1]).to(device)
    train_target = torch.from_numpy(data[3:, 1:]).to(device)
    test_data = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)

    # build model
    model = MyCell.MyLSTM(1, 50).to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=LR)
    # train and test
    for epoch in range(EPOCH):
        train(model, optimizer, epoch, criterion, train_data, train_target)
        test(model, epoch, test_data, test_target)
