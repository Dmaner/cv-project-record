import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_ih = nn.Linear(input_size, hidden_size*4)
        self.linear_hh = nn.Linear(hidden_size, hidden_size*4)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.hidden_size**0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(len(x), self.hidden_size, device=x.device)
            cx = torch.zeros(len(x), self.hidden_size, device=x.device)
        else:
            hx, cx = hx
        gates = self.linear_ih(x) + self.linear_hh(hx)
        gates = torch.chunk(gates, 4, 1)
        ingate = gates[0].sigmoid_()
        forgetgate = gates[1].sigmoid_()
        cellgate = gates[2].tanh_()
        outgate = gates[3].sigmoid_()
        cy = (forgetgate * cx).add_(ingate * cellgate)
        hy = outgate * cy.tanh()
        return hy, cy


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_ih = nn.Linear(input_size, hidden_size * 3)
        self.linear_hh = nn.Linear(hidden_size, hidden_size * 3)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(len(input), self.hidden_size, device=input.device)
        igates = torch.chunk(self.linear_ih(input), 3, 1)
        hgates = torch.chunk(self.linear_hh(hidden), 3, 1)
        reset_gate = hgates[0].add_(igates[0]).sigmoid_()
        input_gate = hgates[1].add_(igates[1]).sigmoid_()
        new_gate = igates[2].add_(hgates[2].mul_(reset_gate)).tanh_()
        return (hidden - new_gate).mul_(input_gate).add_(new_gate)


class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size=1):
        super(MyLSTM, self).__init__()
        self.hidden = hidden_size
        # (feature length, hidden size, layers)
        self.lstm1 = LSTMCell(feature_size, hidden_size)
        self.lstm2 = LSTMCell(hidden_size, hidden_size)
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

        # predict the next sequence with
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


# 定义神经网络
class AnalyWithGRU(torch.nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
        super(AnalyWithGRU, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        # 指定 GRU 的各个参数
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        # 最后一层：一个线性层，全连接，用于分类
        self.out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, word_inputs, hidden):
        '''
        batch_size:  batch 的大小，这里默认是1，表示一句话
        word_inputs: 输入的向量
        hidden: 上下文输出
        '''
        # resize 输入的数据
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)
        output, hidden = self.gru(inputs, hidden)
        output = self.out(output)
        # 仅返回最后一个向量,用 RNN 表示
        output = output[:, -1, :]
        return output, hidden

    def init_hidden(self):
        # 每次第一个向量没有上下文，在这里返回一个上下文
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden