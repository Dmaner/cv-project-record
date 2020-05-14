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
