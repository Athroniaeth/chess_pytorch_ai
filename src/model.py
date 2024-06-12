import torch
from torch import nn


class ChessModel(nn.Module):
    def __init__(self, input_size: int = 837):
        super(ChessModel, self).__init__()

        self.linear_1 = nn.Linear(input_size, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.linear_5 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.relu(self.linear_1(x))
        _x = self.relu(self.linear_2(_x))
        _x = self.relu(self.linear_3(_x))
        _x = self.relu(self.linear_4(_x))
        _x = self.sigmoid(self.linear_5(_x))
        return _x
