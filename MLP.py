import torch
from torch import nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_num, output_num):
        super(MLP, self).__init__()
        self.layer = nn.Linear(input_num, output_num)

    def forward(self, x):
        x = self.layer(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)