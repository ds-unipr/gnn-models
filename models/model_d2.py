import torch
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv, global_mean_pool
from typing import List

alpha = 0.01
train_batch_size = 256
test_batch_size = 256
hidden_channels = [1]
criterion = torch.nn.MSELoss()

class Model(torch.nn.Module):
    def __init__(self, h_channels: List[int]):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        # non trainable
        self.simpleConv = SimpleConv()
        self.lin = torch.nn.Linear(
            in_features=1,
            out_features=1,
            bias=True
        )

    def forward(self, x, edge_index, batch):
        x = self.simpleConv(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x