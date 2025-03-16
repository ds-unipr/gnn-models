import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List

alpha = 0.01
train_batch_size = 256
test_batch_size = 128
hidden_channels = [4]
criterion = torch.nn.MSELoss()

class Model(torch.nn.Module):
    def __init__(self, h_channels: List[int]):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        self.sage1 = SAGEConv(
            in_channels=1,
            out_channels=h_channels[0],
            aggr="sum",
            root_weight=False,
            bias=True
        )
        self.sage2_sum = SAGEConv(
            in_channels=h_channels[0],
            out_channels=h_channels[0],
            aggr="sum",
            root_weight=False,
            bias=True
        )
        self.sage2_mean = SAGEConv(
            in_channels=h_channels[0],
            out_channels=h_channels[0],
            aggr="mean",
            root_weight=False,
            bias=False
        )
        self.lin = torch.nn.Linear(
            in_features=h_channels[0],
            out_features=1,
            bias=True
        )

    def forward(self, x, edge_index, batch):
        x = self.sage1(x, edge_index)
        x = x.relu()
        x = self.sage2_sum(x,edge_index)
        x = x + self.sage2_mean(x,edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = x.relu()
        return x