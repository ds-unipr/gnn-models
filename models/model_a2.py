import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from typing import List
import utils

##########################

##  model D5 for integer invariants

#########################

alpha = 7.5e-5
train_batch_size = 256
test_batch_size = 256
hidden_channels = [64]
out_features=51
criterion = torch.nn.CrossEntropyLoss()


class Model(torch.nn.Module):
    def __init__(self, h_channels: List[int]):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        self.sageConv1 = SAGEConv(
            in_channels=1,
            out_channels=h_channels[0],
            aggr="sum",
            root_weight=True,
            bias=True
        )
        self.sageConv2 = SAGEConv(
            in_channels=h_channels[0],
            out_channels=h_channels[0],
            aggr="sum",
            root_weight=True,
            bias=True
        )
        self.sageConv3 = SAGEConv(
            in_channels=h_channels[0],
            out_channels=h_channels[0],
            aggr="mean",
            root_weight=True,
            bias=True
        )
        self.lin = torch.nn.Linear(
            in_features=h_channels[0] * 3,
            out_features=out_features,
            bias=True
        )

    def forward(self, x, edge_index, batch):
        x = self.sageConv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.sageConv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.sageConv3(x, edge_index)
        x = F.leaky_relu(x)
        mean_pool = global_mean_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        x = torch.cat([mean_pool, max_pool, add_pool], dim=1)
        x = self.lin(x)
        return x
