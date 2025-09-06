import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List


##########################

##  model D3 for integer invariants

#########################

alpha = 1e-4
train_batch_size = 512
test_batch_size = 512
hidden_channels = [32]                                ####provare aumentare hidden channel   32
criterion = torch.nn.CrossEntropyLoss()

class Model(torch.nn.Module):
    def __init__(self, h_channels: List[int]):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        self.sageConv = SAGEConv(
            in_channels=1,
            out_channels=h_channels[0],
            aggr="sum",
            root_weight=True,
            bias=True
        )
        self.lin = torch.nn.Linear(
            in_features=h_channels[0],
            out_features=51,                  #48 diameter, 26 matching_number, 49 clique_number, 51 indipendence number
            bias=True
        )

    def forward(self, x, edge_index, batch):
        x = self.sageConv(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x