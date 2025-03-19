import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import List

alpha = 0.01
train_batch_size = 256
test_batch_size = 128
hidden_channels = [8]  # Il numero di unità nei layer hidden
criterion = torch.nn.MSELoss()

class Model(torch.nn.Module):
    def __init__(self, h_channels: List[int], heads: int = 8):
        super(Model, self).__init__()
        self.gat1 = GATConv(in_channels=1, out_channels=h_channels[0], heads=heads, concat=True)
        self.gat2 = GATConv(in_channels=h_channels[0] * heads, out_channels=h_channels[0], heads=1, concat=False)
        self.lin = torch.nn.Linear(in_features=h_channels[0], out_features=1)
        
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.relu(x)  
        x = self.gat2(x, edge_index)
        x = F.relu(x) 
        x = global_mean_pool(x, batch) 
        x = self.lin(x) 
        return x
