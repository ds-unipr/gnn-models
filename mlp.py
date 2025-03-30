import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from graphs_dataset import RandomUndirectedGraphsDataset
import utils


torch.manual_seed(123)

class GraphMLP(nn.Module):
    def __init__(self, max_num_edges=1225, hidden_dim=64, output_dim=1):
        super(GraphMLP, self).__init__()
        
        input_dim = 2 * max_num_edges  # adjacency + mask
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, vectorized_graph):
        out = self.mlp(vectorized_graph)
        return out
    
    
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = GraphMLP()

utils.create_out_dirs("MLP")
utils.write_model("MLP", model, alpha=1e-4, batch_size=256)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = utils.MSLELoss()
    
def train(data_loader, model, epoch: utils.EpochSummary):
    model.train()
    for (data, labels) in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        out = torch.squeeze(out)
        reshaped_labels = labels.view(-1, 7)
        target = reshaped_labels[:, invariant_idx]
        loss = criterion(out, target)
        epoch.add_training_loss(loss.item())
        loss.backward()
        optimizer.step()


def test(data_loader, model):
    errs = torch.zeros(len(data_loader), device=device)
    model.eval()
    for i, (data, labels) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(data)
            out = torch.squeeze(out)
            err = utils.calc_error(y=labels, out=out, invariant_index=invariant_idx)
            errs[i] = err
    error = errs.mean().item()
    print(f"test error: {error}")
    return error


dataset = RandomUndirectedGraphsDataset(root="data", plain_vector=True)

invariant_idx = dataset.invariants_order.index("spectral_radius_laplacian")

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

epochs = 100
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    test_err = test(model=model, data_loader=test_loader)
    epoch_summary.commit(test_error=test_err)
    epoch_summaries.append(epoch_summary)
    epoch_summary.print_avg_loss()

utils.write_epoch_summary(model_name="MLP", epochs=epoch_summaries)
