import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from graphs_dataset import RandomUndirectedGraphsDataset
from graphs_plots import plot_single_model
import utils


torch.manual_seed(123)

class GraphMLP(nn.Module):
    def __init__(self, max_num_edges=1225, hidden_dim=64, output_dim=51):
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
        return self.mlp(vectorized_graph)
    
    
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = GraphMLP()

utils.create_out_dirs("MLP_clique_number")
utils.write_model("MLP_clique_number", model, alpha=1e-4, batch_size=256)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
    
def train(data_loader, model, epoch: utils.EpochSummary):
    model.train()
    for (data, labels) in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        out = model(data)
        #out = torch.squeeze(out)
        reshaped_labels = labels.view(-1, 7)
        target = reshaped_labels[:, invariant_idx].long()
        valid_mask = target != -1
        out = out[valid_mask]
        target = target[valid_mask].long()
        loss = criterion(out, target)
        epoch.add_training_loss(loss.item())
        loss.backward()
        optimizer.step()


def test(data_loader, model):
    top1_correct = 0
    top3_correct = 0
    total = 0
    model.eval()

    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        reshaped_labels = labels.view(-1, 7)
        target = reshaped_labels[:, invariant_idx]
        valid_mask = target != -1
        if valid_mask.sum() == 0:
            continue  
        with torch.no_grad():
            out = model(data)  
            out_valid = out[valid_mask]
            target_valid = target[valid_mask].long()
            _, top1_pred = out_valid.max(dim=1)
            top3_preds = torch.topk(out_valid, 3, dim=1).indices
            top1_correct += (top1_pred == target_valid).sum().item()
            top3_correct += (top3_preds == target_valid.view(-1, 1)).sum().item()
            total += len(target_valid)
    if total > 0:
        top1_accuracy = top1_correct / total
        top3_accuracy = top3_correct / total
        print(f"Test top-1 accuracy: {top1_accuracy:.4f}")
        print(f"Test top-3 accuracy: {top3_accuracy:.4f}")
    else:
        top1_accuracy = 0.0
        top3_accuracy = 0.0
    return top3_accuracy


dataset = RandomUndirectedGraphsDataset(root="data", plain_vector=True)

invariant_idx = dataset.invariants_order.index("clique_number")

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

epochs = 100
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    test_accuracy = test(model=model, data_loader=test_loader)
    epoch_summary.commit_integer(test_accuracy = test_accuracy)
    epoch_summaries.append(epoch_summary)
    epoch_summary.print_avg_loss()

utils.write_epoch_summary_integer(model_name="MLP_clique_number", epochs=epoch_summaries)
plot_single_model(model_name="MLP_clique_number")
