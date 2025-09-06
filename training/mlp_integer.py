import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from graphs_dataset import RandomUndirectedGraphsDataset
from graphs_plots import plot_single_model
from sklearn.metrics import balanced_accuracy_score
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

model_name = "MLP_independence_number"
utils.create_out_dirs(model_name)
utils.write_model(model_name, model, alpha=1e-4, batch_size=256)

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


def test(data_loader, model, return_preds=False):
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for (data, labels) in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.long()

            out = model(data)
            reshaped_labels = labels.view(-1, 7)
            target = reshaped_labels[:, invariant_idx]
            valid_mask = target != -1
            logits_valid = out[valid_mask]
            targets_valid = target[valid_mask]
            if len(targets_valid) == 0:
                continue
            pred_labels = logits_valid.argmax(dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_targets.extend(targets_valid.cpu().numpy())

    if len(all_targets) == 0:
        return (0.0, [], []) if return_preds else 0.0
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    return (balanced_acc, all_preds, all_targets) if return_preds else balanced_acc  


dataset = RandomUndirectedGraphsDataset(root="data", plain_vector=True)

invariant_idx = dataset.invariants_order.index("independence_number")

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

epochs = 100
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    if epoch == epochs - 1:
        test_accuracy, all_preds, all_targets = test(model=model, data_loader=test_loader, return_preds=True)
        cm_path = f"out/{model_name}/confusion_matrix.png"
        utils.plot_confusion_matrix(all_targets, all_preds, save_path=cm_path)
    else:
        test_accuracy = test(model=model, data_loader=test_loader, return_preds=False)
    epoch_summary.commit_integer(test_accuracy = test_accuracy)
    epoch_summaries.append(epoch_summary)
    epoch_summary.print_avg_loss()

utils.write_epoch_summary_integer(model_name=model_name, epochs=epoch_summaries)
plot_single_model(model_name=model_name)
