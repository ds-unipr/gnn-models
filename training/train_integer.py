from graphs_dataset import RandomUndirectedGraphsDataset
import torch
from torch.utils.data import random_split
from graphs_plots import plot_single_model
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import sys
import utils
import os
from torch_geometric.loader import DataLoader    
import time


##### TO BE CHANGED FOR EVERY TRY #####

import models.model_a4 as m
model_name = 'modelA4_independence_number'

invariant_target = "independence_number"  #  matching_number, diameter, clique_number, independence_number

#######################################

torch.manual_seed(123)

model = m.Model(h_channels=m.hidden_channels)
 
utils.create_out_dirs(model_name)
utils.write_model(model_name, model, m.alpha, batch_size=m.train_batch_size)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=m.alpha)
model = model.to(device)
criterion = m.criterion


def train(data_loader, model, epoch: utils.EpochSummary):
    model.train()
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        out = torch.squeeze(out)
        reshaped_y = data.y.view(-1, 7)
        target = reshaped_y[:, invariant_idx]
        valid_mask = target != -1
        target = target[valid_mask]
        target = target.long()
        out = out[valid_mask]
        loss = criterion(out, target)
        epoch.add_training_loss(loss.item())
        loss.backward()
        optimizer.step()


def test(data_loader, model, return_preds=False):
    all_preds = []
    all_targets = []

    model.eval()
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            out = torch.squeeze(out)
            reshaped_y = data.y.view(-1, 7)
            targets = reshaped_y[:, invariant_idx]

            valid_mask = targets != -1
            logits_valid = out[valid_mask]
            targets_valid = targets[valid_mask]

            if len(targets_valid) == 0:
                continue

            pred_labels = logits_valid.argmax(dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_targets.extend(targets_valid.cpu().numpy())

    if len(all_targets) == 0:
        return 0.0 if not return_preds else (0.0, [], [])

    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    if return_preds:
        return balanced_acc, all_preds, all_targets
    return balanced_acc

# calcolare loss come criterion((out/target), 1)
# calcolare errore come |out - target|/|target|


dataset = RandomUndirectedGraphsDataset(root="../data")
dataset = dataset.shuffle()


invariant_idx = dataset[0].invariants_order.index(invariant_target)

train_dataset, test_dataset = random_split(dataset, [.8, .2])

#class_weights = utils.compute_class_weights(train_dataset, invariant_idx, power=0.5)
#class_weights = class_weights.to(device)
#criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

train_sampler = utils.StratifiedBatchSampler(train_dataset, invariant_idx, batch_size=m.train_batch_size, min_examples_per_class=10)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
test_sampler = utils.StratifiedBatchSampler(test_dataset, invariant_idx, batch_size=m.test_batch_size, min_examples_per_class=5)
test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
#test_loader = DataLoader(test_dataset, batch_size=m.test_batch_size, shuffle=True)

epochs = 100
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    if epoch == epochs - 1:
        test_score, all_preds, all_targets = test(model=model, data_loader=test_loader, return_preds=True)
        cm_path = f"out/{model_name}/confusion_matrix.png"
        utils.plot_confusion_matrix(all_targets, all_preds, save_path=cm_path)
    else:
        test_score = test(model=model, data_loader=test_loader)
    epoch_summary.commit_integer(test_accuracy=test_score)
    epoch_summaries.append(epoch_summary)
    epoch_summary.print_avg_loss()

utils.write_epoch_summary_integer(model_name=model_name, epochs=epoch_summaries)
plot_single_model(model_name=model_name)