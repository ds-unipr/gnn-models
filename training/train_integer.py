from graphs_dataset import RandomUndirectedGraphsDataset
import torch
from torch.utils.data import random_split
from graphs_plots import plot_single_model
import numpy as np
import sys
import utils
import os
from torch_geometric.loader import DataLoader
import time


##### TO BE CHANGED FOR EVERY TRY #####

import models.model_a3 as m
model_name = 'modelA3_clique_number'

invariant_target = "clique_number"

#######################################

torch.manual_seed(123)

model = m.Model(h_channels=m.hidden_channels)

utils.create_out_dirs(model_name)
utils.write_model(model_name, model, m.alpha, batch_size=m.train_batch_size)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

criterion = m.criterion
optimizer = torch.optim.Adam(model.parameters(), lr=m.alpha)
model = model.to(device)


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


def test(data_loader, model):
    top1_correct = 0
    top3_correct = 0
    total = 0
    model.eval()
    for i, data in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            out = torch.squeeze(out)
            reshaped_y = data.y.view(-1, 7)
            target = reshaped_y[:, invariant_idx]
            valid_mask = target != -1
            out_valid = out[valid_mask]
            target_valid = target[valid_mask]
            if len(target_valid) == 0:
                continue 
            _, top1_pred = out_valid.max(dim=1)
            top1_correct_batch = (top1_pred == target_valid).sum().item()
            top3_preds = torch.topk(out_valid, 3, dim=1).indices
            top3_correct_batch = (top3_preds == target_valid.view(-1, 1)).sum().item()
            total_batch = len(target_valid)
            top1_correct += top1_correct_batch
            top3_correct += top3_correct_batch
            total += total_batch
    if total > 0:
        top1_accuracy = top1_correct / total
        top3_accuracy = top3_correct / total
        print(f"Test top-1 accuracy: {top1_accuracy}")
        print(f"Test top-3 accuracy: {top3_accuracy}")
    else:
        top1_accuracy = 0
        top3_accuracy = 0

    return top3_accuracy

# calcolare loss come criterion((out/target), 1)
# calcolare errore come |out - target|/|target|


dataset = RandomUndirectedGraphsDataset(root="../data")
dataset = dataset.shuffle()


invariant_idx = dataset[0].invariants_order.index(invariant_target)

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(
    train_dataset, batch_size=m.train_batch_size, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=m.test_batch_size, shuffle=True)

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

utils.write_epoch_summary_integer(model_name=model_name, epochs=epoch_summaries)
plot_single_model(model_name=model_name)