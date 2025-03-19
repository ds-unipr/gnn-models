from graphs_dataset import RandomUndirectedGraphsDataset
import torch
from torch.utils.data import random_split
import utils
import os
from torch_geometric.loader import DataLoader


##### TO BE CHANGED FOR EVERY TRY #####

import models.model_d5 as m
model_name = 'modelD5'

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
criterion = m.criterion

model = model.to(device)

def train(data_loader, model, epoch: utils.EpochSummary):
    errs = torch.zeros(len(data_loader), device=device)
    model.train()
    for i, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        out = torch.squeeze(out)
        reshaped_y = data.y.view(-1, 7)
        target = reshaped_y[:, invariant_idx]
        out_target_distance = torch.abs(out - target) / ((torch.abs(out) + torch.abs(target)) / 2 + 1e-9)
        err = out_target_distance.mean()
        errs[i] = err
        loss = criterion(out, target)
        epoch.add_trainging_loss(loss.item())
        loss.backward()
        optimizer.step()
    error = errs.mean().item()
    print(f"training error: {error}")
    return error


def test(data_loader, model):
    errs = torch.zeros(len(data_loader), device=device)
    model.eval()
    for i, data in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            out = torch.squeeze(out)
            target = data.y.view(-1, 7)[:, invariant_idx]
            out_target_distance = torch.abs(out - target) / ((torch.abs(out) + torch.abs(target)) / 2 + 1e-9)
            err = out_target_distance.mean()
            errs[i] = err
    error = errs.mean().item()
    print(f"test error: {error}")
    return error


dataset = RandomUndirectedGraphsDataset(root="data")
dataset = dataset.shuffle()

invariant_idx = dataset[0].invariants_order.index("spectral_radius_laplacian")

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(train_dataset, batch_size=m.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=m.test_batch_size, shuffle=True)

epochs = 50
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_err = train(model=model, data_loader=train_loader, epoch=epoch_summary)
    test_err = test(model=model, data_loader=test_loader)
    epoch_summary.commit(train_error=train_err, test_error=test_err)
    epoch_summaries.append(epoch_summary)

utils.write_epoch_summary(model_name=model_name, epochs=epoch_summaries)