from graphs_dataset import RandomUndirectedGraphsDataset
import torch
from torch.utils.data import random_split
import utils
import os
from torch_geometric.loader import DataLoader


##### TO BE CHANGED FOR EVERY TRY #####

import models.model_0_better as m
model_name = 'model0Better'

#######################################

torch.manual_seed(123)

model = m.Model(h_channels=m.hidden_channels)

utils.create_out_dirs(model_name)
utils.write_model(model_name, model, m.alpha, m.train_batch_size, hidden_channels=m.hidden_channels)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(data_loader, model, epoch: utils.EpochSummary):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = m.criterion
    model.train()
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch)
        out = torch.squeeze(out)
        reshaped_y = data.y.view(-1, 7)
        target = reshaped_y[:, invariant_idx]
        loss = criterion(out, target)
        epoch.add_trainging_loss(torch.sum(loss)/len(data))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(data_loader, model):
    errs = torch.zeros(len(data_loader))
    model.eval()
    for i, data in enumerate(data_loader):
        out = model(data.x, data.edge_index, data.batch)
        out = torch.squeeze(out)
        out[out == 0] = 0.1
        data.y[data.y == 0] = 0.1
        err = (torch.ones(data.batch_size) - (out/data.y.view(-1, 7)[:, invariant_idx])).abs().mean()
        errs[i] = err
    error = errs.mean().item()
    print(1 - error)
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
    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    err = test(model=model, data_loader=test_loader)
    epoch_summary.commit(test_loss=err)
    epoch_summaries.append(epoch_summary)

utils.write_epoch_summary(model_name=model_name, epochs=epoch_summaries)