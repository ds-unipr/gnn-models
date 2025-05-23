from graphs_dataset import RandomUndirectedGraphsDataset
import torch
from torch.utils.data import random_split
from graphs_plots import plot_single_model
import utils
import os
from torch_geometric.loader import DataLoader
import time


##### TO BE CHANGED FOR EVERY TRY #####

import models.model_l3_v3 as m
model_name = 'modelL3_v3_sgd_cosine_100epochs'

#######################################

torch.manual_seed(123)

model = m.Model(h_channels=m.hidden_channels)

utils.create_out_dirs(model_name)
utils.write_model(model_name, model, m.alpha, batch_size=m.train_batch_size)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# optimizer = torch.optim.AdamW(model.parameters(), lr=m.alpha, weight_decay=1e-4)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=m.alpha,
    momentum=0.9,
    weight_decay=5e-5
)


criterion = m.criterion

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)


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
        loss = criterion(out, target)
        epoch.add_training_loss(loss.item())
        loss.backward()
        optimizer.step()


def test(data_loader, model):
    errs = torch.zeros(len(data_loader), device=device)
    model.eval()
    for i, data in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            out = torch.squeeze(out)
            err = utils.calc_error(
                y=data.y, out=out, invariant_index=invariant_idx)
            errs[i] = err
    error = errs.mean().item()
    print(f"test error: {error}")
    return error

# calcolare loss come criterion((out/target), 1)
# calcolare errore come |out - target|/|target|


dataset = RandomUndirectedGraphsDataset(root="data")
dataset = dataset.shuffle()


invariant_idx = dataset[0].invariants_order.index("spectral_radius_laplacian")

train_dataset, test_dataset = random_split(dataset, [.8, .2])

train_loader = DataLoader(
    train_dataset, batch_size=m.train_batch_size, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=m.test_batch_size, shuffle=True)

# prev_lr = m.alpha

epochs = 100
epoch_summaries = []
for epoch in range(epochs):
    epoch_summary = utils.EpochSummary(index=epoch)
    print(f"\nEpoch {epoch+1}/{epochs}")

    train(model=model, data_loader=train_loader, epoch=epoch_summary)
    test_err = test(model=model, data_loader=test_loader)
    epoch_summary.commit(test_error=test_err)

    scheduler.step()

    # current_lr = scheduler.optimizer.param_groups[0]['lr']
    # if current_lr != prev_lr:
    #    print(f"learning rate decreased to {current_lr}")
    # prev_lr = current_lr

    epoch_summaries.append(epoch_summary)
    epoch_summary.print_avg_loss()

utils.write_epoch_summary(model_name=model_name, epochs=epoch_summaries)
plot_single_model(model_name=model_name)
