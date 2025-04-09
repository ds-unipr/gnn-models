from pathlib import Path
import os
from typing import List
import time
import torch
import sys
import torch.nn as nn
import numpy as np


# loss = mse(out/(target + epsilon), 1) => if target = 0  and out = 0.1 => loss = mse(10e9, 1)
# loss = msle(out, target) => if target = 0 and out = 0.1 => loss = mse(0.09, 0)
# MSLELoss: Mean Squared Logarithmic Error
# to deal with negative values ln(x+1) is reflected over the origin
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, out, target):
        return self.mse(torch.sign(out) * torch.log(torch.abs(out) + 1), torch.sign(target) * torch.log(torch.abs(target) + 1))
    
class EpochSummary():
    def __init__(self, index):
        self.index = index
        self.start = time.time()
        self.batch_training_losses = []
        self.test_error = 0

    def print_avg_loss(self):
        print(f"train loss: {np.mean(self.batch_training_losses)}")

    def add_training_loss(self, loss):
        self.batch_training_losses.append(loss)

    def commit(self, test_error):
        self.elapsed_time = time.time() - self.start
        self.test_error = test_error

    def format(self):
        return '\n'.join(f"{self.index};{self.elapsed_time:.2f};{self.test_error:.4f};{i};{batch_loss:.4f}" for i, batch_loss in enumerate(self.batch_training_losses))
    
    def commit_integer(self, test_accuracy):
        self.elapsed_time = time.time() - self.start
        self.test_accuracy = test_accuracy

    def format_integer(self):
        return '\n'.join(f"{self.index};{self.elapsed_time:.2f};{self.test_accuracy:.4f};{i};{batch_loss:.4f}" for i, batch_loss in enumerate(self.batch_training_losses))
    
def create_out_dirs(model_name):
    Path(f"out/{model_name}").mkdir(parents=True, exist_ok=True)


def write_model(model_name, model, alpha, batch_size):
    with open(os.path.join(f"out/{model_name}/model.txt"), 'w') as file:
        file.write(str(model))
        file.write(f"\nalpha: {alpha}\nbatch_size: {batch_size}\n")


def write_epoch_summary(model_name, epochs: List[EpochSummary]):
    with open(os.path.join(f"out/{model_name}/training.csv"), 'w') as file:
        file.write(
            "epoch;epoch_training_time;epoch_test_error;batch_index;batch_loss\n")
        file.writelines([f"{row.format()}\n" for row in epochs])

def write_epoch_summary_integer(model_name, epochs: List[EpochSummary]):
    with open(os.path.join(f"out/{model_name}/training.csv"), 'w') as file:
        file.write(
            "epoch;epoch_training_time;epoch_test_accuracy;batch_index;batch_loss\n")
        file.writelines([f"{row.format_integer()}\n" for row in epochs])


def calc_error(y, out, invariant_index):
    target = y.view(-1, 7)[:, invariant_index]
    out_target_distance = torch.where(target == 0, torch.abs(
        out - target), torch.abs((out - target)/target))
    return out_target_distance.mean()
