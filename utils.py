from pathlib import Path
import os
from typing import List
import time


class EpochSummary():
    def __init__(self, index):
        self.index = index
        self.start = time.time()
        self.batch_training_losses = []
        self.test_loss = 0

    def add_trainging_loss(self, loss):
        self.batch_training_losses.append(loss)

    def commit(self, test_loss):
        self.elapsed_time = time.time() - self.start
        self.test_loss = test_loss

    def format(self):
        return '\n'.join(f"{self.index};{self.elapsed_time:.2f};{self.test_loss:.4f};{i};{batch_loss:.4f}" for i, batch_loss in enumerate(self.batch_training_losses))


def create_out_dirs(model_name):
    Path(f"out/{model_name}").mkdir(parents=True, exist_ok=True)

def write_model(model_name, model, alpha, batch_size, hidden_channels):
    with open(os.path.join(f"out/{model_name}/model.txt"), 'w') as file:
        file.write(str(model))
        file.write(f"\nalpha: {alpha}\nbatch_size: {batch_size}\nhidden_channels: {hidden_channels}")

def write_epoch_summary(model_name, epochs: List[EpochSummary]):
    with open(os.path.join(f"out/{model_name}/model.csv"), 'w') as file:
        file.write("epoch;elapsed_time;test_loss;batch_index;batch_loss\n")
        file.writelines([f"{row.format()}\n" for row in epochs])