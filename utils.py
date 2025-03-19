from pathlib import Path
import os
from typing import List
import time


class EpochSummary():
    def __init__(self, index):
        self.index = index
        self.start = time.time()
        self.batch_training_losses = []
        self.train_error = 0
        self.test_error = 0

    def add_trainging_loss(self, loss):
        self.batch_training_losses.append(loss)

    def commit(self, train_error, test_error):
        self.elapsed_time = time.time() - self.start
        self.test_error = test_error
        self.train_error = train_error

    def format(self):
        return '\n'.join(f"{self.index};{self.elapsed_time:.2f};{self.test_error:.4f};{self.test_error:.4f};{i};{batch_loss:.4f}" for i, batch_loss in enumerate(self.batch_training_losses))


def create_out_dirs(model_name):
    Path(f"out/{model_name}").mkdir(parents=True, exist_ok=True)

def write_model(model_name, model, alpha, batch_size):
    with open(os.path.join(f"out/{model_name}/model.txt"), 'w') as file:
        file.write(str(model))
        file.write(f"\nalpha: {alpha}\nbatch_size: {batch_size}\n")

def write_epoch_summary(model_name, epochs: List[EpochSummary]):
    with open(os.path.join(f"out/{model_name}/training.csv"), 'w') as file:
        file.write("epoch;epoch_training_time;epoch_training_error;epoch_test_error;batch_index;batch_loss\n")
        file.writelines([f"{row.format()}\n" for row in epochs])