from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import List
import time
import torch
import sys
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
import random


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
    
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    # Applichiamo una scala logaritmica per visualizzare meglio i valori bassi accanto ai picchi
    cm_log = np.log1p(cm)  # log(1 + x) evita log(0)

    fig, ax = plt.subplots(figsize=(max(10, num_classes // 2), max(8, num_classes // 3)))

    cax = ax.matshow(cm_log, cmap='Blues')
    fig.colorbar(cax)

    # Etichette sugli assi
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (log scale)')

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(np.arange(num_classes), rotation=90)
    ax.set_yticklabels(np.arange(num_classes))

    # Evidenzia la diagonale con un bordo rosso
    for i in range(num_classes):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
        ax.add_patch(rect)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix salvata in: {save_path}")
    else:
        plt.show()

    plt.close()

def compute_class_weights(dataset, invariant_idx, power=0.5):
    counts = defaultdict(int)
    for data in dataset:
        label = int(data.y[invariant_idx].item())
        if label != -1:
            counts[label] += 1

    num_classes = max(counts.keys()) + 1
    freq = torch.zeros(num_classes)
    for cls, count in counts.items():
        freq[cls] = count

    weights = (1.0 / (freq + 1e-6)) ** power
    weights = weights / weights.mean()  # normalizzazione
    return weights

class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, invariant_idx, batch_size, min_examples_per_class=1, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.invariant_idx = invariant_idx
        self.min_examples_per_class = min_examples_per_class
        self.label_to_indices = defaultdict(list)
        for idx, data in enumerate(dataset):
            label = int(data.y[invariant_idx].item())
            if label != -1:
                self.label_to_indices[label].append(idx)
        self.labels_set = sorted(self.label_to_indices.keys())
        self.num_classes = len(self.labels_set)

        print(f"Numero di classi nel dataset: {self.num_classes}")
        print(f"Batch size passato: {self.batch_size}")
        self.samples_per_class = (self.batch_size - self.num_classes * self.min_examples_per_class) // self.num_classes
        if self.samples_per_class < 0:
            raise ValueError(f"Il numero minimo di esempi per classe ({self.min_examples_per_class}) è troppo grande per il batch size ({self.batch_size}).")

    def __iter__(self):
        indices_per_class = {
            label: indices.copy()
            for label, indices in self.label_to_indices.items()
        }
        if self.shuffle:
            for indices in indices_per_class.values():
                random.shuffle(indices)
        batches = []
        while True:
            batch = []
            exhausted_classes = []  # Lista per tracciare le classi esaurite
            for label in self.labels_set:
                indices = indices_per_class[label]
                # Se la classe ha abbastanza campioni, aggiungiamo il minimo richiesto
                if len(indices) >= self.min_examples_per_class:
                    min_samples = indices[:self.min_examples_per_class]
                    indices_per_class[label] = indices[self.min_examples_per_class:]
                else:
                    # Se la classe ha esaurito gli esempi, la segnaliamo come esaurita
                    min_samples = indices[:]
                    indices_per_class[label] = []
                    exhausted_classes.append(label)
                
                batch.extend(min_samples)

            remaining_samples = self.batch_size - len(batch)
            if remaining_samples > 0:
                # Completiamo il batch con campioni casuali da altre classi
                all_indices = []
                for label in self.labels_set:
                    indices = indices_per_class[label]
                    if len(indices) > 0:
                        random.shuffle(indices)
                        all_indices.extend(indices)
                batch.extend(all_indices[:remaining_samples])

            batches.append(batch)

            if len(batches) * self.batch_size >= len(self.dataset):
                break

            for label in exhausted_classes:
                indices_per_class[label] = self.label_to_indices[label].copy()
                if self.shuffle:
                    random.shuffle(indices_per_class[label])

        return iter(batches)

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        return total_samples // self.batch_size
    
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
