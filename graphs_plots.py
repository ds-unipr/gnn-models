import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np 
import os

def plot_single_model(model_name):
    
    save_folder = os.path.join("out", model_name)
    csv_files = [f for f in os.listdir(save_folder) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the specified folder.")
    csv_path = os.path.join(save_folder, csv_files[0]) 
    df = pd.read_csv(csv_path, delimiter=';')

    epochs = df['epoch'].unique()
    training_time = df['epoch_training_time']
    batch_indices = df['batch_index']
    batch_losses = df['batch_loss']
    if 'epoch_test_error' in df.columns:
        test_errors = df.groupby('epoch')['epoch_test_error'].mean()
    else:
        test_accuracy = df.groupby('epoch')['epoch_test_accuracy'].mean()
    df["batch_index_continuo"] = df["batch_index"] + df["epoch"] * df["batch_index"].max()
    max_batch_index = df["batch_index_continuo"].max()

    os.makedirs(save_folder, exist_ok=True)
    pdf_path = os.path.join(save_folder, "model_plots.pdf")
    fig, axes = plt.subplots(3, 1, figsize=(25, 18))

    # GRAPH 1: Training Loss 
    transformed_losses = batch_losses
    epoch_start_indices = df[df['batch_index'] == 0].index 
    axes[0].plot(df["batch_index_continuo"].iloc[epoch_start_indices], transformed_losses.iloc[epoch_start_indices], 
                 label="Training Loss (epoch)", linestyle='None', marker='o', color='blue', markersize=4)
    axes[0].plot(df["batch_index_continuo"][::48], transformed_losses[::48], 
                 label="Training Loss", linestyle='-', alpha=0.7)
    
    axes[0].set_title("Training Loss over Batches View", fontsize=20)
    axes[0].set_xlabel("Batch Index", fontsize=18)
    axes[0].set_ylabel("Loss", fontsize=18)
    xticks_step = max_batch_index // 10
    axes[0].set_xticks(np.arange(0, max_batch_index + 1, xticks_step))
    #axes[0].set_yticks(np.arange(0, 10, 1))  # Tick ogni 0.5
    #axes[0].set_ylim(0, 10)
    axes[0].legend(fontsize=16)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # GRAPH 2: Training Loss (detail)
    transformed_losses = np.clip(batch_losses, None, 1.5)  
    axes[1].plot(df["batch_index_continuo"].iloc[epoch_start_indices], transformed_losses.iloc[epoch_start_indices], 
                 label="Training Loss (epoch)", linestyle='None', marker='o', color='blue', markersize=4)
    x_points = []
    y_points = []   
    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        batch_indices = epoch_data['batch_index_continuo'].values
        batch_losses = epoch_data['batch_loss'].values
        
        x_points.append(batch_indices[0])
        y_points.append(batch_losses[0])
        
        for i in range(48, len(batch_indices), 48):
            x_points.append(batch_indices[i])
            y_points.append(batch_losses[i])

    if 'test_errors' in locals() and test_errors is not None:
        axes[1].plot(x_points, y_points, label="Training Loss", linestyle='-', alpha=0.7)
        axes[1].set_title("Training Loss over Batches (Detail)", fontsize=20)
        axes[1].set_xlabel("Batch Index", fontsize=18)
        axes[1].set_ylabel("Loss", fontsize=18)
        xticks_step = max_batch_index // 10
        axes[1].set_xticks(np.arange(0, max_batch_index + 1, xticks_step))
        axes[1].set_yticks(np.arange(0, 0.035, 0.01))
        axes[1].set_ylim(0, 0.03)
        axes[1].legend(fontsize=16)
        axes[1].grid(True, linestyle='--', alpha=0.5)
    else:
        axes[1].plot(x_points, y_points, label="Training Loss", linestyle='-', alpha=0.7)
        axes[1].set_title("Training Loss over Batches (Detail)", fontsize=20)
        axes[1].set_xlabel("Batch Index", fontsize=18)
        axes[1].set_ylabel("Loss", fontsize=18)
        xticks_step = max_batch_index // 10
        axes[1].set_xticks(np.arange(0, max_batch_index + 1, xticks_step))
        axes[1].set_yticks(np.arange(0, 1.55, 0.2))
        axes[1].set_ylim(0, 1.5)
        axes[1].legend(fontsize=16)
        axes[1].grid(True, linestyle='--', alpha=0.5)

    # GRAPH 3: Test Error
    if 'test_errors' in locals() and test_errors is not None:
        axes[2].plot(epochs, test_errors, label="Test Error", marker='s', linestyle='-')
        axes[2].set_title("Test Error Over Epochs", fontsize=20)
        axes[2].set_xlabel("Epochs", fontsize=18)
        axes[2].set_ylabel("Error", fontsize=18)
        axes[2].set_xticks(np.arange(0, max(epochs) + 1, 5))  # Tick ogni 5 epoch
        axes[2].legend(fontsize=16)
        axes[2].grid(True, linestyle='--', alpha=0.5)
    else:
        # Se test_errors non è definito, usa test_accuracy
        axes[2].plot(epochs, test_accuracy, label="Test Accuracy", marker='s', linestyle='-')
        axes[2].set_title("Test Accuracy Over Epochs", fontsize=20)
        axes[2].set_xlabel("Epochs", fontsize=18)
        axes[2].set_ylabel("Accuracy", fontsize=18)
        axes[2].set_xticks(np.arange(0, max(epochs) + 1, 5))  # Tick ogni 5 epoch
        axes[2].legend(fontsize=16)
        axes[2].grid(True, linestyle='--', alpha=0.5)

    # Save in one PDF
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf')
    plt.close()


def plot_selected_models(model_names):
    base_folder = "out"
    save_folder = os.path.join(base_folder, "graph_plots")  # Save in "out/graph_plots"
    os.makedirs(save_folder, exist_ok=True) 

    model_short_names = "_".join([name.removeprefix("model") for name in model_names])
    pdf_filename = f"compare_{model_short_names}.pdf"
    pdf_path = os.path.join(save_folder, pdf_filename)
    max_batch_index = 0

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(4, 1, figsize=(25, 24))
        
        # GRAPH 1: Plot Training Loss
        for model_name in model_names:
            model_folder = os.path.join(base_folder, model_name)
            if os.path.isdir(model_folder):
                csv_files = [f for f in os.listdir(model_folder) if f.endswith('.csv')]
                for file in csv_files:
                    df = pd.read_csv(os.path.join(model_folder, file), delimiter=';')
                    df["batch_index_continuo"] = df["batch_index"] + df["epoch"] * df["batch_index"].max()
                    max_batch_index = max(max_batch_index, df["batch_index_continuo"].max())
                    batch_indices = df["batch_index_continuo"]
                    batch_losses = df["batch_loss"]
                    axes[0].plot(batch_indices, batch_losses, label=f"{model_name}", linestyle='-')
        axes[0].set_xlabel("Batch Index", fontsize=18)
        axes[0].set_ylabel("Training Loss", fontsize=18)
        xticks_step = max_batch_index // 10 
        axes[0].set_xticks(np.arange(0, max_batch_index + 1, xticks_step))
        axes[0].set_title("Comparison of Training Loss Across Selected Models", fontsize=22)
        axes[0].legend(fontsize=18)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # GRAPH 2: Plot Training Loss detail
        for model_name in model_names:
            model_folder = os.path.join(base_folder, model_name)
            if os.path.isdir(model_folder):
                csv_files = [f for f in os.listdir(model_folder) if f.endswith('.csv')]
                for file in csv_files:
                    df = pd.read_csv(os.path.join(model_folder, file), delimiter=';')
                    df["batch_index_continuo"] = df["batch_index"] + df["epoch"] * df["batch_index"].max()
                    max_batch_index = max(max_batch_index, df["batch_index_continuo"].max())
                    batch_indices = df["batch_index_continuo"]
                    batch_losses = df["batch_loss"]
                    if 'epoch_test_error' in df.columns:
                        transformed_losses = np.clip(batch_losses, None, 0.02)
                    else:
                        transformed_losses = np.clip(batch_losses, None, 1.5)  
                    axes[1].plot(batch_indices[::48], transformed_losses[::48], label=f"{model_name}", linestyle='-')
        axes[1].set_xlabel("Batch Index", fontsize=18)
        axes[1].set_ylabel("Training Loss", fontsize=18)
        axes[1].set_title("Comparison of Training (detail) Loss Across Selected Models", fontsize=22)
        xticks_step = max_batch_index // 10  
        axes[1].set_xticks(np.arange(0, max_batch_index + 1, xticks_step))
        if 'epoch_test_error' in df.columns:
            axes[1].set_yticks(np.arange(0, 0.025, 0.005)) 
            axes[1].set_ylim(0, 0.02)
        else:
            axes[1].set_yticks(np.arange(0, 1.55, 0.1)) 
            axes[1].set_ylim(0, 1.5)
        axes[1].legend(fontsize=18)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # GRAPH 3: Plot Test Error
        for model_name in model_names:
            model_folder = os.path.join(base_folder, model_name)
            if os.path.isdir(model_folder):
                csv_files = [f for f in os.listdir(model_folder) if f.endswith('.csv')]
                for file in csv_files:
                    df = pd.read_csv(os.path.join(model_folder, file), delimiter=';')
                    epochs = df["epoch"].unique()
                    if 'epoch_test_error' in df.columns:
                        test_errors = df.groupby('epoch')['epoch_test_error'].mean()
                        axes[2].plot(epochs, test_errors, label=f"{model_name}", linestyle='-')
                    else:
                        test_accuracy = df.groupby('epoch')['epoch_test_accuracy'].mean()
                        axes[2].plot(epochs, test_accuracy, label=f"{model_name}", linestyle='-')
        axes[2].set_xlabel("Epochs", fontsize=18)
        axes[2].set_title("Comparison of Test Errors Across Selected Models", fontsize=22)
        if 'test_errors' in locals() and test_errors is not None:
            axes[2].set_ylabel("Test Error", fontsize=18)
        else:
            axes[2].set_ylabel("Test Accuracy", fontsize=18)
        axes[2].set_xticks(np.arange(0, max(epochs) + 2, 5)) 
        axes[2].legend(fontsize=18)
        axes[2].grid(True, linestyle='--', alpha=0.5)

        # GRAPH 4: Plot Test Error detail
        for model_name in model_names:
            model_folder = os.path.join(base_folder, model_name)
            if os.path.isdir(model_folder):
                csv_files = [f for f in os.listdir(model_folder) if f.endswith('.csv')]
                for file in csv_files:
                    df = pd.read_csv(os.path.join(model_folder, file), delimiter=';')
                    epochs = df["epoch"].unique()
                    if 'epoch_test_error' in df.columns:
                        test_errors = df.groupby('epoch')['epoch_test_error'].mean()
                        transformed_errors = np.clip(test_errors, None, 0.1) 
                        axes[3].plot(epochs, transformed_errors, label=f"{model_name}", linestyle='-')
                    else:
                        test_accuracy = df.groupby('epoch')['epoch_test_accuracy'].mean()
                        transformed_errors = np.clip(test_accuracy, 0.7, 1) 
                        axes[3].plot(epochs, transformed_errors, label=f"{model_name}", linestyle='-')
        axes[3].set_xlabel("Epochs", fontsize=18)
        axes[3].set_title("Comparison of Test Errors (detail) Across Selected Models", fontsize=22)
        axes[3].set_xticks(np.arange(0, max(epochs) + 2, 5))
        if 'test_errors' in locals() and test_errors is not None:
            axes[3].set_ylabel("Test Error", fontsize=18)
            axes[3].set_yticks(np.arange(0, 0.1, 0.02))  
            axes[3].set_ylim(0, 0.1)
        else:
            axes[3].set_ylabel("Test Accuracy", fontsize=18)
            axes[3].set_yticks(np.arange(0.65, 1.1, 0.05))  
            axes[3].set_ylim(0.7, 1) 
        axes[3].legend(fontsize=18)
        axes[3].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)