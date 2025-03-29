import torch
import torch.nn as nn
import torch.optim as optim

class GraphMLP(nn.Module):
    def __init__(self, max_num_edges=1225, hidden_dim=128, output_dim=1):
        super(GraphMLP, self).__init__()
        
        input_dim = 2 * max_num_edges  # adjacency + mask
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, adjacency_vector, mask_vector):
        """
        adjacency_vector: (batch_size, max_num_edges)
        mask_vector:      (batch_size, max_num_edges)
        """
        # Concatenate adjacency and mask along dim=1
        x = torch.cat([adjacency_vector, mask_vector], dim=1)  # shape (batch_size, 2*max_num_edges)
        out = self.mlp(x)
        return out

# Example usage
if __name__ == "__main__":
    # Let's assume we have up to 50 nodes, so the upper triangle has 50*(50-1)/2 = 1225 possible edges
    max_num_edges = 1225
    batch_size = 4   # example batch size
    hidden_dim = 128
    output_dim = 1   # for regression or binary classification, e.g.

    # Create random adjacency and mask for demonstration
    # adjacency_vector in {0,1}, shape = (batch_size, max_num_edges)
    adjacency_vector = torch.randint(0, 2, (batch_size, max_num_edges)).float()

    # mask_vector in {0,1}, shape = (batch_size, max_num_edges)
    # e.g. 1 for valid positions, 0 for padding
    mask_vector = torch.randint(0, 2, (batch_size, max_num_edges)).float()

    # Suppose we have a target label for each graph (for example, regression or classification)
    # shape = (batch_size,) or (batch_size,1)
    targets = torch.randn(batch_size, 1)

    # Instantiate the model
    model = GraphMLP(max_num_edges=max_num_edges, hidden_dim=hidden_dim, output_dim=output_dim)

    # Define a loss and an optimizer
    criterion = nn.MSELoss()  # or nn.BCEWithLogitsLoss() if binary classification
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass
    outputs = model(adjacency_vector, mask_vector)
    loss = criterion(outputs, targets)

    # Backprop
    loss.backward()
    optimizer.step()

    print(f"Outputs shape: {outputs.shape}, Loss: {loss.item():.4f}")