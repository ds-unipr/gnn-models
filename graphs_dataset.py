import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected, to_dense_adj
import os
import json
import zipfile

def extract_upper_triangle(data, max_nodes=50):
    n = data.num_nodes
    adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0]
    adj.fill_diagonal_(0)
    
    tril_indices = torch.tril_indices(n, n, offset=-1)
    upper_vec = adj[tril_indices[0], tril_indices[1]]
    
    max_edges = max_nodes * (max_nodes - 1) // 2
    
    padded_vec = torch.zeros(max_edges, dtype=upper_vec.dtype)
    mask = torch.zeros(max_edges, dtype=torch.float)
    
    num_edges = upper_vec.numel()
    
    padded_vec[:num_edges] = upper_vec
    mask[:num_edges] = 1.0
    
    return padded_vec, mask


class RandomUndirectedGraphsDataset(InMemoryDataset):
    url = f'https://www.dropbox.com/scl/fi/z1neoiyzs8pzdciifwvmd/dataset.zip?rlkey=hsvg7uhq65p9skovuc6lcucn7&st=v7w9vqop&dl=0&dl=1'

    raw_zip_file = 'data.zip'
    processed_file = 'data.pt'

    def __init__(self, root, force_reload=False, transform=None, pre_transform=None, plain_vector=False):
        super().__init__(root, transform, pre_transform)
        self.force_reload = force_reload
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.plain_vector = plain_vector
        self.invariants_order = self._data.invariants_order[0] if len(self._data.invariants_order) > 0 else None

    @property
    def raw_file_names(self):
        return [self.raw_zip_file]

    @property
    def processed_file_names(self):
        return [self.processed_file]

    def download(self):
        raw_dir = self.raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        print("Downloading dataset...")
        download_url(self.url, raw_dir, filename=self.raw_zip_file)
        print("Download completed.")

    def process(self):
        raw_zip_path = os.path.join(self.raw_dir, self.raw_zip_file)
        extract_dir = os.path.join(self.raw_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        print(raw_zip_path)
        print("Extracting zip file...")
        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed.")

        data_list = []
        invariants_order = None
        for root_dir, _, files in os.walk(extract_dir):
            count = 1
            for file in files:
                if file.lower().endswith('.json'):
                    json_path = os.path.join(root_dir, file)
                    print(
                        f"Reading {json_path}\t {count}/{len(files)}", end='\r')
                    count += 1
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")
                        continue
                    nodes_count = json_data.get("nodes_count", None)
                    invariants_order = json_data.get("invariants_order", [])

                    for graph in json_data.get("graph_list", []):
                        num_nodes = nodes_count if nodes_count is not None else len(
                            graph.get("nodes", []))
                        x = torch.ones((num_nodes, 1), dtype=torch.float)
                        edges = graph.get("edges", [])
                        if len(edges) > 0:
                            edge_index = torch.tensor(
                                edges, dtype=torch.long).t().contiguous()
                        else:
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_index = to_undirected(edge_index)
                        invarinats_values = torch.tensor(
                            graph.get("graph_features", []), dtype=torch.float)
                        data = Data(x=x, edge_index=edge_index,
                                    y=invarinats_values)
                        data.invariants_order = invariants_order
                        (vec, mask) = extract_upper_triangle(data, max_nodes=50)
                        data.vectorized_representation = vec
                        data.mask = mask
                        data_list.append(data)

        data, slices = self.collate(data_list)
        self.invariants_order = invariants_order
        torch.save((data, slices), self.processed_paths[0])
        print("Processing completed.")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.plain_vector == True:
            vec = item.vectorized_representation
            mask = item.mask
            label = item.y
            return torch.cat([vec, mask], dim=0), label
        return item