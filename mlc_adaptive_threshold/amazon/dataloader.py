# Streaming DataLoader for AmazonCat-13K chunked format
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


class AmazonCatDataset(Dataset):
    def __init__(self, data_dir, chunk_idx):
        self.data_dir = data_dir
        self.chunk_idx = chunk_idx
        self.labels = torch.load(os.path.join(data_dir, f"labels_chunk_{chunk_idx}.pt"))

        # Load sparse TF-IDF structure (ensure correct array types)
        data, indices, indptr, shape = torch.load(os.path.join(data_dir, "features_sparse.pt"))
        data = np.array(data, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)
        indptr = np.array(indptr, dtype=np.int32)
        self.X_sparse = csr_matrix((data, indices, indptr), shape)

        # Track index offset for slicing correct rows
        self.offset = chunk_idx * self.labels.shape[0]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        real_idx = self.offset + idx
        x = self.X_sparse[real_idx].toarray().squeeze().astype(np.float32)
        y = self.labels[idx]
        return torch.from_numpy(x), y


# Example usage:
# dataset = AmazonCatDataset("features_amazoncat", chunk_idx=0)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# for x, y in loader:
#     ...
