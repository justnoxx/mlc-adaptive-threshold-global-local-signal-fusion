# Script to parse AmazonCat-13K TF-IDF LibSVM format into PyTorch tensors
import os
import bz2
import io
import torch
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer


def parse_amazoncat13k(libsvm_bz2_path, raw_text_bz2_path=None, output_dir="features_amazoncat", chunk_size=50000):
    os.makedirs(output_dir, exist_ok=True)

    print("[✓] Loading sparse TF-IDF features and labels from:", libsvm_bz2_path)
    with bz2.open(libsvm_bz2_path, 'rb') as f:
        X_sparse, y_lists = load_svmlight_file(io.BytesIO(f.read()), multilabel=True)

    print("[✓] Fitting MultiLabelBinarizer")
    mlb = MultiLabelBinarizer()
    mlb.fit(y_lists)

    print("[✓] Transforming and saving labels in chunks to save memory")
    total_rows = 0
    for i in range(0, len(y_lists), chunk_size):
        chunk = y_lists[i:i+chunk_size]
        Y_chunk = torch.tensor(mlb.transform(chunk), dtype=torch.float32)
        torch.save(Y_chunk, os.path.join(output_dir, f"labels_chunk_{i//chunk_size}.pt"))
        total_rows += Y_chunk.size(0)

    print(f"[✓] Saved labels in {total_rows // chunk_size + 1} chunks")

    print("[✓] Skipping tfidf/knn computation until chunk merging or streaming")

    torch.save((X_sparse.data, X_sparse.indices, X_sparse.indptr, X_sparse.shape), os.path.join(output_dir, "features_sparse.pt"))
    torch.save(mlb.classes_, os.path.join(output_dir, "label_classes.pt"))

    print("[✓] Done. You can stream chunks or merge offline for training.")


# Example usage:
# parse_amazoncat13k("amazon-13k/AmazonCat-13K_tfidf_train_ver1.svm.bz2")