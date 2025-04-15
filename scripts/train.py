import torch
from torch.utils.data import DataLoader, random_split
from mlc_adaptive_threshold.amazon.dataloader import AmazonCatDataset
from mlc_adaptive_threshold.amazon.model import MultiLabelWithAdaptiveThreshold, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
label_classes = torch.load("data/label_classes.pt")
input_size = 203_882
num_labels = len(label_classes)
chunk_idx = 0

dataset = AmazonCatDataset("data", chunk_idx=chunk_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

# Improved model
model = MultiLabelWithAdaptiveThreshold(
    input_size=input_size,
    num_labels=num_labels,
    disable_tfidf=False,
    disable_knn=False,
    use_fixed_threshold=False,
    use_bias_init=True,
    use_threshold_bias=True,
    normalize_logits=True,
    use_margin_loss=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=1500,
    output_dir="outputs_amazoncat_improved",
    use_pos_weight=True,
    pos_weight_value=10.0
)
