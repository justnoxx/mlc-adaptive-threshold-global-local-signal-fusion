import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os


class MultiLabelWithAdaptiveThreshold(nn.Module):
    def __init__(self, input_size, num_labels, hidden_size=2048,
                 disable_tfidf=False, disable_knn=False, use_fixed_threshold=False,
                 use_bias_init=True, use_threshold_bias=True,
                 normalize_logits=True, use_margin_loss=True):
        super().__init__()
        self.num_labels = num_labels
        self.disable_tfidf = disable_tfidf
        self.disable_knn = disable_knn
        self.use_fixed_threshold = use_fixed_threshold
        self.normalize_logits = normalize_logits
        self.use_margin_loss = use_margin_loss

        self.base_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(hidden_size, num_labels)
        if use_bias_init:
            self.classifier.bias.data.fill_(0.5)

        self.alpha = nn.Parameter(torch.randn(num_labels))
        self.beta = nn.Parameter(torch.randn(num_labels))
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.threshold_bias = nn.Parameter(torch.zeros(num_labels)) if use_threshold_bias else None

    def threshold_layer(self, tfidf, knn):
        if self.use_fixed_threshold:
            return torch.full_like(tfidf, 0.5)

        w_tfidf = 0 if self.disable_tfidf else self.alpha * tfidf
        w_knn = 0 if self.disable_knn else self.beta * knn
        threshold = self.lam * w_tfidf + (1 - self.lam) * w_knn
        if self.threshold_bias is not None:
            threshold = threshold + self.threshold_bias
        return torch.sigmoid(threshold)

    def forward(self, x, tfidf=None, knn=None):
        h = self.base_model(x)
        logits = self.classifier(h)
        if self.normalize_logits:
            logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-8)

        if tfidf is None:
            tfidf = torch.zeros_like(logits)
        if knn is None:
            knn = torch.zeros_like(logits)

        theta = self.threshold_layer(tfidf, knn)
        adjusted_logits = logits - theta
        return adjusted_logits, logits, theta


def compute_metrics(preds, targets):
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'macro/precision': precision.mean().item(),
        'macro/recall': recall.mean().item(),
        'macro/f1': f1.mean().item(),
        'micro/precision': tp.sum().item() / (tp + fp).sum().item(),
        'micro/recall': tp.sum().item() / (tp + fn).sum().item(),
        'micro/f1': 2 * tp.sum().item() / (2 * tp.sum().item() + fp.sum().item() + fn.sum().item()),
    }


def train_model(model, train_loader, val_loader, optimizer, device, epochs=20,
                output_dir="outputs_improved", use_pos_weight=True, pos_weight_value=10.0):
    os.makedirs(output_dir, exist_ok=True)
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value).to(device) if use_pos_weight else None
    )
    best_f1 = 0
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            tfidf = y.float() / (y.sum(dim=1, keepdim=True) + 1e-8)
            tfidf = tfidf.nan_to_num(0.0)

            knn = torch.matmul(y.float(), y.float().T)
            knn = knn / (y.sum(dim=1, keepdim=True) + 1e-8)
            knn = torch.matmul(knn, y.float())
            knn = knn.nan_to_num(0.0)

            x, y, tfidf, knn = x.to(device), y.to(device), tfidf.to(device), knn.to(device)

            optimizer.zero_grad()
            adjusted_logits, raw_logits, thresholds = model(x, tfidf, knn)

            loss = bce_loss_fn(adjusted_logits, y)
            if model.use_margin_loss:
                margin_loss = F.margin_ranking_loss(
                    raw_logits.view(-1),
                    thresholds.view(-1),
                    (2 * y - 1).view(-1).float(),
                    margin=0.1
                )
                loss += 0.1 * margin_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                tfidf = y.float() / (y.sum(dim=1, keepdim=True) + 1e-8)
                tfidf = tfidf.nan_to_num(0.0)

                knn = torch.matmul(y.float(), y.float().T)
                knn = knn / (y.sum(dim=1, keepdim=True) + 1e-8)
                knn = torch.matmul(knn, y.float())
                knn = knn.nan_to_num(0.0)

                x, y, tfidf, knn = x.to(device), y.to(device), tfidf.to(device), knn.to(device)
                adjusted_logits, _, _ = model(x, tfidf, knn)
                preds = (torch.sigmoid(adjusted_logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(preds, targets)

        positive_ratio = preds.sum().item() / preds.numel()
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Macro F1: {metrics['macro/f1']:.4f} | Pos %: {positive_ratio:.4f}")

        history.append({"epoch": epoch, "loss": avg_loss, "positive_ratio": positive_ratio, **metrics})

        stats_path = os.path.join(output_dir, "threshold_weights.json")
        log_entry = {
            "epoch": epoch,
            "lambda": model.lam.item(),
            "alpha_mean": model.alpha.mean().item(),
            "beta_mean": model.beta.mean().item(),
            "alpha_std": model.alpha.std().item(),
            "beta_std": model.beta.std().item(),
        }
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                all_stats = json.load(f)
        else:
            all_stats = []
        all_stats.append(log_entry)
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

        if metrics['macro/f1'] > best_f1:
            best_f1 = metrics['macro/f1']
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(output_dir, "best_model.pt"))

    if best_f1 > 0:
        model.load_state_dict(best_state)
        print(f"[✓] Loaded best model state with F1: {best_f1:.4f}")

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
