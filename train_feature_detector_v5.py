#!/usr/bin/env python
"""
Training script for Feature Detector V5 (Enhanced Architecture with Transformer).

Key improvements from V4:
1. Deeper GNN encoder (6 layers)
2. Transformer decoder with multi-head attention
3. Larger hidden dimension (384)
4. Focal loss for type prediction
5. Cosine annealing scheduler with warmup
6. Adaptive loss weighting
7. Better parameter importance weighting
"""
import argparse
import time
import json
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import matplotlib.pyplot as plt

from feature_detector_v5 import FeatureDetectorV5


class FeatureDataset(Dataset):
    """Dataset for feature detection."""

    def __init__(self, data_list, normalize_stats):
        self.data_list = data_list
        self.normalize_stats = normalize_stats

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Normalize node and edge features (keep on CPU)
        x = (data['x'] - self.normalize_stats['node_mean'].cpu()) / self.normalize_stats['node_std'].cpu()
        # Clip to prevent extreme values from causing NaN losses
        x = torch.clamp(x, -10.0, 10.0)

        if data['edge_attr'].size(0) > 0:
            edge_attr = (data['edge_attr'] - self.normalize_stats['edge_mean'].cpu()) / self.normalize_stats['edge_std'].cpu()
            edge_attr = torch.clamp(edge_attr, -10.0, 10.0)
        else:
            edge_attr = data['edge_attr']

        # Create PyG Data
        graph = Data(
            x=x,
            edge_index=data['edge_index'],
            edge_attr=edge_attr
        )

        # Features: [num_features x (9 types + 10 params)]
        features = []
        for feat in data['features'][:-1]:  # Exclude EOS token!
            feature_vec = torch.zeros(19)
            feature_vec[feat['type']] = 1.0  # One-hot type
            # Normalize parameters before adding to feature vector
            params = torch.from_numpy(feat['params']).float()
            params_norm = (params - self.normalize_stats['param_mean'].cpu()) / self.normalize_stats['param_std'].cpu()
            # Clip to prevent extreme values
            params_norm = torch.clamp(params_norm, -10.0, 10.0)
            feature_vec[9:] = params_norm
            features.append(feature_vec)

        if len(features) > 0:
            features = torch.stack(features)
        else:
            features = torch.zeros(1, 19)  # Empty sample

        # Count (exclude EOS)
        count = len(data['features']) - 1

        return graph, features, count


def collate_fn(batch):
    """Collate function for DataLoader."""
    graphs, features, counts = zip(*batch)

    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)

    # Pad features to same length
    max_features = max(f.size(0) for f in features)
    max_features = max(max_features, 1)  # At least 1

    batched_features = []
    masks = []

    for feat in features:
        num_feat = feat.size(0)
        if num_feat < max_features:
            # Pad
            padding = torch.zeros(max_features - num_feat, 19)
            feat_padded = torch.cat([feat, padding], dim=0)
        else:
            feat_padded = feat[:max_features]

        # Mask
        mask = torch.zeros(max_features)
        mask[:min(num_feat, max_features)] = 1.0

        batched_features.append(feat_padded)
        masks.append(mask)

    batched_features = torch.stack(batched_features)
    masks = torch.stack(masks)
    counts = torch.tensor(counts, dtype=torch.long)

    return batched_graph, batched_features, masks, counts


class FeatureLossV5(nn.Module):
    """
    Enhanced loss for V5 architecture.

    Key improvements from V4:
    - Focal loss for type prediction (handles class imbalance)
    - Adaptive loss weighting (weights adjust during training)
    - Parameter importance weighting (diameter, depth more important)
    - Label smoothing for count loss
    """

    def __init__(self, count_weight=10.0, type_weight=15.0, param_weight=5.0, label_smoothing=0.1,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.base_count_weight = count_weight
        self.base_type_weight = type_weight
        self.base_param_weight = param_weight
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Parameter importance weights (diameter, depth, width, length, height, bbox_x/y/z, volume, confidence)
        # Critical dimensions (diameter, depth) get 2x weight, others get 1x
        self.param_importance = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5])

    def count_loss_with_label_smoothing(self, logits, targets):
        """
        Cross-entropy with label smoothing to prevent overconfidence.

        Instead of: [0, 0, 0, 0, 1, 0, ...] (one-hot)
        Use: [ε, ε, ε, ε, 1-ε, ε, ...] (smoothed)
        """
        confidence = 1.0 - self.label_smoothing
        log_probs = F.log_softmax(logits, dim=-1)

        # NLL loss (standard cross-entropy)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Smooth loss (encourage uniform distribution)
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()

    def focal_loss_with_alpha(self, logits, targets, gamma=2.0, alpha=0.25):
        """
        Focal loss with alpha balancing for class imbalance.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            logits: [N, C] class logits
            targets: [N] class indices
            gamma: focusing parameter (higher = focus more on hard examples)
            alpha: balancing parameter for class weights
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** gamma
        focal_loss = alpha * focal_weight * ce_loss
        return focal_loss.mean()

    def forward(self, count_logits, type_logits, param_preds, features, masks, counts, param_mean, param_std):
        """
        Args:
            count_logits: [batch x (max_count+1)]
            type_logits: [batch x max_features x 9]
            param_preds: [batch x max_features x 10]
            features: [batch x max_features x 19]
            masks: [batch x max_features]
            counts: [batch]
            param_mean, param_std: for normalization
        """
        batch_size, max_features = masks.shape

        # 1. Count loss (with label smoothing)
        count_loss = self.count_loss_with_label_smoothing(count_logits, counts)

        # 2. Type loss (FOCAL LOSS for better class imbalance handling)
        gt_types = torch.argmax(features[:, :, :9], dim=-1)
        type_logits_flat = type_logits.reshape(-1, 9)
        gt_types_flat = gt_types.reshape(-1)

        # Compute focal loss
        ce_loss = F.cross_entropy(type_logits_flat, gt_types_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        type_loss_per_elem = self.focal_alpha * focal_weight * ce_loss

        # Apply mask
        type_loss = (type_loss_per_elem.reshape(batch_size, max_features) * masks).sum() / masks.sum()

        # 3. Parameter loss (MSE with masking for non-zero params + importance weighting)
        # Note: gt_params are ALREADY normalized and clipped in __getitem__
        gt_params_norm = features[:, :, 9:]

        # Mask for non-zero parameters (use original params before normalization for masking)
        # We need to denormalize to check if original value was zero
        gt_params_denorm = gt_params_norm * param_std + param_mean
        param_mask = (torch.abs(gt_params_denorm) > 1e-6).float()

        # Per-parameter MSE
        param_errors = F.mse_loss(param_preds, gt_params_norm, reduction='none')

        # Apply parameter importance weights (diameter, depth more important than others)
        importance_weights = self.param_importance.to(param_errors.device).view(1, 1, -1)
        param_errors_weighted = param_errors * importance_weights

        # Apply both feature mask and parameter mask
        feature_mask_expanded = masks.unsqueeze(-1).expand_as(param_errors_weighted)
        combined_mask = feature_mask_expanded * param_mask

        # Compute masked loss
        param_loss = (param_errors_weighted * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        # Total loss (with base weights)
        total_loss = (
            self.base_count_weight * count_loss +
            self.base_type_weight * type_loss +
            self.base_param_weight * param_loss
        )

        return {
            'total': total_loss,
            'count': count_loss.item(),
            'type': type_loss.item(),
            'param': param_loss.item()
        }

    def set_adaptive_weights(self, epoch, max_epochs):
        """
        Adjust loss weights dynamically during training.

        Count weight increases over time to ensure good count prediction first.
        """
        # Count weight increases from 10.0 to 15.0 over training
        progress = epoch / max_epochs
        self.count_weight = self.base_count_weight * (1.0 + 0.5 * progress)
        self.type_weight = self.base_type_weight
        self.param_weight = self.base_param_weight


def train_epoch(model, loader, criterion, optimizer, device, param_mean, param_std):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    count_loss = 0
    type_loss = 0
    param_loss = 0

    for graph, features, masks, counts in tqdm(loader, desc="Training", leave=False):
        graph = graph.to(device)
        features = features.to(device)
        masks = masks.to(device)
        counts = counts.to(device)

        # Forward
        count_logits, type_logits, param_preds = model(graph, features, counts)

        # Loss
        loss_dict = criterion(count_logits, type_logits, param_preds, features, masks, counts, param_mean, param_std)

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['total'].item()
        count_loss += loss_dict['count']
        type_loss += loss_dict['type']
        param_loss += loss_dict['param']

    n = len(loader)
    return {
        'total': total_loss / n,
        'count': count_loss / n,
        'type': type_loss / n,
        'param': param_loss / n
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, param_mean, param_std):
    """Evaluate model."""
    # Use training mode for loss computation (uses ground truth counts)
    model.train()

    total_loss = 0
    count_loss = 0
    type_loss = 0
    param_loss = 0

    for graph, features, masks, counts in tqdm(loader, desc="Evaluating (loss)", leave=False):
        graph = graph.to(device)
        features = features.to(device)
        masks = masks.to(device)
        counts = counts.to(device)

        # Forward (with ground truth counts for consistent loss)
        count_logits, type_logits, param_preds = model(graph, features, counts)

        # Loss
        loss_dict = criterion(count_logits, type_logits, param_preds, features, masks, counts, param_mean, param_std)

        total_loss += loss_dict['total'].item()
        count_loss += loss_dict['count']
        type_loss += loss_dict['type']
        param_loss += loss_dict['param']

    # Now eval mode for accuracy (uses predicted counts)
    model.eval()

    count_correct = 0
    type_correct = 0
    total_counts = 0
    total_types = 0

    for graph, features, masks, counts in tqdm(loader, desc="Evaluating (acc)", leave=False):
        graph = graph.to(device)
        features = features.to(device)
        masks = masks.to(device)
        counts = counts.to(device)

        # Forward (with predicted counts)
        count_logits, type_logits, param_preds = model(graph)

        # Count accuracy
        pred_counts = torch.argmax(count_logits, dim=-1)
        count_correct += (pred_counts == counts).sum().item()
        total_counts += counts.size(0)

        # Type accuracy (only compare up to min length)
        gt_types = torch.argmax(features[:, :, :9], dim=-1)
        pred_types = torch.argmax(type_logits, dim=-1)

        # Compare position by position up to ground truth length
        for i in range(features.size(0)):
            gt_count = counts[i].item()
            valid_mask = masks[i]

            # Only compare valid positions
            for t in range(min(pred_types.size(1), gt_count)):
                if valid_mask[t] > 0:
                    if pred_types[i, t] == gt_types[i, t]:
                        type_correct += 1
                    total_types += 1

    n = len(loader)
    return {
        'loss': {
            'total': total_loss / n,
            'count': count_loss / n,
            'type': type_loss / n,
            'param': param_loss / n
        },
        'count_accuracy': count_correct / total_counts,
        'type_accuracy': type_correct / total_types if total_types > 0 else 0.0
    }


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience=25, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--count-weight', type=float, default=10.0, help='Weight for count loss')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing for count loss')
    parser.add_argument('--use-focal', action='store_true', help='Use focal loss for count')
    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE DETECTOR V4 TRAINING (COUNT-FIRST + MODE COLLAPSE FIXES)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print(f"Count weight: {args.count_weight}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Use focal loss: {args.use_focal}")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load dataset
    print(f"\nLoading dataset...")
    dataset_dict = torch.load(Path(args.data) / 'dataset.pt', weights_only=False)
    data_list = dataset_dict['data']
    normalize_stats = {k: v.to(args.device) for k, v in dataset_dict['normalize_stats'].items()}
    feature_types = dataset_dict['feature_types']

    print(f"Loaded {len(data_list)} samples")

    # Split into train/val/test
    n = len(data_list)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    torch.manual_seed(42)
    indices = torch.randperm(n).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create dataloaders
    train_dataset = FeatureDataset(train_data, normalize_stats)
    val_dataset = FeatureDataset(val_data, normalize_stats)
    test_dataset = FeatureDataset(test_data, normalize_stats)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create model
    print(f"\nCreating model...")
    node_feature_dim = normalize_stats['node_mean'].size(0)
    edge_feature_dim = normalize_stats['edge_mean'].size(0)

    model = FeatureDetectorV5(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=384,  # Increased from 256 for better capacity
        num_gnn_layers=6,  # Increased from 4 for deeper encoding
        num_transformer_layers=4,  # Transformer decoder instead of LSTM
        num_feature_types=9,
        num_params=10,
        max_count=30,  # Increased from 20 to handle 10k dataset (max 24 features)
        nhead=8  # Multi-head attention with 8 heads
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Disable Flash Attention to fix "illegal instruction" CUDA errors
    # Force use of math backend which is more stable across hardware
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("Using MATH SDPA backend (Flash Attention disabled for stability)")

    # Loss and optimizer
    criterion = FeatureLossV5(
        count_weight=10.0,  # Base weight for count loss (will adapt to 15.0)
        type_weight=15.0,  # Higher weight for type loss (using focal loss)
        param_weight=5.0,  # Weight for parameter loss (with importance weighting)
        label_smoothing=0.1,  # Label smoothing for better generalization
        focal_gamma=2.0,  # Focal loss gamma (down-weights easy examples)
        focal_alpha=0.25  # Focal loss alpha (balances positive/negative)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the cycle length after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=25)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device,
                                    normalize_stats['param_mean'], normalize_stats['param_std'])

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device,
                              normalize_stats['param_mean'], normalize_stats['param_std'])

        # Update scheduler and adaptive loss weights
        scheduler.step()
        criterion.set_adaptive_weights(epoch - 1, args.epochs)  # epoch-1 for 0-based indexing

        # Clear CUDA cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print
        print(f"Train - Loss: {train_metrics['total']:.4f} "
              f"(Count: {train_metrics['count']:.4f}, Type: {train_metrics['type']:.4f}, Param: {train_metrics['param']:.4f})")
        print(f"Val   - Loss: {val_metrics['loss']['total']:.4f}, "
              f"Count Acc: {100*val_metrics['count_accuracy']:.1f}%, "
              f"Type Acc: {100*val_metrics['type_accuracy']:.1f}%")

        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model
        if val_metrics['loss']['total'] < best_val_loss:
            best_val_loss = val_metrics['loss']['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'normalize_stats': normalize_stats,
                'feature_types': feature_types,
                'feature_type_to_idx': dataset_dict['feature_type_to_idx']
            }, output_dir / 'best_model.pt')
            print(f"  [OK] Saved best model")

        # Early stopping
        early_stopping(val_metrics['loss']['total'])
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, args.device,
                           normalize_stats['param_mean'], normalize_stats['param_std'])

    print(f"Test Loss: {test_metrics['loss']['total']:.4f}")
    print(f"Count Accuracy: {100*test_metrics['count_accuracy']:.1f}%")
    print(f"Type Accuracy: {100*test_metrics['type_accuracy']:.1f}%")

    # Save results
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    results = {
        'test_metrics': test_metrics,
        'val_metrics': history['val'][-1] if history['val'] else {},
        'history': history,
        'training_time': f"{hours}h {minutes}m"
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal training time: {hours}h {minutes}m")
    print(f"Results saved to: {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
