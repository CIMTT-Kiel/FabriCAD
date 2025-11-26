#!/usr/bin/env python
"""Evaluate V5 model checkpoint on test set."""
import torch
from pathlib import Path
from feature_detector_v5 import FeatureDetectorV5
from train_feature_detector_v5 import FeatureDataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np


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


def evaluate_model(model, loader, device, param_mean, param_std):
    """Evaluate model on a dataset."""
    model.eval()

    total_samples = 0
    correct_counts = 0
    correct_types = 0
    total_types = 0

    with torch.no_grad():
        for graph, features, masks, counts in tqdm(loader, desc="Evaluating"):
            graph = graph.to(device)
            features = features.to(device)
            counts = counts.to(device)
            masks = masks.to(device)

            # Forward pass
            count_logits, type_logits, param_preds = model(graph, features=None, counts=None)

            # Count accuracy
            pred_counts = torch.argmax(count_logits, dim=-1)
            correct_counts += (pred_counts == counts).sum().item()
            total_samples += counts.size(0)

            # Type accuracy (for valid features only)
            gt_types = torch.argmax(features[:, :, :9], dim=-1)
            pred_types = torch.argmax(type_logits, dim=-1)

            # Handle size mismatch between predictions and ground truth
            batch_size = gt_types.size(0)
            min_len = min(pred_types.size(1), gt_types.size(1))

            # Truncate both to same length
            gt_types_trunc = gt_types[:, :min_len]
            pred_types_trunc = pred_types[:, :min_len]
            masks_trunc = masks[:, :min_len]

            # Only count types where mask is 1
            valid_mask = masks_trunc.bool()
            correct_types += ((pred_types_trunc == gt_types_trunc) & valid_mask).sum().item()
            total_types += valid_mask.sum().item()

    count_acc = 100.0 * correct_counts / total_samples
    type_acc = 100.0 * correct_types / total_types if total_types > 0 else 0.0

    return {
        'count_accuracy': count_acc,
        'type_accuracy': type_acc,
        'total_samples': total_samples,
        'total_features': total_types
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    print("Loading dataset...")
    data_path = Path('data/feature_detection_100k_v4/dataset.pt')
    dataset_dict = torch.load(data_path, weights_only=False)

    # Get normalization stats
    param_mean = dataset_dict['normalize_stats']['param_mean']
    param_std = dataset_dict['normalize_stats']['param_std']
    normalize_stats = dataset_dict['normalize_stats']

    # Split data (70/15/15)
    data_list = dataset_dict['data']
    np.random.seed(42)
    indices = np.random.permutation(len(data_list))

    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))

    test_indices = indices[train_size + val_size:]
    test_data = [data_list[i] for i in test_indices]

    # Create test dataset
    test_dataset = FeatureDataset(test_data, normalize_stats)

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("\nCreating model...")
    node_feature_dim = dataset_dict['normalize_stats']['node_mean'].size(0)
    edge_feature_dim = dataset_dict['normalize_stats']['edge_mean'].size(0)

    model = FeatureDetectorV5(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=384,
        num_gnn_layers=6,
        num_transformer_layers=4,
        num_feature_types=9,
        num_params=10,
        max_count=30,
        nhead=8
    ).to(device)

    # Load checkpoint
    checkpoint_path = Path('models/feature_detector_v5_100k/best_model.pt')
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Best model loaded (from Epoch 23: Val Count 80.6%, Val Type 85.9%)")

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    results = evaluate_model(model, test_loader, device, param_mean, param_std)

    print(f"\nTest Results:")
    print(f"  Count Accuracy: {results['count_accuracy']:.2f}%")
    print(f"  Type Accuracy: {results['type_accuracy']:.2f}%")
    print(f"  Total Samples: {results['total_samples']}")
    print(f"  Total Features: {results['total_features']}")

    # Save results
    output = {
        'test_count_accuracy': results['count_accuracy'],
        'test_type_accuracy': results['type_accuracy'],
        'val_count_accuracy': 80.6,  # From Epoch 23
        'val_type_accuracy': 85.9,  # From Epoch 23
        'epoch': 23,
        'total_samples': results['total_samples'],
        'total_features': results['total_features']
    }

    output_file = Path('models/feature_detector_v5_100k/test_results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("COMPARISON WITH V4 BASELINE")
    print("="*70)
    print("V4 (10k dataset):  Count: 59.4%, Type: 79.8%")
    print(f"V5 (100k dataset): Count: {results['count_accuracy']:.1f}%, Type: {results['type_accuracy']:.1f}%")
    print(f"Improvement:       Count: +{results['count_accuracy']-59.4:.1f}%, Type: +{results['type_accuracy']-79.8:.1f}%")


if __name__ == '__main__':
    main()
