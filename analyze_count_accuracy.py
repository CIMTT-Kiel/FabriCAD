#!/usr/bin/env python
"""
Analyze count accuracy in detail:
- Binary exact match (what we already have)
- Mean Absolute Error (MAE) - average features off
- Percentage-based accuracy (treating each prediction as % correct)
- Off-by-1 accuracy (within ±1 feature)
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm

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

        # Normalize node and edge features
        x = (data['x'] - self.normalize_stats['node_mean'].cpu()) / self.normalize_stats['node_std'].cpu()
        x = torch.clamp(x, -10.0, 10.0)

        if data['edge_attr'].size(0) > 0:
            edge_attr = (data['edge_attr'] - self.normalize_stats['edge_mean'].cpu()) / self.normalize_stats['edge_std'].cpu()
            edge_attr = torch.clamp(edge_attr, -10.0, 10.0)
        else:
            edge_attr = data['edge_attr']

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
            # Normalize parameters
            params = torch.from_numpy(feat['params']).float()
            params_norm = (params - self.normalize_stats['param_mean'].cpu()) / self.normalize_stats['param_std'].cpu()
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

    # Pad features
    max_features = max(f.size(0) for f in features)
    max_features = max(max_features, 1)

    batched_features = []
    masks = []

    for feat in features:
        num_feat = feat.size(0)

        if num_feat < max_features:
            padding = torch.zeros(max_features - num_feat, 19)
            feat_padded = torch.cat([feat, padding], dim=0)
        else:
            feat_padded = feat[:max_features]

        # Create mask
        mask = torch.zeros(max_features)
        mask[:min(num_feat, max_features)] = 1.0

        batched_features.append(feat_padded)
        masks.append(mask)

    batched_features = torch.stack(batched_features)
    masks = torch.stack(masks)
    counts = torch.tensor(counts, dtype=torch.long)

    return batched_graph, batched_features, masks, counts


@torch.no_grad()
def analyze_count_predictions(model, loader, device):
    """Analyze count predictions in detail."""
    model.eval()

    all_predicted_counts = []
    all_true_counts = []

    for graph, features, masks, counts in tqdm(loader, desc="Analyzing"):
        graph = graph.to(device)
        features = features.to(device)
        counts = counts.to(device)

        # Predict (use ground truth for decoder input during evaluation)
        count_logits, _, _ = model(graph, features, counts)

        # Get predicted counts
        predicted_counts = torch.argmax(count_logits, dim=-1)

        all_predicted_counts.extend(predicted_counts.cpu().numpy())
        all_true_counts.extend(counts.cpu().numpy())

    all_predicted_counts = np.array(all_predicted_counts)
    all_true_counts = np.array(all_true_counts)

    # Calculate metrics

    # 1. Binary exact match (what we already have)
    exact_match = (all_predicted_counts == all_true_counts).mean() * 100

    # 2. Mean Absolute Error
    mae = np.abs(all_predicted_counts - all_true_counts).mean()

    # 3. Percentage-based accuracy (per sample)
    # For each sample: min(pred, true) / max(pred, true)
    percentage_accuracies = []
    for pred, true in zip(all_predicted_counts, all_true_counts):
        if true == 0 and pred == 0:
            percentage_accuracies.append(100.0)
        elif true == 0 or pred == 0:
            percentage_accuracies.append(0.0)
        else:
            percentage_accuracies.append(min(pred, true) / max(pred, true) * 100)

    avg_percentage_accuracy = np.mean(percentage_accuracies)

    # 4. Off-by-N accuracy
    off_by_0 = (all_predicted_counts == all_true_counts).mean() * 100
    off_by_1 = (np.abs(all_predicted_counts - all_true_counts) <= 1).mean() * 100
    off_by_2 = (np.abs(all_predicted_counts - all_true_counts) <= 2).mean() * 100
    off_by_3 = (np.abs(all_predicted_counts - all_true_counts) <= 3).mean() * 100

    # 5. Distribution of errors
    errors = all_predicted_counts - all_true_counts

    # Count distribution by error magnitude
    error_dist = {}
    for i in range(-10, 11):
        count = (errors == i).sum()
        if count > 0:
            error_dist[i] = int(count)

    return {
        'exact_match': exact_match,
        'mae': mae,
        'avg_percentage_accuracy': avg_percentage_accuracy,
        'off_by_0': off_by_0,
        'off_by_1': off_by_1,
        'off_by_2': off_by_2,
        'off_by_3': off_by_3,
        'error_distribution': error_dist,
        'total_samples': len(all_true_counts),
        'predictions': all_predicted_counts,
        'ground_truth': all_true_counts
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    print("Loading dataset...")
    dataset_dict = torch.load('data/feature_detection_100k_v4/dataset.pt', weights_only=False)
    data_list = dataset_dict['data']
    normalize_stats = {k: v.to(device) for k, v in dataset_dict['normalize_stats'].items()}

    # Split into test set (same split as training)
    n = len(data_list)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    np.random.seed(42)
    indices = np.random.permutation(n)

    test_indices = indices[train_size + val_size:]
    test_data = [data_list[i] for i in test_indices]

    print(f"Test samples: {len(test_data)}")

    # Create dataset and loader
    test_dataset = FeatureDataset(test_data, normalize_stats)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Load model
    print("Loading model...")
    checkpoint = torch.load('models/feature_detector_v5_100k/best_model.pt', weights_only=False)

    node_feature_dim = normalize_stats['node_mean'].size(0)
    edge_feature_dim = normalize_stats['edge_mean'].size(0)

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

    model.load_state_dict(checkpoint['model_state_dict'])

    # Analyze
    print("Analyzing count predictions...")
    results = analyze_count_predictions(model, test_loader, device)

    # Print results
    print("\n" + "="*70)
    print("COUNT ACCURACY ANALYSIS")
    print("="*70)
    print(f"\nTotal test samples: {results['total_samples']}")
    print()

    print("ACCURACY METRICS:")
    print("-" * 70)
    print(f"Exact Match (Binary):        {results['exact_match']:.2f}%")
    print(f"  -> Model gets exact count on {results['exact_match']:.1f}% of parts")
    print()

    print(f"Average Percentage Accuracy: {results['avg_percentage_accuracy']:.2f}%")
    print(f"  -> If part has 10 features and model predicts 9, that's 90%")
    print(f"  -> Averaged across all samples: {results['avg_percentage_accuracy']:.2f}%")
    print()

    print(f"Mean Absolute Error (MAE):   {results['mae']:.2f} features")
    print(f"  -> On average, predictions are off by {results['mae']:.2f} features")
    print()

    print("TOLERANCE ANALYSIS:")
    print("-" * 70)
    print(f"Exactly correct (+/-0):      {results['off_by_0']:.2f}%")
    print(f"Within +/-1 feature:         {results['off_by_1']:.2f}%")
    print(f"Within +/-2 features:        {results['off_by_2']:.2f}%")
    print(f"Within +/-3 features:        {results['off_by_3']:.2f}%")
    print()

    print("ERROR DISTRIBUTION:")
    print("-" * 70)
    print("Error | Count | Percentage")
    print("-" * 70)

    sorted_errors = sorted(results['error_distribution'].items())
    for error, count in sorted_errors:
        pct = count / results['total_samples'] * 100
        if error == 0:
            print(f"  {error:+3d} | {count:5d} | {pct:5.2f}%  <-- Exact match")
        else:
            print(f"  {error:+3d} | {count:5d} | {pct:5.2f}%")

    print()
    print("Note: Negative error = underprediction (predicted fewer features)")
    print("      Positive error = overprediction (predicted more features)")
    print()

    # Save detailed results
    output_path = Path('models/feature_detector_v5_100k/count_accuracy_detailed.json')
    import json

    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'exact_match': float(results['exact_match']),
        'mae': float(results['mae']),
        'avg_percentage_accuracy': float(results['avg_percentage_accuracy']),
        'off_by_0': float(results['off_by_0']),
        'off_by_1': float(results['off_by_1']),
        'off_by_2': float(results['off_by_2']),
        'off_by_3': float(results['off_by_3']),
        'error_distribution': results['error_distribution'],
        'total_samples': results['total_samples']
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Detailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
