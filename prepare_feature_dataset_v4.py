#!/usr/bin/env python
"""
Prepare dataset for feature-level detection (batch-processing crash-resistant version).

Processes samples in batches within isolated subprocesses. If a batch crashes, skip it.
Much faster than per-sample isolation.
"""
import argparse
import json
import torch
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys


# Feature type mapping
FEATURE_TYPES = [
    'hole', 'pocket', 'step', 'weld', 'chamfer', 'fillet', 'unknown', 'additive', 'no_feature'
]
FEATURE_TYPE_TO_IDX = {t: i for i, t in enumerate(FEATURE_TYPES)}


class WelfordAccumulator:
    """Online algorithm for computing mean and variance."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        if values.size(0) == 0:
            return
        if self.mean is None:
            self.mean = torch.zeros(values.size(1), dtype=torch.float64)
            self.m2 = torch.zeros(values.size(1), dtype=torch.float64)
        values = values.double()
        for value in values:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def get_stats(self, default_dim=None):
        if self.mean is None:
            if default_dim is None:
                raise ValueError("Accumulator never initialized")
            return torch.zeros(default_dim), torch.ones(default_dim)
        if self.count < 2:
            return torch.zeros_like(self.mean).float(), torch.ones_like(self.mean).float()
        variance = self.m2 / self.count
        std = torch.sqrt(variance)
        std[std == 0] = 1.0

        # Replace any NaN/Inf values with default (0 mean, 1 std)
        mean_float = self.mean.float()
        std_float = std.float()
        mean_float[torch.isnan(mean_float) | torch.isinf(mean_float)] = 0.0
        std_float[torch.isnan(std_float) | torch.isinf(std_float)] = 1.0

        return mean_float, std_float


def process_batch_isolated(batch_ids, data_root, ground_truth_file, output_file):
    """Process a batch of samples in isolated subprocess."""

    # Create batch processing script
    script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent}")
import json
import torch
import numpy as np
from pathlib import Path
import cadquery as cq
from fabricad.graph_builder import build_face_graph

FEATURE_TYPE_TO_IDX = {FEATURE_TYPE_TO_IDX}

def parse_feature_params(feature):
    params = feature.get('parameters', {{}})
    parsed_dims = params.get('parsed_dimensions', {{}})
    diameter = parsed_dims.get('diameter', 0.0)
    depth = parsed_dims.get('depth', 0.0)
    width = parsed_dims.get('width', 0.0)
    length = parsed_dims.get('length', 0.0)
    height = parsed_dims.get('height', 0.0)
    bbox = params.get('bbox', [0.0, 0.0, 0.0])
    if not isinstance(bbox, list):
        bbox = [0.0, 0.0, 0.0]
    bbox_x = bbox[0] if len(bbox) > 0 else 0.0
    bbox_y = bbox[1] if len(bbox) > 1 else 0.0
    bbox_z = bbox[2] if len(bbox) > 2 else 0.0
    volume = abs(feature.get('volume_change', 0.0))
    confidence = feature.get('confidence', 0.0)
    return np.array([diameter, depth, width, length, height, bbox_x, bbox_y, bbox_z, volume, confidence], dtype=np.float32)

# Load ground truth for all samples in batch
ground_truth_map = {{}}
with open(r"{ground_truth_file}", 'r') as f:
    for line in f:
        sample = json.loads(line)
        ground_truth_map[sample['sample_id']] = sample

batch_ids = {batch_ids}
data_root = Path(r"{data_root}")
results = []

for sample_id in batch_ids:
    sample_dir = data_root / sample_id
    if not sample_dir.exists():
        continue

    geometry_files = list(sample_dir.glob('geometry_*.STEP'))
    if not geometry_files:
        continue

    ground_truth = ground_truth_map.get(sample_id)
    if not ground_truth:
        continue

    try:
        # Load STEP
        wp = cq.importers.importStep(str(geometry_files[0]))
        shape = wp.val() if hasattr(wp, 'val') else wp.objects[0]

        # Build graph
        graph = build_face_graph(shape, feature_labels=None)
        node_features = torch.from_numpy(np.vstack(graph.face_features)).float()

        if len(graph.adjacency) > 0:
            edge_index = torch.tensor(graph.adjacency, dtype=torch.long).t()
            edge_features = torch.from_numpy(np.vstack(graph.edge_features)).float()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, 4), dtype=torch.float32)

        # Extract features
        feature_list = []
        for feature in ground_truth.get('features', []):
            feature_type = feature.get('feature_type', 'unknown')
            type_idx = FEATURE_TYPE_TO_IDX.get(feature_type, FEATURE_TYPE_TO_IDX['unknown'])
            params = parse_feature_params(feature)
            feature_list.append({{'type': type_idx, 'params': params}})

        # Add EOS
        feature_list.append({{'type': FEATURE_TYPE_TO_IDX['no_feature'], 'params': np.zeros(10, dtype=np.float32)}})

        results.append({{
            'sample_id': sample_id,
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features,
            'features': feature_list
        }})
    except Exception as e:
        # Skip this sample on error
        continue

# Save batch results
torch.save(results, r"{output_file}")
'''

    # Write and execute script
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    script_file.write(script)
    script_file.close()

    try:
        result = subprocess.run(
            [sys.executable, script_file.name],
            timeout=300,  # 5 min timeout for whole batch
            capture_output=True
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        return -1
    except Exception:
        return -2
    finally:
        Path(script_file.name).unlink(missing_ok=True)


def load_ground_truth(results_file):
    """Load feature labels from JSONL file."""
    samples = {}
    with open(results_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples[sample['sample_id']] = sample
    return samples


def compute_normalization_stats_v4(batches, data_root, ground_truth_file, temp_dir):
    """Pass 1: Compute normalization statistics using batch processing."""
    print("\n" + "="*70)
    print("PASS 1: Computing normalization statistics (batch processing)")
    print("="*70)

    node_acc = WelfordAccumulator()
    edge_acc = WelfordAccumulator()
    param_acc = WelfordAccumulator()

    failed_batches = 0
    processed_samples = 0
    crashed_batches = 0

    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="Pass 1: Batches")):
        output_file = temp_dir / f"batch_{batch_idx}.pt"
        exit_code = process_batch_isolated(batch_ids, data_root, ground_truth_file, output_file)

        # If output exists, batch succeeded (even if exit code is 139 from cleanup crash)
        if output_file.exists():
            try:
                batch_results = torch.load(output_file, weights_only=False)

                for data in batch_results:
                    processed_samples += 1
                    node_acc.update(data['x'])
                    if data['edge_attr'].size(0) > 0:
                        edge_acc.update(data['edge_attr'])

                    params_list = [feat['params'] for feat in data['features'][:-1]]
                    if len(params_list) > 0:
                        params_tensor = torch.from_numpy(np.stack(params_list))
                        param_acc.update(params_tensor)

                output_file.unlink()
            except Exception:
                failed_batches += 1
        elif exit_code == 139:
            crashed_batches += 1  # Crashed without producing output
        else:
            failed_batches += 1  # Failed for other reasons

    print(f"\nProcessed samples: {processed_samples}")
    print(f"Failed batches: {failed_batches}")
    print(f"Crashed batches: {crashed_batches}")

    node_mean, node_std = node_acc.get_stats(default_dim=18)
    edge_mean, edge_std = edge_acc.get_stats(default_dim=4)
    param_mean, param_std = param_acc.get_stats(default_dim=10)

    return {
        'node_mean': node_mean, 'node_std': node_std,
        'edge_mean': edge_mean, 'edge_std': edge_std,
        'param_mean': param_mean, 'param_std': param_std
    }, processed_samples


def process_and_save_dataset_v4(batches, data_root, ground_truth_file, temp_dir):
    """Pass 2: Process and collect dataset."""
    print("\n" + "="*70)
    print("PASS 2: Processing and saving dataset (batch processing)")
    print("="*70)

    processed_data = []
    failed_batches = 0
    crashed_batches = 0

    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="Pass 2: Batches")):
        output_file = temp_dir / f"batch_{batch_idx}.pt"
        exit_code = process_batch_isolated(batch_ids, data_root, ground_truth_file, output_file)

        # If output exists, batch succeeded (even if exit code is 139 from cleanup crash)
        if output_file.exists():
            try:
                batch_results = torch.load(output_file, weights_only=False)
                processed_data.extend(batch_results)
                output_file.unlink()
            except Exception:
                failed_batches += 1
        elif exit_code == 139:
            crashed_batches += 1  # Crashed without producing output
        else:
            failed_batches += 1  # Failed for other reasons

    print(f"\nProcessed samples: {len(processed_data)}")
    print(f"Failed batches: {failed_batches}")
    print(f"Crashed batches: {crashed_batches}")

    if len(processed_data) > 0:
        num_features = [len(d['features']) - 1 for d in processed_data]
        print(f"\nFeature count statistics:")
        print(f"  Mean: {np.mean(num_features):.1f}")
        print(f"  Median: {np.median(num_features):.1f}")
        print(f"  Min: {np.min(num_features)}, Max: {np.max(num_features)}")

    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset (batch crash-resistant)')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=50, help='Samples per batch')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_file = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PREPARING DATASET (BATCH CRASH-RESISTANT)")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Results: {results_file}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")

    ground_truth = load_ground_truth(results_file)
    sample_ids = list(ground_truth.keys())

    # Create batches
    batches = [sample_ids[i:i+args.batch_size] for i in range(0, len(sample_ids), args.batch_size)]
    print(f"\nSamples: {len(sample_ids)}")
    print(f"Batches: {len(batches)}")

    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # Pass 1
    normalize_stats, num_processed = compute_normalization_stats_v4(
        batches, data_root, results_file, temp_dir
    )

    # Pass 2
    processed_data = process_and_save_dataset_v4(
        batches, data_root, results_file, temp_dir
    )

    # Clean up
    temp_dir.rmdir()

    # Save
    print(f"\nSaving dataset...")
    torch.save({
        'data': processed_data,
        'normalize_stats': normalize_stats,
        'feature_types': FEATURE_TYPES,
        'feature_type_to_idx': FEATURE_TYPE_TO_IDX
    }, output_dir / 'dataset.pt')

    print(f"\n{'='*70}")
    print(f"DATASET SAVED!")
    print(f"{'='*70}")
    print(f"Output: {output_dir / 'dataset.pt'}")
    print(f"Samples: {len(processed_data)}")


if __name__ == '__main__':
    main()
