#!/usr/bin/env python
"""
Extract features from CSV files (for 10k dataset without substep STEP files).

This script reads features.csv directly and maps them to the expected feature format.
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re


def parse_tool_info(tool_str):
    """
    Parse tool string to extract tool type and diameter.

    Examples:
        "Schlichtdrehmeißel (0.0)" -> ('Schlichtdrehmeißel', 0.0)
        "Einstichmeißel (10.0[mm]) (0.0)" -> ('Einstichmeißel', 10.0)
        "Konturdrehmeißel (0.0)" -> ('Konturdrehmeißel', 0.0)
    """
    if not isinstance(tool_str, str):
        return ('unknown', 0.0)

    # Extract diameter from patterns like "10.0[mm]" or just "(0.0)"
    diameter = 0.0
    diameter_match = re.search(r'(\d+\.?\d*)\s*\[mm\]', tool_str)
    if diameter_match:
        diameter = float(diameter_match.group(1))

    # Extract tool name (before first parenthesis)
    tool_name = tool_str.split('(')[0].strip()

    return (tool_name, diameter)


def map_feature_type(tool_name, volume, kurztext):
    """
    Map tool name and volume sign to feature type.

    Feature types: hole, pocket, step, weld, chamfer, fillet, unknown, additive
    """
    # Check if additive based on positive volume
    is_additive = volume > 0

    # Check Kurztext for explicit feature mentions
    kurztext_lower = str(kurztext).lower()

    # Weld detection
    if 'schweiß' in kurztext_lower or 'weld' in kurztext_lower:
        return 'weld', True

    # If additive, it's likely weld or additive manufacturing
    if is_additive:
        if 'auftrag' in kurztext_lower:
            return 'additive', True
        return 'weld', True

    # Map tool names to subtractive features
    tool_lower = tool_name.lower()

    if 'bohr' in tool_lower or 'drill' in tool_lower:
        return 'hole', False
    elif 'fräs' in tool_lower or 'mill' in tool_lower:
        # Could be pocket or step depending on depth
        if abs(volume) > 5000:  # Large volume -> likely pocket
            return 'pocket', False
        return 'step', False
    elif 'einstich' in tool_lower or 'groove' in tool_lower:
        return 'step', False
    elif 'schrupp' in tool_lower or 'schlicht' in tool_lower:
        # Rough/finish turning - treat as step
        return 'step', False
    elif 'fase' in tool_lower or 'chamfer' in tool_lower:
        return 'chamfer', False
    elif 'rundung' in tool_lower or 'fillet' in tool_lower:
        return 'fillet', False

    # Default to unknown for subtractive
    return 'unknown', False


def parse_weld_seam_length(weld_col):
    """Parse weld seam length from column value."""
    if pd.isna(weld_col) or weld_col == '-':
        return 0.0

    try:
        # Try to extract numeric value
        match = re.search(r'(\d+\.?\d*)', str(weld_col))
        if match:
            return float(match.group(1))
    except:
        pass

    return 0.0


def extract_parameters(row, feature_type, tool_diameter):
    """
    Extract feature parameters from CSV row.

    Returns dict compatible with existing format.
    """
    volume = abs(float(row['Volumen[mm^3]']))

    params = {
        'ground_truth_volume': float(row['Volumen[mm^3]']),
        'ground_truth_text': str(row['Kurztext']),
    }

    if feature_type == 'hole':
        params['diameter'] = tool_diameter
        # Estimate depth from volume and diameter
        if tool_diameter > 0:
            # V = π * r^2 * h -> h = V / (π * r^2)
            import math
            radius = tool_diameter / 2
            depth = volume / (math.pi * radius * radius) if radius > 0 else 0
            params['depth'] = depth
        else:
            params['depth'] = 0.0
        params['bbox'] = [tool_diameter, tool_diameter, params['depth']]

    elif feature_type in ['pocket', 'step']:
        # Estimate dimensions from volume (assume cubic shape)
        side = volume ** (1/3) if volume > 0 else 0
        params['width'] = side
        params['length'] = side
        params['depth'] = side / 2  # Assume depth is half of side
        params['bbox'] = [side, side, side / 2]

    elif feature_type == 'weld':
        weld_length = parse_weld_seam_length(row.get('Schweißnaht [mm]', '-'))
        params['weld_seam_length'] = weld_length
        params['length'] = weld_length
        # Estimate width and height from volume
        if weld_length > 0:
            cross_section = volume / weld_length
            side = cross_section ** 0.5
            params['width'] = side
            params['height'] = side
        else:
            params['width'] = 0.0
            params['height'] = 0.0
        params['description'] = f"Weld seam length: {weld_length}mm"

    elif feature_type in ['chamfer', 'fillet']:
        # Small features - use tool diameter
        params['size'] = tool_diameter if tool_diameter > 0 else 1.0
        params['bbox'] = [tool_diameter, tool_diameter, tool_diameter]

    else:  # unknown or additive
        side = volume ** (1/3) if volume > 0 else 0
        params['bbox'] = [side, side, side]

    return params


def process_sample(sample_id, sample_dir):
    """
    Process single sample: extract features from CSV.

    Returns dict in same format as geometry-based recognizer.
    """
    csv_path = sample_dir / 'interim' / 'substeps' / 'features.csv'

    if not csv_path.exists():
        return None

    try:
        # Load CSV
        df = pd.read_csv(csv_path, sep=';', encoding='latin1')
        df.columns = df.columns.str.strip()

        # Filter out null features
        df = df[df['Kurztext'] != 'null']

        features = []

        for _, row in df.iterrows():
            # Parse tool info
            tool_name, tool_diameter = parse_tool_info(row['Werkzeug[mm](D)'])

            # Get volume
            volume = float(row['Volumen[mm^3]'])

            # Map feature type
            feature_type, is_additive = map_feature_type(
                tool_name, volume, row['Kurztext']
            )

            # Extract parameters
            parameters = extract_parameters(row, feature_type, tool_diameter)

            # Create feature dict
            feature = {
                'success': True,
                'feature_type': feature_type,
                'is_additive': is_additive,
                'volume_change': volume,
                'confidence': 0.8,  # Fixed confidence for CSV extraction
                'parameters': parameters,
                'ground_truth': {
                    'Arbeitsschritt': int(row['Arbeitsschritt']),
                    'Subschritt': str(row['Subschritt']),
                    'Kurztext': str(row['Kurztext']),
                    'Volumen[mm^3]': volume,
                    'Werkzeug[mm](D)': str(row['Werkzeug[mm](D)']),
                    'Schweißnaht [mm]': str(row.get('Schweißnaht [mm]', '-'))
                },
                'substep_id': str(row['Subschritt'])
            }

            features.append(feature)

        return {
            'sample_id': sample_id,
            'success': True,
            'features': features,
            'summary': {
                'total_features': len(features),
                'successful': len(features),
                'failed': 0,
                'success_rate': 1.0,
                'feature_types': {}
            }
        }

    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract features from CSV files (10k dataset)')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of 10k dataset')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXTRACTING FEATURES FROM CSV (10K DATASET)")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Output: {output_path}")

    # Find all sample directories
    sample_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if args.max_samples:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"\nProcessing {len(sample_dirs)} samples...")

    # Process samples
    results = []
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Extracting"):
        sample_id = sample_dir.name
        result = process_sample(sample_id, sample_dir)

        if result:
            results.append(result)
        else:
            failed += 1

    print(f"\nProcessed: {len(results)} samples")
    print(f"Failed: {failed} samples")

    # Compute overall statistics
    total_features = sum(len(r['features']) for r in results)
    print(f"Total features extracted: {total_features}")

    # Feature type distribution
    feature_types = {}
    for r in results:
        for f in r['features']:
            ftype = f['feature_type']
            feature_types[ftype] = feature_types.get(ftype, 0) + 1

    print(f"\nFeature type distribution:")
    for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_features if total_features > 0 else 0
        print(f"  {ftype}: {count} ({pct:.1f}%)")

    # Save to JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
