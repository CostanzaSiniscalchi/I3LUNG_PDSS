"""
mask_modalities.py

Creates masked bags for sensitivity analysis by zeroing out specified modalities
in pre-built all-modality bags while preserving the full bag structure.

Usage:
    # Generate ALL masked subset combinations for radpy source (7 subsets)
    python mask_modalities.py --source radpy

    # Generate ALL for both sources (14 total)
    python mask_modalities.py --source radpy radfm

    # Single combination only
    python mask_modalities.py --source radpy --keep rwd radpy

    # Dry run
    python mask_modalities.py --source radpy --dry-run
"""

import argparse
import os
import sys
from itertools import combinations

import torch
from tqdm import tqdm

# Canonical modality order in all-modality bags (0-indexed position in mask)
MODALITY_INDEX = {
    'rwd': 0,       # feature1, mask[0]
    'radpy': 1,     # feature2, mask[1]
    'radfm': 1,     # feature2, mask[1] -- mutually exclusive with radpy
    'dp': 2,        # feature3, mask[2]
    'genomics': 3,  # feature4, mask[3]
}

# All modalities for each radiomics source (canonical order)
ALL_MODALITIES = {
    'radpy': ['rwd', 'radpy', 'dp', 'genomics'],
    'radfm': ['rwd', 'radfm', 'dp', 'genomics'],
}

# Source bag directory names
SOURCE_BAG_DIRS = {
    'radpy': 'bags_rwd_radpy_dp_genomics',
    'radfm': 'bags_rwd_radfm_dp_genomics',
}

# Expected feature dimensions per index (for verification)
EXPECTED_DIMS = {
    'radpy': {0: 11, 1: 128, 2: 768, 3: 4},
    'radfm': {0: 11, 1: 128, 2: 768, 3: 4},
}


def get_all_subsets(source):
    """Return all modality subsets to generate (rwd always kept, excluding full set).

    For 3 optional modalities, this yields 2^3 - 1 = 7 subsets.
    """
    all_mods = ALL_MODALITIES[source]
    optional_mods = [m for m in all_mods if m != 'rwd']

    subsets = []
    for r in range(len(optional_mods)):  # 0..2 optional mods kept
        for combo in combinations(optional_mods, r):
            subsets.append(['rwd'] + list(combo))

    # Also include subsets with all but one optional mod (size = len(optional) - 1 already covered)
    # r goes 0, 1, 2 for 3 optional mods => subsets of size 1, 2, 3
    # We exclude the full set (r == len(optional_mods)) already via range()
    return subsets


def verify_feature_ordering(source_dir, source):
    """Load one sample bag and verify feature dimensions match expected modalities."""
    bag_files = [f for f in os.listdir(source_dir) if f.endswith('.pt')]
    if not bag_files:
        print(f"[ERROR] No .pt files found in {source_dir}")
        sys.exit(1)

    sample_path = os.path.join(source_dir, bag_files[0])
    bag = torch.load(sample_path, weights_only=True)

    expected = EXPECTED_DIMS[source]
    mod_names = ALL_MODALITIES[source]

    for idx, mod_name in enumerate(mod_names):
        feat_key = f'feature{idx + 1}'
        if feat_key not in bag:
            print(f"[ERROR] Expected key '{feat_key}' not found in {sample_path}")
            print(f"        Available keys: {list(bag.keys())}")
            sys.exit(1)

        actual_dim = bag[feat_key].shape[0]
        expected_dim = expected[idx]
        if actual_dim != expected_dim:
            print(f"[ERROR] Dimension mismatch for {feat_key} ({mod_name}):")
            print(f"        Expected {expected_dim}, got {actual_dim}")
            print(f"        Bag file: {sample_path}")
            sys.exit(1)

    print(f"[INFO] Feature ordering verified on {bag_files[0]}:")
    for idx, mod_name in enumerate(mod_names):
        feat_key = f'feature{idx + 1}'
        print(f"        {feat_key} = {mod_name} (dim {bag[feat_key].shape[0]})")


def build_output_dirname(keep_modalities, all_modalities):
    """Build output directory name: bags_{kept}_masked_{masked} in canonical order."""
    kept = [m for m in all_modalities if m in keep_modalities]
    masked = [m for m in all_modalities if m not in keep_modalities]

    kept_str = "_".join(kept)
    masked_str = "_".join(masked)

    return f"bags_{kept_str}_masked_{masked_str}"


def mask_bag(bag_dict, mask_indices):
    """Create a masked copy of a bag by zeroing specified modality indices.

    Args:
        bag_dict: dict with keys 'feature1'...'featureN' and 'mask'
        mask_indices: set of 0-based indices to mask out

    Returns:
        dict: new bag dictionary with masking applied
    """
    new_bag = {}
    new_mask = bag_dict['mask'].clone()

    for i in range(len(new_mask)):
        feat_key = f'feature{i + 1}'
        if i in mask_indices:
            new_bag[feat_key] = torch.zeros_like(bag_dict[feat_key])
            new_mask[i] = False
        else:
            new_bag[feat_key] = bag_dict[feat_key].clone()

    new_bag['mask'] = new_mask
    return new_bag


def mask_modalities(source_dir, output_dir, keep_modalities, source):
    """Process all bags in source_dir, masking modalities not in keep_modalities.

    Args:
        source_dir: path to source bag directory
        output_dir: path to output directory for masked bags
        keep_modalities: list of modality names to keep
        source: 'radpy' or 'radfm'
    """
    all_mods = ALL_MODALITIES[source]
    mask_indices = {MODALITY_INDEX[m] for m in all_mods if m not in keep_modalities}
    masked_names = [m for m in all_mods if m not in keep_modalities]

    print(f"\n[INFO] Keeping: {keep_modalities}")
    print(f"[INFO] Masking: {masked_names} (feature indices {sorted(mask_indices)})")
    print(f"[INFO] Output:  {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    bag_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
    saved = 0
    skipped = 0

    for bag_file in tqdm(bag_files, desc="Masking bags"):
        src_path = os.path.join(source_dir, bag_file)
        try:
            bag_dict = torch.load(src_path, weights_only=True)
        except Exception as e:
            print(f"\n[WARN] Skipping {bag_file}: {e}")
            skipped += 1
            continue

        # Only include patients that have ALL modalities present
        if not bag_dict['mask'].all():
            skipped += 1
            continue

        masked_bag = mask_bag(bag_dict, mask_indices)
        torch.save(masked_bag, os.path.join(output_dir, bag_file))
        saved += 1

    print(f"[INFO] Saved {saved} masked bags to {output_dir} (skipped {skipped} with incomplete modalities)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create masked bags for sensitivity analysis. "
                    "Zeros out unselected modalities while preserving the full bag structure."
    )
    parser.add_argument(
        '--source',
        nargs='+',
        required=True,
        choices=['radpy', 'radfm'],
        help='Radiomics source type(s). Determines source bag directory.'
    )
    parser.add_argument(
        '--keep',
        nargs='+',
        default=None,
        choices=['rwd', 'radpy', 'radfm', 'dp', 'genomics'],
        help='Modalities to keep (others masked). If omitted, generates ALL subsets.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without creating any files.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    for source in args.source:
        all_mods = ALL_MODALITIES[source]
        source_dir = os.path.join(project_root, 'bags', SOURCE_BAG_DIRS[source])

        if not os.path.isdir(source_dir):
            print(f"[ERROR] Source directory not found: {source_dir}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Source: {source} ({source_dir})")
        print(f"{'='*60}")

        # Determine which subsets to generate
        if args.keep is not None:
            # Single run mode
            if 'rwd' not in args.keep:
                print("[ERROR] rwd must always be kept.")
                sys.exit(1)
            if source == 'radpy' and 'radfm' in args.keep:
                print("[ERROR] Cannot keep 'radfm' when source is 'radpy'.")
                sys.exit(1)
            if source == 'radfm' and 'radpy' in args.keep:
                print("[ERROR] Cannot keep 'radpy' when source is 'radfm'.")
                sys.exit(1)
            invalid = set(args.keep) - set(all_mods)
            if invalid:
                print(f"[ERROR] Invalid modalities for source '{source}': {invalid}")
                sys.exit(1)
            if set(args.keep) >= set(all_mods):
                print("[ERROR] All modalities selected -- nothing to mask.")
                sys.exit(1)
            subsets = [args.keep]
        else:
            # Batch mode: all subsets
            subsets = get_all_subsets(source)

        # Verify feature ordering before processing
        if not args.dry_run:
            verify_feature_ordering(source_dir, source)

        # In dry-run mode, count how many bags have all modalities present
        if args.dry_run:
            bag_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
            complete = 0
            for bag_file in tqdm(bag_files, desc="Scanning source bags"):
                bag = torch.load(os.path.join(source_dir, bag_file), weights_only=True)
                if bag['mask'].all():
                    complete += 1
            print(f"\n[DRY RUN] {complete}/{len(bag_files)} patients have all modalities present")

        for keep in subsets:
            dirname = build_output_dirname(keep, all_mods)
            output_dir = os.path.join(project_root, 'bags', dirname)
            masked_mods = [m for m in all_mods if m not in keep]
            mask_idx = sorted({MODALITY_INDEX[m] for m in masked_mods})

            if args.dry_run:
                print(f"\n[DRY RUN] Keep: {keep}")
                print(f"[DRY RUN] Mask: {masked_mods} (feature indices {mask_idx})")
                print(f"[DRY RUN] Output: {output_dir}")
                print(f"[DRY RUN] Bags: {complete}")
            else:
                if os.path.exists(output_dir) and os.listdir(output_dir):
                    print(f"\n[WARN] Output directory already exists: {output_dir}")
                    print(f"[WARN] Existing files will be overwritten.")
                mask_modalities(source_dir, output_dir, keep, source)


if __name__ == '__main__':
    main()
