#!/usr/bin/env python3
"""Test loading training examples from manifest and verify images load correctly."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset import load_training_examples_from_manifest

def create_id_mapping():
    """Create ID mapping from training_participants.txt and manifest."""
    # Load original participant IDs in processing order
    with open('./data/training_participants.txt') as f:
        original_ids = [line.strip() for line in f if line.strip()]

    # Load manifest to see which anonymous IDs exist
    with open('./data/training/train_manifest.json') as f:
        manifest = json.load(f)

    # Get unique anonymous IDs from manifest
    anon_ids_in_manifest = set()
    for ex in manifest["examples"]:
        anon_ids_in_manifest.add(ex["person_id"])

    # Also check val and test manifests
    for split in ["val", "test"]:
        with open(f'./data/training/{split}_manifest.json') as f:
            m = json.load(f)
            for ex in m["examples"]:
                anon_ids_in_manifest.add(ex["person_id"])

    # Build mapping: anonymous IDs are assigned in order P1000, P1001, ...
    # to participants that successfully processed (have images)
    id_mapping = {}
    anon_counter = 1000

    for orig_id in original_ids:
        anon_id = f"P{anon_counter}"
        if anon_id in anon_ids_in_manifest:
            id_mapping[anon_id] = orig_id
            anon_counter += 1

    return id_mapping


def main():
    print("Creating ID mapping...")
    id_mapping = create_id_mapping()
    print(f"Mapped {len(id_mapping)} participants")

    # Save the mapping
    output_path = Path('./data/training/id_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"Saved mapping to {output_path}")

    # Show first few mappings
    print("\nFirst 5 mappings:")
    for i, (anon, orig) in enumerate(list(id_mapping.items())[:5]):
        print(f"  {anon} -> {orig}")

    # Test loading training examples
    print("\n" + "=" * 60)
    print("Testing loading 5 training examples with images...")
    print("=" * 60)

    cache_dir = Path('./data')
    manifest_path = Path('./data/training/train_manifest.json')

    # Load just the first 5 for testing
    examples = load_training_examples_from_manifest(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        person_id_mapping=id_mapping,
    )

    print(f"\nSuccessfully loaded {len(examples)} examples")

    # Show details of first 5
    print("\nFirst 5 examples:")
    for i, ex in enumerate(examples[:5]):
        print(f"\n[{i+1}] {ex.example_id}")
        print(f"    Image size: {ex.image.size}")
        print(f"    Image mode: {ex.image.mode}")
        print(f"    Clinical context: {len(ex.clinical_context)} chars")
        print(f"    CGM context: {len(ex.cgm_context)} chars")
        print(f"    Target response: {len(ex.target_response)} chars")

    if len(examples) >= 5:
        print("\n✓ Successfully loaded and verified 5+ training examples!")
        print("  Pipeline is ready for training.")
    else:
        print(f"\n⚠ Only loaded {len(examples)} examples - check for missing images")

    return examples


if __name__ == "__main__":
    examples = main()
