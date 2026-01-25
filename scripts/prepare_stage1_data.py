#!/usr/bin/env python3
"""Stage 1: Prepare training data for visual understanding.

Trains MedGemma to identify retinal findings from images alone.
- Input: Retinal fundus image
- Output: Retinal findings description (DR, AMD, RVO)

Ground truth comes from clinical diagnoses in condition_occurrence.csv.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.dataset import TrainingConfig, split_examples, save_dataset_manifest
from src.training.retinal_findings import load_retinal_findings, format_retinal_findings_for_target


def main():
    print("=" * 60)
    print("STAGE 1: Visual Understanding Training Data")
    print("Task: Image → Retinal Findings")
    print("=" * 60)

    # Load participant IDs
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"\nPreparing data for {len(person_ids)} participants")

    # Initialize loader (use cached data, don't try to download)
    loader = ParticipantLoader(cache_dir='./data', auto_download=False)
    # Skip Azure downloads - use locally cached data
    # loader.ensure_metadata()
    # loader.ensure_clinical_data()

    # Load retinal findings (ground truth)
    print("\nLoading retinal findings from condition_occurrence.csv...")
    retinal_findings = load_retinal_findings()

    # Count findings in our training set
    dr_count = sum(1 for pid in person_ids if retinal_findings.get(pid, {}).get("diabetic_retinopathy"))
    amd_count = sum(1 for pid in person_ids if retinal_findings.get(pid, {}).get("amd"))
    rvo_count = sum(1 for pid in person_ids if retinal_findings.get(pid, {}).get("rvo"))
    print(f"  Training set findings: DR={dr_count}, AMD={amd_count}, RVO={rvo_count}")

    config = TrainingConfig(
        min_cgm_days=0,  # Don't require CGM for Stage 1
        require_complete=False,  # Only need images
        anonymize_ids=True,
        use_both_eyes=True,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
    )

    # Create examples
    print("\nCreating training examples...")
    from dataclasses import dataclass
    from PIL import Image

    @dataclass
    class Stage1Example:
        example_id: str
        image: Image.Image
        target_findings: str  # Ground truth retinal findings
        person_id: str
        eye: str
        has_dr: bool
        has_amd: bool
        has_rvo: bool

    examples = []
    anon_counter = 1000
    id_mapping = {}  # Track anon_id -> original_id mapping

    for i, person_id in enumerate(person_ids):
        print(f"[{i+1}/{len(person_ids)}] Processing {person_id}...", end=" ", flush=True)
        try:
            data = loader.load(person_id)

            # Need at least one eye image
            if not data.fundus_left and not data.fundus_right:
                print("✗ (no images)")
                continue

            anon_id = f"P{anon_counter}" if config.anonymize_ids else person_id
            anon_counter += 1

            # Save ID mapping
            id_mapping[anon_id] = person_id

            # Get retinal findings for this participant
            findings = retinal_findings.get(person_id, {})
            target_text = format_retinal_findings_for_target(findings)

            eyes_to_process = []
            if data.fundus_left:
                eyes_to_process.append(("left", data.fundus_left))
            if data.fundus_right and config.use_both_eyes:
                eyes_to_process.append(("right", data.fundus_right))
            elif data.fundus_right and not data.fundus_left:
                eyes_to_process.append(("right", data.fundus_right))

            batch_count = 0
            for eye, image in eyes_to_process:
                example = Stage1Example(
                    example_id=f"{anon_id}_{eye}",
                    image=image,
                    target_findings=target_text,
                    person_id=anon_id,
                    eye=eye,
                    has_dr=findings.get("diabetic_retinopathy", False),
                    has_amd=findings.get("amd", False),
                    has_rvo=findings.get("rvo", False),
                )
                examples.append(example)
                batch_count += 1

            print(f"✓ ({batch_count} examples)")
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\nTotal Stage 1 examples: {len(examples)}")

    # Split by participant with STRATIFICATION (ensure positives in all splits)
    import random
    random.seed(config.random_seed)

    by_participant = {}
    for ex in examples:
        if ex.person_id not in by_participant:
            by_participant[ex.person_id] = []
        by_participant[ex.person_id].append(ex)

    # Separate participants by DR status FIRST (priority for diabetes project)
    # Then by other positive findings, then negatives
    dr_participants = []
    other_positive_participants = []  # AMD or RVO but not DR
    negative_participants = []

    for pid, exs in by_participant.items():
        has_dr = any(ex.has_dr for ex in exs)
        has_other = any(ex.has_amd or ex.has_rvo for ex in exs)

        if has_dr:
            dr_participants.append(pid)
        elif has_other:
            other_positive_participants.append(pid)
        else:
            negative_participants.append(pid)

    random.shuffle(dr_participants)
    random.shuffle(other_positive_participants)
    random.shuffle(negative_participants)

    print(f"\n  DR participants: {len(dr_participants)} (PRIORITY)")
    print(f"  Other positive participants (AMD/RVO only): {len(other_positive_participants)}")
    print(f"  Negative participants: {len(negative_participants)}")

    # Stratified split - ensure DR cases in ALL splits (most important)
    def stratified_split_by_priority(dr_list, other_pos_list, neg_list, train_frac, val_frac):
        """Split with DR cases guaranteed in each split."""
        n_dr = len(dr_list)
        n_other = len(other_pos_list)
        n_neg = len(neg_list)

        # DR: ensure at least 1 in val and test, rest in train
        # With small numbers, use ceiling for val/test to ensure representation
        dr_test_count = max(1, int(n_dr * (1 - train_frac - val_frac) + 0.5))
        dr_val_count = max(1, int(n_dr * val_frac + 0.5))
        dr_train_count = n_dr - dr_val_count - dr_test_count

        # If not enough DR cases, at minimum put 1 in each split
        if n_dr < 3:
            print(f"  WARNING: Only {n_dr} DR participants - cannot guarantee DR in all splits")
            dr_train_count = max(0, n_dr - 2)
            dr_val_count = min(1, n_dr - dr_train_count)
            dr_test_count = n_dr - dr_train_count - dr_val_count

        # Other positives: proportional split
        other_train_end = int(n_other * train_frac)
        other_val_end = other_train_end + int(n_other * val_frac)

        # Negatives: proportional split
        neg_train_end = int(n_neg * train_frac)
        neg_val_end = neg_train_end + int(n_neg * val_frac)

        train = (dr_list[:dr_train_count] +
                 other_pos_list[:other_train_end] +
                 neg_list[:neg_train_end])
        val = (dr_list[dr_train_count:dr_train_count + dr_val_count] +
               other_pos_list[other_train_end:other_val_end] +
               neg_list[neg_train_end:neg_val_end])
        test = (dr_list[dr_train_count + dr_val_count:] +
                other_pos_list[other_val_end:] +
                neg_list[neg_val_end:])

        return train, val, test

    train_pids, val_pids, test_pids = stratified_split_by_priority(
        dr_participants, other_positive_participants, negative_participants,
        config.train_split, config.val_split
    )

    train_examples = [ex for p in train_pids for ex in by_participant[p]]
    val_examples = [ex for p in val_pids for ex in by_participant[p]]
    test_examples = [ex for p in test_pids for ex in by_participant[p]]

    # OVERSAMPLE positive examples in training to balance classes
    print("\n  Oversampling positive cases in training...")
    positive_train = [ex for ex in train_examples if ex.has_dr or ex.has_amd or ex.has_rvo]
    negative_train = [ex for ex in train_examples if not (ex.has_dr or ex.has_amd or ex.has_rvo)]

    # Target: ~50% positive (oversample positives to match negatives)
    if positive_train and len(positive_train) < len(negative_train):
        oversample_factor = len(negative_train) // len(positive_train)
        oversampled_positive = positive_train * oversample_factor
        train_examples = negative_train + oversampled_positive
        random.shuffle(train_examples)
        print(f"    Oversampled {len(positive_train)} positives by {oversample_factor}x → {len(oversampled_positive)}")

    # Print detailed split statistics with DR highlighted
    def count_findings(examples):
        dr = sum(1 for e in examples if e.has_dr)
        amd = sum(1 for e in examples if e.has_amd)
        rvo = sum(1 for e in examples if e.has_rvo)
        return dr, amd, rvo

    train_dr, train_amd, train_rvo = count_findings(train_examples)
    val_dr, val_amd, val_rvo = count_findings(val_examples)
    test_dr, test_amd, test_rvo = count_findings(test_examples)

    print(f"\nStratified Split (DR-prioritized):")
    print(f"  TRAIN: {len(train_examples):3d} examples | DR+={train_dr:2d} AMD+={train_amd:2d} RVO+={train_rvo:2d}")
    print(f"  VAL  : {len(val_examples):3d} examples | DR+={val_dr:2d} AMD+={val_amd:2d} RVO+={val_rvo:2d}")
    print(f"  TEST : {len(test_examples):3d} examples | DR+={test_dr:2d} AMD+={test_amd:2d} RVO+={test_rvo:2d}")

    # Verify DR distribution
    if val_dr == 0 or test_dr == 0:
        print(f"\n  ⚠️ WARNING: DR cases missing from {'val' if val_dr == 0 else 'test'} split!")
        print(f"     This will affect evaluation quality.")
    else:
        print(f"\n  ✓ DR cases present in all splits (train={train_dr}, val={val_dr}, test={test_dr})")

    # Save participant splits for Stage 2 consistency
    splits_path = Path('./data/training/participant_splits.json')
    with open(splits_path, 'w') as f:
        json.dump({
            "train": train_pids,
            "val": val_pids,
            "test": test_pids,
            "note": "Stratified by positive/negative findings"
        }, f, indent=2)
    print(f"\n  Saved participant splits to {splits_path}")

    # Save ID mapping (anon_id -> original_id)
    id_mapping_path = Path('./data/training/id_mapping.json')
    with open(id_mapping_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"  Saved ID mapping to {id_mapping_path}")

    # Save manifests
    output_dir = Path('./data/training/stage1_visual')
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_stage1_manifest(examples, path):
        manifest = {
            "stage": 1,
            "task": "visual_understanding",
            "description": "Image → Retinal Findings",
            "num_examples": len(examples),
            "examples": [
                {
                    "example_id": ex.example_id,
                    "person_id": ex.person_id,
                    "eye": ex.eye,
                    "target_findings": ex.target_findings,
                    "has_dr": ex.has_dr,
                    "has_amd": ex.has_amd,
                    "has_rvo": ex.has_rvo,
                }
                for ex in examples
            ],
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    save_stage1_manifest(train_examples, output_dir / "train_manifest.json")
    save_stage1_manifest(val_examples, output_dir / "val_manifest.json")
    save_stage1_manifest(test_examples, output_dir / "test_manifest.json")

    # Save class distribution for analysis
    summary = {
        "total_examples": len(examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
        "class_distribution": {
            "diabetic_retinopathy": sum(1 for ex in examples if ex.has_dr),
            "amd": sum(1 for ex in examples if ex.has_amd),
            "rvo": sum(1 for ex in examples if ex.has_rvo),
            "no_findings": sum(1 for ex in examples if not (ex.has_dr or ex.has_amd or ex.has_rvo)),
        }
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print("\n" + "=" * 60)
    print("STAGE 1 DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
