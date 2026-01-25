#!/usr/bin/env python3
"""Prepare training data from selected participants.

Downloads data from Azure and generates target responses with GPT-5.2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.dataset import (
    TrainingConfig,
    create_training_examples,
    split_examples,
    save_dataset_manifest,
)
import json


def load_retinal_findings() -> dict[str, dict]:
    """Load retinal findings from condition_occurrence.csv.

    Returns dict mapping person_id to findings:
    {
        "1001": {"diabetic_retinopathy": True, "amd": False, "rvo": False},
        ...
    }
    """
    import pandas as pd
    path = "./data/clinical_data/condition_occurrence.csv"

    df = pd.read_csv(path, dtype={"person_id": str})

    # Filter to retinal findings rows
    retinal_patterns = ["mhoccur_pdr", "mhoccur_amd", "mhoccur_rvo"]
    retinal_df = df[df["condition_source_value"].str.contains("|".join(retinal_patterns), na=False)]

    findings = {}
    for _, row in retinal_df.iterrows():
        pid = row["person_id"]
        if pid not in findings:
            findings[pid] = {"diabetic_retinopathy": False, "amd": False, "rvo": False}

        source = str(row["condition_source_value"])
        if "mhoccur_pdr" in source:
            findings[pid]["diabetic_retinopathy"] = True
        elif "mhoccur_amd" in source:
            findings[pid]["amd"] = True
        elif "mhoccur_rvo" in source:
            findings[pid]["rvo"] = True

    return findings


def format_retinal_findings(findings: dict) -> str:
    """Format retinal findings for prompt context."""
    if not findings:
        return "Retinal Eye Exam: No significant findings documented."

    conditions = []
    if findings.get("diabetic_retinopathy"):
        conditions.append("Diabetic retinopathy (diabetes-related changes in the retina)")
    if findings.get("amd"):
        conditions.append("Age-related macular degeneration (AMD)")
    if findings.get("rvo"):
        conditions.append("Retinal vascular occlusion (blockage in retinal blood vessels)")

    if not conditions:
        return "Retinal Eye Exam: No significant findings documented."

    return "Retinal Eye Exam Findings:\n- " + "\n- ".join(conditions)


def generate_target_with_gpt5(
    image,
    clinical_context: str,
    cgm_context: str,
    retinal_context: str,
) -> str:
    """Generate target response using GPT-5 via Azure.

    Now includes retinal findings metadata so GPT-5.2 can describe
    what was found in the eye exam.
    """
    from scripts.azure_query import query, load_env

    load_env()

    prompt = f"""You are creating training data for a medical AI model that analyzes retinal images
alongside clinical data. Generate a patient-friendly health explanation based on ALL the data below.

The explanation should be:
- Written for someone with no medical background
- Warm and supportive in tone
- Clear about what the eye exam findings mean
- Explain how eye health relates to diabetes and overall health
- Practical with actionable insights

{retinal_context}

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Generate a response with these sections:
1. What Your Eye Exam Shows (1-2 paragraphs explaining the retinal findings)
2. How This Relates to Your Overall Health (1-2 paragraphs connecting eye, glucose, and clinical data)
3. Key Points to Remember (3-4 bullet points)
4. Questions to Discuss with Your Doctor (2-3 questions)

IMPORTANT: If retinal findings show diabetic retinopathy, explain what this means in simple terms
and how it relates to glucose control. If no significant eye findings, still explain the value
of regular eye exams for someone managing diabetes.

Keep the total response under 600 words."""

    return query(prompt)


def main():
    # Load selected participant IDs
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"Preparing training data for {len(person_ids)} participants")
    print("=" * 60)

    # Initialize loader with auto-download
    loader = ParticipantLoader(
        cache_dir='./data',
        auto_download=True,
    )

    # Download metadata and clinical data first
    print("\nStep 1: Downloading shared metadata and clinical data...")
    loader.ensure_metadata()
    loader.ensure_clinical_data()

    # Load retinal findings for all participants
    print("\nStep 1b: Loading retinal findings metadata...")
    retinal_findings = load_retinal_findings()
    dr_count = sum(1 for f in retinal_findings.values() if f.get("diabetic_retinopathy"))
    amd_count = sum(1 for f in retinal_findings.values() if f.get("amd"))
    rvo_count = sum(1 for f in retinal_findings.values() if f.get("rvo"))
    print(f"  Found retinal findings: DR={dr_count}, AMD={amd_count}, RVO={rvo_count}")

    # Configure dataset creation
    config = TrainingConfig(
        min_cgm_days=5.0,  # Slightly relaxed for more data
        require_complete=True,
        anonymize_ids=True,
        jitter_values=True,
        use_both_eyes=True,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
    )

    # Create examples with progress tracking
    print(f"\nStep 2: Loading participant data and generating targets...")
    print("(This downloads retinal/CGM data and calls GPT-5.2 for each)")
    print()

    # Define a progress-tracking wrapper for the response generator
    progress = {"current": 0, "total": len(person_ids)}

    def generate_target_with_progress(image, clinical_context: str, cgm_context: str) -> str:
        return generate_target_with_gpt5(image, clinical_context, cgm_context)

    # Pre-download and track progress manually
    examples = []
    anon_counter = 1000

    for i, person_id in enumerate(person_ids):
        print(f"[{i+1}/{len(person_ids)}] Processing {person_id}...", end=" ", flush=True)
        try:
            data = loader.load(person_id)

            # Skip incomplete data if required
            if config.require_complete and not data.is_complete():
                print("✗ (incomplete data)")
                continue

            # Skip if CGM recording too short
            if data.cgm_metrics and data.cgm_metrics.duration_days < config.min_cgm_days:
                print(f"✗ (CGM only {data.cgm_metrics.duration_days:.1f} days)")
                continue

            # Generate anonymized ID (consistent counter)
            anon_id = f"P{anon_counter}" if config.anonymize_ids else person_id
            anon_counter += 1

            # Get context strings
            clinical_context = ""
            if data.clinical:
                clinical_context = data.clinical.to_summary()
                if config.anonymize_ids:
                    from src.training.dataset import anonymize_summary
                    clinical_context = anonymize_summary(clinical_context, person_id, anon_id)

            cgm_context = ""
            if data.cgm_metrics:
                cgm_context = data.cgm_metrics.to_summary()

            # Create examples for each available eye
            batch_count = 0
            eyes_to_process = []
            if data.fundus_left:
                eyes_to_process.append(("left", data.fundus_left))
            if data.fundus_right and config.use_both_eyes:
                eyes_to_process.append(("right", data.fundus_right))
            elif data.fundus_right and not data.fundus_left:
                eyes_to_process.append(("right", data.fundus_right))

            for eye, image in eyes_to_process:
                # Get retinal findings for this participant
                participant_findings = retinal_findings.get(person_id, {})
                retinal_context = format_retinal_findings(participant_findings)

                # Generate target response using GPT-5.2 (now with retinal findings!)
                target = generate_target_with_gpt5(
                    image, clinical_context, cgm_context, retinal_context
                )

                from src.training.dataset import TrainingExample
                example = TrainingExample(
                    example_id=f"{anon_id}_{eye}",
                    image=image,
                    clinical_context=clinical_context,
                    cgm_context=cgm_context,
                    retinal_context=retinal_context,
                    target_response=target,
                    person_id=anon_id,
                    eye=eye,
                )
                examples.append(example)
                batch_count += 1

            print(f"✓ ({batch_count} examples)")
        except Exception as e:
            print(f"✗ Error: {e}")

    print()
    print(f"Total examples created: {len(examples)}")

    # Split into train/val/test
    print("\nStep 3: Splitting into train/val/test...")
    train_examples, val_examples, test_examples = split_examples(examples, config)

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")

    # Save manifests
    output_dir = Path('./data/training')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 4: Saving to {output_dir}...")
    save_dataset_manifest(train_examples, output_dir / "train_manifest.json")
    save_dataset_manifest(val_examples, output_dir / "val_manifest.json")
    save_dataset_manifest(test_examples, output_dir / "test_manifest.json")

    # Save ID mapping (anonymized -> original) for image loading
    id_mapping = {}
    anon_counter_check = 1000
    for i, person_id in enumerate(person_ids):
        # Reconstruct the mapping based on processing order
        # Skip participants that were skipped during processing
        pass  # We'll build this from the examples

    # Build mapping from examples
    seen_anon_ids = set()
    for ex in examples:
        if ex.person_id not in seen_anon_ids:
            seen_anon_ids.add(ex.person_id)

    # Reconstruct mapping by matching order
    anon_counter_rebuild = 1000
    for person_id in person_ids:
        anon_id = f"P{anon_counter_rebuild}"
        if anon_id in seen_anon_ids:
            id_mapping[anon_id] = person_id
            anon_counter_rebuild += 1

    with open(output_dir / "id_mapping.json", "w") as f:
        json.dump(id_mapping, f, indent=2)

    # Save config and summary
    summary = {
        "num_participants": len(person_ids),
        "num_examples": len(examples),
        "num_train": len(train_examples),
        "num_val": len(val_examples),
        "num_test": len(test_examples),
        "config": {
            "min_cgm_days": config.min_cgm_days,
            "anonymize_ids": config.anonymize_ids,
            "jitter_values": config.jitter_values,
            "use_both_eyes": config.use_both_eyes,
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    main()
