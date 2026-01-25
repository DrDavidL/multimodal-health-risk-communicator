#!/usr/bin/env python3
"""Stage 2: Prepare training data for report generation (text-only).

Trains the Stage 1 model to generate patient reports from text inputs.
- Input: Retinal findings text + Clinical data + CGM data (NO images)
- Output: Patient-friendly health report

This stage teaches the model to synthesize structured findings into
accessible explanations. Uses GPT-5.2 to generate target reports.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.dataset import TrainingConfig, save_dataset_manifest, anonymize_summary
from src.training.retinal_findings import load_retinal_findings, format_retinal_findings
from scripts.azure_query import query, load_env


def generate_report_with_gpt5(
    retinal_findings: str,
    clinical_context: str,
    cgm_context: str,
) -> str:
    """Generate target patient report using GPT-5.2 via Azure.

    This is text-only (no images) - GPT-5.2 synthesizes structured data
    into a patient-friendly report.
    """
    load_env()

    prompt = f"""You are creating training data for a medical AI model. Generate a patient-friendly
health report that synthesizes the following structured data. NO images are provided - only text.

The report should be:
- Written for someone with no medical background
- Warm and supportive in tone
- Clear about what findings mean and how they relate to overall health
- Practical with actionable insights

{retinal_findings}

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Generate a report with these sections:
1. Understanding Your Eye Health (1-2 paragraphs explaining retinal findings)
2. Your Overall Health Picture (1-2 paragraphs connecting eye, glucose, and clinical data)
3. Key Takeaways (3-4 bullet points)
4. Questions for Your Doctor (2-3 questions)

Keep the total response under 600 words."""

    return query(prompt)


def main():
    print("=" * 60)
    print("STAGE 2: Report Generation Training Data (Text-Only)")
    print("Task: (Retinal Findings + Clinical + CGM) → Patient Report")
    print("=" * 60)

    # Load participant IDs
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"\nPreparing data for {len(person_ids)} participants")

    # Initialize loader
    loader = ParticipantLoader(cache_dir='./data', auto_download=True)
    loader.ensure_metadata()
    loader.ensure_clinical_data()

    # Load retinal findings
    print("\nLoading retinal findings...")
    retinal_findings = load_retinal_findings()

    config = TrainingConfig(
        min_cgm_days=5.0,
        require_complete=True,  # Need all modalities for Stage 2
        anonymize_ids=True,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
    )

    # Create examples
    print("\nCreating training examples (calling GPT-5.2 for each)...")
    from dataclasses import dataclass

    @dataclass
    class Stage2Example:
        example_id: str
        retinal_context: str  # Text description of findings
        clinical_context: str
        cgm_context: str
        target_report: str  # GPT-5.2 generated report
        person_id: str
        has_dr: bool
        has_amd: bool
        has_rvo: bool

    examples = []
    anon_counter = 1000

    for i, person_id in enumerate(person_ids):
        print(f"[{i+1}/{len(person_ids)}] Processing {person_id}...", end=" ", flush=True)
        try:
            data = loader.load(person_id)

            # Skip incomplete data
            if config.require_complete and not data.is_complete():
                print("✗ (incomplete data)")
                continue

            if data.cgm_metrics and data.cgm_metrics.duration_days < config.min_cgm_days:
                print(f"✗ (CGM only {data.cgm_metrics.duration_days:.1f} days)")
                continue

            anon_id = f"P{anon_counter}" if config.anonymize_ids else person_id
            anon_counter += 1

            # Get contexts
            clinical_context = ""
            if data.clinical:
                clinical_context = data.clinical.to_summary()
                if config.anonymize_ids:
                    clinical_context = anonymize_summary(clinical_context, person_id, anon_id)

            cgm_context = ""
            if data.cgm_metrics:
                cgm_context = data.cgm_metrics.to_summary()

            # Get retinal findings
            findings = retinal_findings.get(person_id, {})
            retinal_context = format_retinal_findings(findings)

            # Generate target report with GPT-5.2
            target_report = generate_report_with_gpt5(
                retinal_context, clinical_context, cgm_context
            )

            example = Stage2Example(
                example_id=f"{anon_id}_report",
                retinal_context=retinal_context,
                clinical_context=clinical_context,
                cgm_context=cgm_context,
                target_report=target_report,
                person_id=anon_id,
                has_dr=findings.get("diabetic_retinopathy", False),
                has_amd=findings.get("amd", False),
                has_rvo=findings.get("rvo", False),
            )
            examples.append(example)
            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\nTotal Stage 2 examples: {len(examples)}")

    # CRITICAL: Use the SAME participant splits as Stage 1 to prevent leakage
    splits_path = Path('./data/training/participant_splits.json')
    if splits_path.exists():
        print("\nUsing Stage 1 participant splits (no leakage)")
        with open(splits_path) as f:
            splits = json.load(f)
        train_pids = set(splits["train"])
        val_pids = set(splits["val"])
        test_pids = set(splits["test"])
    else:
        print("\n⚠️ WARNING: Stage 1 splits not found, creating new splits")
        print("  Run prepare_stage1_data.py first to ensure consistent splits!")
        import random
        random.seed(config.random_seed)

        participants = list(set(ex.person_id for ex in examples))
        random.shuffle(participants)

        n = len(participants)
        train_end = int(n * config.train_split)
        val_end = train_end + int(n * config.val_split)

        train_pids = set(participants[:train_end])
        val_pids = set(participants[train_end:val_end])
        test_pids = set(participants[val_end:])

    train_examples = [ex for ex in examples if ex.person_id in train_pids]
    val_examples = [ex for ex in examples if ex.person_id in val_pids]
    test_examples = [ex for ex in examples if ex.person_id in test_pids]

    print(f"\nSplit: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")

    # Save manifests
    output_dir = Path('./data/training/stage2_report')
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_stage2_manifest(examples, path):
        manifest = {
            "stage": 2,
            "task": "report_generation",
            "description": "(Retinal Findings + Clinical + CGM) → Patient Report",
            "note": "Text-only inputs, no images",
            "num_examples": len(examples),
            "examples": [
                {
                    "example_id": ex.example_id,
                    "person_id": ex.person_id,
                    "retinal_context": ex.retinal_context,
                    "clinical_context": ex.clinical_context,
                    "cgm_context": ex.cgm_context,
                    "target_report": ex.target_report,
                    "has_dr": ex.has_dr,
                    "has_amd": ex.has_amd,
                    "has_rvo": ex.has_rvo,
                }
                for ex in examples
            ],
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    save_stage2_manifest(train_examples, output_dir / "train_manifest.json")
    save_stage2_manifest(val_examples, output_dir / "val_manifest.json")
    save_stage2_manifest(test_examples, output_dir / "test_manifest.json")

    # Save summary
    summary = {
        "total_examples": len(examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print("\n" + "=" * 60)
    print("STAGE 2 DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
