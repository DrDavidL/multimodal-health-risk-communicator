#!/usr/bin/env python3
"""Stage 2: Prepare training data with probabilistic DR findings.

This version uses ground truth DR labels to assign realistic P(DR) values,
then has GPT-5 generate reports that communicate findings probabilistically.

NO IMAGE PROCESSING NEEDED - GPT-5.2 only sees text inputs.

Key approach:
- DR+ patients: Assign P(DR) in range 0.5-0.95 (variety for training)
- DR- patients: Assign P(DR) in range 0.01-0.25 (mostly low, some borderline)
- This teaches the model to generate appropriate reports for different probability levels
"""

import sys
from pathlib import Path
import json
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.dataset import TrainingConfig, anonymize_summary
from src.training.retinal_findings import load_retinal_findings
from scripts.azure_query import query, load_env


def assign_p_dr(has_dr: bool, seed: int) -> tuple[float, str, str]:
    """Assign realistic P(DR) value based on ground truth.

    Returns:
        Tuple of (p_dr, predicted_grade, urgency)
    """
    rng = random.Random(seed)

    if has_dr:
        # DR+ patients: mostly high probability, some moderate
        # This reflects that the model should detect most true cases
        p_dr = rng.uniform(0.50, 0.95)

        # Assign grade based on probability (higher P = likely more severe)
        if p_dr >= 0.8:
            grade = rng.choice(["C", "D", "E"])  # Moderate to proliferative
        elif p_dr >= 0.6:
            grade = rng.choice(["B", "C"])  # Mild to moderate
        else:
            grade = "B"  # Mild
    else:
        # DR- patients: mostly low probability, some borderline for training variety
        roll = rng.random()
        if roll < 0.7:
            # 70%: clearly negative
            p_dr = rng.uniform(0.01, 0.10)
        elif roll < 0.9:
            # 20%: low but not negligible
            p_dr = rng.uniform(0.10, 0.20)
        else:
            # 10%: borderline (false positive territory - important for training)
            p_dr = rng.uniform(0.20, 0.35)

        grade = "A"  # No apparent retinopathy

    # Assign urgency based on P(DR)
    if p_dr >= 0.7:
        urgency = "urgent"
    elif p_dr >= 0.3:
        urgency = "moderate"
    else:
        urgency = "routine"

    return p_dr, grade, urgency


def generate_probabilistic_report_with_gpt5(
    dr_probability: float,
    dr_grade: str,
    urgency: str,
    clinical_context: str,
    cgm_context: str,
    has_amd: bool = False,
    has_rvo: bool = False,
) -> str:
    """Generate patient report with probabilistic DR communication."""
    load_env()

    # Convert probability to natural frequency
    n_out_of_10 = round(dr_probability * 10)

    # Determine certainty language
    if dr_probability >= 0.7:
        certainty = "likely"
        confidence_phrase = "The screening found clear signs"
    elif dr_probability >= 0.3:
        certainty = "possible"
        confidence_phrase = "The screening found some signs"
    else:
        certainty = "unlikely"
        confidence_phrase = "The screening did not find significant signs"

    # Map grade to description
    grade_descriptions = {
        "A": "no apparent diabetic retinopathy",
        "B": "mild early-stage changes",
        "C": "moderate changes",
        "D": "more advanced changes",
        "E": "advanced proliferative changes",
    }
    grade_desc = grade_descriptions.get(dr_grade, "some changes")

    # Build other findings context
    other_findings = []
    if has_amd:
        other_findings.append("age-related macular degeneration (AMD)")
    if has_rvo:
        other_findings.append("retinal vascular changes")

    other_findings_text = ""
    if other_findings:
        other_findings_text = f"\nOther findings detected: {', '.join(other_findings)}"

    # Urgency recommendations
    urgency_guidance = {
        "urgent": "Schedule an appointment with an eye specialist within 2 weeks.",
        "moderate": "Discuss these findings at your next doctor visit, or schedule an eye exam within 1-2 months.",
        "routine": "Continue your regular yearly eye exams as recommended for people with diabetes.",
    }
    urgency_text = urgency_guidance.get(urgency, urgency_guidance["routine"])

    prompt = f"""You are creating training data for a medical AI that helps patients understand their health.
Generate a patient-friendly report that explains diabetic retinopathy screening results PROBABILISTICALLY.

CRITICAL REQUIREMENTS:
1. Explain the probability using NATURAL FREQUENCIES (e.g., "X out of 10 people"), not percentages
2. Be clear this is a SCREENING result, not a definitive diagnosis
3. Acknowledge uncertainty while still being helpful
4. Use simple language (8th grade reading level)
5. Avoid medical jargon - explain any terms used
6. Be warm and supportive, not alarming

SCREENING RESULTS:
- Probability of diabetic retinopathy: {dr_probability:.1%} (about {n_out_of_10} out of 10)
- Screening assessment: Diabetic retinopathy is {certainty}
- Grade observed: {grade_desc}
- Urgency level: {urgency.upper()}
- Recommended action: {urgency_text}{other_findings_text}

CLINICAL INFORMATION:
{clinical_context}

GLUCOSE MONITORING:
{cgm_context}

Generate a report with these sections:

## Understanding Your Retinal Screening Results
(2-3 paragraphs explaining what the screening found and what the probability means.
Use the natural frequency: "If we screened 10 people with similar results, about {n_out_of_10}
would have diabetic retinopathy and {10 - n_out_of_10} would not."
Emphasize this is screening, not diagnosis.)

## Connecting Your Eye Health to Your Diabetes
(1-2 paragraphs connecting glucose patterns to eye health. Reference specific CGM data if available.)

## What You Should Do Next
(Clear, specific action items based on the urgency level. Include the recommendation above.)

## Key Points to Remember
(3-4 bullet points summarizing the most important takeaways)

## Questions to Ask Your Eye Doctor
(2-3 specific questions the patient should ask)

Keep total response under 700 words. Focus on being helpful while honestly communicating uncertainty."""

    return query(prompt)


def main():
    print("=" * 60)
    print("STAGE 2: Probabilistic Report Generation Training Data")
    print("Using ground truth labels + GPT-5.2 (no image processing)")
    print("=" * 60)

    # Load participant IDs
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"\nTotal participants in file: {len(person_ids)}")

    # Load ID mapping (anon_id -> original_id)
    id_mapping_path = Path('./data/training/id_mapping.json')
    if id_mapping_path.exists():
        with open(id_mapping_path) as f:
            id_mapping = json.load(f)
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        print(f"ID mapping loaded: {len(id_mapping)} entries")
    else:
        print("ERROR: id_mapping.json not found. Run prepare_stage1_data.py first.")
        return

    # Filter to participants that were processed in Stage 1
    processed_original_ids = set(id_mapping.values())
    person_ids = [pid for pid in person_ids if pid in processed_original_ids]
    print(f"Participants with Stage 1 data: {len(person_ids)}")

    # Initialize loader (for clinical/CGM data only, no images)
    loader = ParticipantLoader(cache_dir='./data', auto_download=False)

    # Load retinal findings (ground truth)
    print("\nLoading retinal findings (ground truth)...")
    retinal_findings = load_retinal_findings()

    dr_count = sum(1 for pid in person_ids if retinal_findings.get(pid, {}).get('diabetic_retinopathy'))
    print(f"  DR+ in our participants: {dr_count}")

    config = TrainingConfig(
        min_cgm_days=0,  # Don't require CGM
        require_complete=False,
        anonymize_ids=True,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
    )

    # Create examples
    print("\nCreating training examples...")
    print("(Calling GPT-5.2 for each participant - no image processing)\n")

    from dataclasses import dataclass, asdict

    @dataclass
    class Stage2ProbExample:
        example_id: str
        person_id: str
        p_dr: float
        dr_grade: str
        urgency: str
        has_dr_ground_truth: bool
        has_amd: bool
        has_rvo: bool
        clinical_context: str
        cgm_context: str
        target_report: str

    examples = []
    skipped = {"no_clinical": 0, "gpt_error": 0}

    for i, person_id in enumerate(person_ids):
        anon_id = reverse_mapping.get(person_id, f"P{1000 + i}")
        print(f"[{i+1}/{len(person_ids)}] {anon_id} ({person_id})...", end=" ", flush=True)

        try:
            # Load participant data (clinical + CGM only)
            data = loader.load(person_id)

            # Get clinical context
            clinical_context = ""
            if data.clinical:
                clinical_context = data.clinical.to_summary()
                if config.anonymize_ids:
                    clinical_context = anonymize_summary(clinical_context, person_id, anon_id)

            if not clinical_context:
                print("SKIP (no clinical)")
                skipped["no_clinical"] += 1
                continue

            # Get CGM context
            cgm_context = "No continuous glucose monitoring data available for this participant."
            if data.cgm_metrics:
                cgm_context = data.cgm_metrics.to_summary()

            # Get ground truth findings
            findings = retinal_findings.get(person_id, {})
            has_dr = findings.get("diabetic_retinopathy", False)
            has_amd = findings.get("amd", False)
            has_rvo = findings.get("rvo", False)

            # Assign P(DR) based on ground truth (with variety for training)
            seed = hash(person_id) % 2**32
            p_dr, dr_grade, urgency = assign_p_dr(has_dr, seed)

            # Generate report with GPT-5.2
            try:
                target_report = generate_probabilistic_report_with_gpt5(
                    dr_probability=p_dr,
                    dr_grade=dr_grade,
                    urgency=urgency,
                    clinical_context=clinical_context,
                    cgm_context=cgm_context,
                    has_amd=has_amd,
                    has_rvo=has_rvo,
                )
            except Exception as e:
                print(f"SKIP (GPT error: {e})")
                skipped["gpt_error"] += 1
                continue

            example = Stage2ProbExample(
                example_id=f"{anon_id}_prob_report",
                person_id=anon_id,
                p_dr=p_dr,
                dr_grade=dr_grade,
                urgency=urgency,
                has_dr_ground_truth=has_dr,
                has_amd=has_amd,
                has_rvo=has_rvo,
                clinical_context=clinical_context,
                cgm_context=cgm_context,
                target_report=target_report,
            )
            examples.append(example)

            dr_marker = "DR+" if has_dr else "DR-"
            print(f"OK P(DR)={p_dr:.2f} [{urgency}] {dr_marker}")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Created {len(examples)} examples")
    print(f"Skipped: {skipped}")
    print(f"{'='*60}")

    if not examples:
        print("No examples created. Check data availability.")
        return

    # Use Stage 1 splits for consistency
    splits_path = Path('./data/training/participant_splits.json')
    if splits_path.exists():
        print("\nUsing Stage 1 participant splits")
        with open(splits_path) as f:
            splits = json.load(f)
        train_pids = set(splits["train"])
        val_pids = set(splits["val"])
        test_pids = set(splits["test"])
    else:
        print("\nWARNING: Stage 1 splits not found, creating new splits")
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
    output_dir = Path('./data/training/stage2_probabilistic')
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(examples, path):
        manifest = {
            "stage": 2,
            "task": "probabilistic_report_generation",
            "description": "(P(DR) + Clinical + CGM) → Probabilistic Patient Report",
            "notes": [
                "P(DR) assigned from ground truth labels with variety for training",
                "Reports communicate uncertainty using natural frequencies",
                "Urgency levels: urgent (P≥0.7), moderate (0.3≤P<0.7), routine (P<0.3)",
                "No image processing - text inputs only",
            ],
            "num_examples": len(examples),
            "examples": [asdict(ex) for ex in examples],
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    save_manifest(train_examples, output_dir / "train_manifest.json")
    save_manifest(val_examples, output_dir / "val_manifest.json")
    save_manifest(test_examples, output_dir / "test_manifest.json")

    # Save summary
    p_dr_values = [ex.p_dr for ex in examples]
    summary = {
        "total_examples": len(examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
        "ground_truth_dr_positive": sum(1 for ex in examples if ex.has_dr_ground_truth),
        "p_dr_distribution": {
            "min": min(p_dr_values),
            "max": max(p_dr_values),
            "mean": sum(p_dr_values) / len(p_dr_values),
            "urgent_count": sum(1 for ex in examples if ex.urgency == "urgent"),
            "moderate_count": sum(1 for ex in examples if ex.urgency == "moderate"),
            "routine_count": sum(1 for ex in examples if ex.urgency == "routine"),
        },
        "skipped": skipped,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print(f"\nGround truth DR+: {summary['ground_truth_dr_positive']}")
    print(f"\nP(DR) distribution:")
    print(f"  Min:  {summary['p_dr_distribution']['min']:.3f}")
    print(f"  Max:  {summary['p_dr_distribution']['max']:.3f}")
    print(f"  Mean: {summary['p_dr_distribution']['mean']:.3f}")
    print(f"\nUrgency distribution:")
    print(f"  Urgent:   {summary['p_dr_distribution']['urgent_count']}")
    print(f"  Moderate: {summary['p_dr_distribution']['moderate_count']}")
    print(f"  Routine:  {summary['p_dr_distribution']['routine_count']}")

    print("\n" + "=" * 60)
    print("STAGE 2 PROBABILISTIC DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
