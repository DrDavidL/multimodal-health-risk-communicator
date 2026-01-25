#!/usr/bin/env python3
"""Demo script for the Multimodal Health Risk Communicator.

Shows end-to-end usage from data loading to report generation.

Usage:
    python scripts/demo_report.py [person_id]

Example:
    python scripts/demo_report.py 1001
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.pipeline import ReportGenerator


def demo_data_loading(person_id: str):
    """Demonstrate data loading capabilities."""
    print("=" * 60)
    print("STEP 1: Loading Participant Data")
    print("=" * 60)

    loader = ParticipantLoader(cache_dir="./data", auto_download=False)
    data = loader.load(person_id)

    print(f"\nParticipant: {data.person_id}")
    print(f"  Has retinal images: {data.has_retinal()}")
    print(f"  Has CGM data: {data.has_cgm()}")
    print(f"  Has clinical data: {data.has_clinical()}")
    print(f"  Complete (all 3): {data.is_complete()}")

    if data.has_clinical():
        print("\n--- Clinical Summary ---")
        print(data.clinical.to_summary())

    if data.has_cgm():
        print("\n--- CGM Summary ---")
        print(data.cgm_metrics.to_summary())

    if data.has_retinal():
        print("\n--- Retinal Images ---")
        if data.fundus_left:
            print(f"  Left eye: {data.fundus_left.size} RGB")
        if data.fundus_right:
            print(f"  Right eye: {data.fundus_right.size} RGB")

    return data


def demo_prompt_context(data):
    """Demonstrate prompt context generation."""
    print("\n" + "=" * 60)
    print("STEP 2: Generated Prompt Context")
    print("=" * 60)

    context = data.to_prompt_context()
    print("\nThis context will be sent to MedGemma:")
    print("-" * 40)
    print(context)
    print("-" * 40)


def demo_report_generation(person_id: str, run_inference: bool = False):
    """Demonstrate report generation."""
    print("\n" + "=" * 60)
    print("STEP 3: Report Generation")
    print("=" * 60)

    if not run_inference:
        print("\n[Skipping MedGemma inference - pass --run-inference to enable]")
        print("\nTo generate a real report, run:")
        print(f"  python -m src.pipeline.report_generator {person_id}")
        return None

    print("\nInitializing ReportGenerator...")
    generator = ReportGenerator(cache_dir="./data", preload_model=True)

    print(f"\nGenerating report for {person_id}...")
    report = generator.generate_report(person_id)

    print("\n--- Generated Report ---")
    print(report.to_markdown())

    # Save report
    output_path = Path(f"./reports/{person_id}_report.md")
    report.save(output_path)
    print(f"\nReport saved to: {output_path}")

    return report


def main():
    """Run the demo."""
    # Parse arguments
    person_id = sys.argv[1] if len(sys.argv) > 1 else "1001"
    run_inference = "--run-inference" in sys.argv

    print("\n" + "=" * 60)
    print("MULTIMODAL HEALTH RISK COMMUNICATOR - DEMO")
    print("=" * 60)
    print(f"\nParticipant ID: {person_id}")

    # Step 1: Load data
    data = demo_data_loading(person_id)

    if not data.is_complete():
        print("\n[Warning: Participant does not have complete data]")

    # Step 2: Show prompt context
    if data.has_clinical() or data.has_cgm():
        demo_prompt_context(data)

    # Step 3: Generate report (optional)
    demo_report_generation(person_id, run_inference)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
