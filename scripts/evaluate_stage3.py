#!/usr/bin/env python3
"""Stage 3: End-to-End Evaluation with Probabilistic Communication.

Tests the complete pipeline:
1. DR detection from images using pre-trained community LoRA
2. Probabilistic report generation using our Stage 2 fine-tuned model

Comparison:
- MedGemma Pipeline: Image → P(DR) → Probabilistic Report (findings inferred)
- GPT-5.2 Baseline: Ground Truth Findings → Report (findings provided)

This demonstrates:
- Visual understanding (inferring DR from images)
- Probabilistic communication (natural frequencies)
- Edge AI deployment (runs locally without cloud)
"""

import sys
from pathlib import Path
import json
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.retinal_findings import load_retinal_findings, format_retinal_findings
from src.training.dataset import anonymize_summary
from scripts.azure_query import query, load_env


@dataclass
class EvaluationResult:
    """Results for a single evaluation example."""
    example_id: str
    person_id: str

    # Ground truth
    has_dr: bool
    has_amd: bool
    has_rvo: bool

    # Model outputs
    medgemma_report: str  # From fine-tuned model (image input, no findings)
    gpt5_report: str  # From GPT-5.2 (text input, with findings)

    # Extracted findings from MedGemma report (for accuracy eval)
    medgemma_detected_dr: bool = False
    medgemma_detected_amd: bool = False
    medgemma_detected_rvo: bool = False

    # Quality scores (to be filled by evaluator)
    medgemma_quality_score: float = 0.0
    gpt5_quality_score: float = 0.0
    findings_accuracy: float = 0.0


def generate_gpt5_baseline(
    retinal_findings: str,
    clinical_context: str,
    cgm_context: str,
) -> str:
    """Generate baseline report using GPT-5.2 WITH retinal findings provided."""
    load_env()

    prompt = f"""Generate a patient-friendly health report based on this data:

{retinal_findings}

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Provide:
1. What Your Eye Exam Shows
2. How This Relates to Your Overall Health
3. Key Takeaways (3-4 bullet points)
4. Questions for Your Doctor

Keep response under 500 words."""

    return query(prompt)


def evaluate_findings_accuracy(
    report: str,
    has_dr: bool,
    has_amd: bool,
    has_rvo: bool,
) -> tuple[bool, bool, bool, float]:
    """Check if the report mentions the correct findings.

    Returns:
        Tuple of (detected_dr, detected_amd, detected_rvo, accuracy_score)
    """
    report_lower = report.lower()

    # Check for mentions of each condition
    dr_keywords = ["diabetic retinopathy", "retinopathy", "diabetes-related eye"]
    amd_keywords = ["macular degeneration", "amd", "macula"]
    rvo_keywords = ["vascular occlusion", "blockage", "rvo", "retinal occlusion"]

    detected_dr = any(kw in report_lower for kw in dr_keywords)
    detected_amd = any(kw in report_lower for kw in amd_keywords)
    detected_rvo = any(kw in report_lower for kw in rvo_keywords)

    # Calculate accuracy
    correct = 0
    total = 0

    # True positives and true negatives
    if has_dr == detected_dr:
        correct += 1
    total += 1

    if has_amd == detected_amd:
        correct += 1
    total += 1

    if has_rvo == detected_rvo:
        correct += 1
    total += 1

    accuracy = correct / total if total > 0 else 0

    return detected_dr, detected_amd, detected_rvo, accuracy


def main():
    print("=" * 60)
    print("STAGE 3: End-to-End Evaluation")
    print("MedGemma (image → findings → report) vs GPT-5.2 (text → report)")
    print("=" * 60)

    # Load test set from Stage 1 (these weren't used in training)
    stage1_test_path = Path('./data/training/stage1_visual/test_manifest.json')
    if not stage1_test_path.exists():
        print("ERROR: Stage 1 test manifest not found. Run prepare_stage1_data.py first.")
        return

    with open(stage1_test_path) as f:
        stage1_test = json.load(f)

    print(f"\nLoaded {stage1_test['num_examples']} test examples from Stage 1")

    # Load retinal findings and participant data
    loader = ParticipantLoader(cache_dir='./data', auto_download=True)
    retinal_findings = load_retinal_findings()

    # Check if fine-tuned model exists
    adapter_path = Path('./outputs/medgemma-stage2/adapter')
    if not adapter_path.exists():
        print("\nWARNING: Fine-tuned model not found at", adapter_path)
        print("Run training stages first. For now, will generate GPT-5.2 baselines only.")
        use_medgemma = False
    else:
        use_medgemma = True
        # Load fine-tuned model
        from src.training.trainer import MedGemmaFineTuner, LoRAConfig
        print("\nLoading fine-tuned MedGemma model...")
        config = LoRAConfig(output_dir=str(adapter_path.parent))
        finetuner = MedGemmaFineTuner(lora_config=config)
        finetuner.prepare_model()
        finetuner.load_adapter(adapter_path)

    results = []

    # CRITICAL: Only use test participants (never seen during training)
    # Load the canonical splits to verify
    splits_path = Path('./data/training/participant_splits.json')
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        test_participants = splits["test"]
        train_pids = set(splits["train"])
        val_pids = set(splits["val"])

        # Verify no leakage
        test_set = set(test_participants)
        if test_set & train_pids:
            print("⚠️ ERROR: Test participants overlap with training set!")
            return
        if test_set & val_pids:
            print("⚠️ ERROR: Test participants overlap with validation set!")
            return
        print(f"✓ No leakage: {len(test_participants)} test participants are disjoint from train/val")
    else:
        test_participants = list(set(ex["person_id"] for ex in stage1_test["examples"]))
    print(f"Evaluating {len(test_participants)} test participants")

    # Load ID mapping to get original person_ids
    with open('./data/training/id_mapping.json') as f:
        id_mapping = json.load(f)

    # Reverse mapping: anon_id -> original_id
    reverse_mapping = {v: k for k, v in id_mapping.items()}

    for i, anon_id in enumerate(test_participants[:10]):  # Limit for demo
        print(f"\n[{i+1}/{min(10, len(test_participants))}] Evaluating {anon_id}...")

        # Get original person_id
        original_id = id_mapping.get(anon_id, anon_id.replace("P", ""))

        try:
            data = loader.load(original_id)

            # Get contexts
            clinical_context = data.clinical.to_summary() if data.clinical else ""
            clinical_context = anonymize_summary(clinical_context, original_id, anon_id)
            cgm_context = data.cgm_metrics.to_summary() if data.cgm_metrics else ""

            # Get ground truth findings
            findings = retinal_findings.get(original_id, {})
            retinal_context = format_retinal_findings(findings)

            # Generate GPT-5.2 baseline (WITH findings)
            print("  Generating GPT-5.2 baseline...")
            gpt5_report = generate_gpt5_baseline(
                retinal_context, clinical_context, cgm_context
            )

            # Generate MedGemma report (image input, NO findings)
            medgemma_report = ""
            if use_medgemma:
                print("  Generating MedGemma report (from image)...")
                image = data.fundus_left or data.fundus_right
                if image:
                    # Create prompt WITHOUT retinal findings
                    prompt = f"""Based on the retinal image and health data below,
provide a patient-friendly explanation of the findings.

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Explain what you see in the eye image and how it relates to overall health.
Include: 1) Eye exam findings, 2) Health connections, 3) Key takeaways, 4) Questions for doctor."""

                    medgemma_report = finetuner.generate(image, prompt)

            # Evaluate findings accuracy in MedGemma report
            detected_dr, detected_amd, detected_rvo, accuracy = evaluate_findings_accuracy(
                medgemma_report,
                findings.get("diabetic_retinopathy", False),
                findings.get("amd", False),
                findings.get("rvo", False),
            )

            result = EvaluationResult(
                example_id=f"{anon_id}_eval",
                person_id=anon_id,
                has_dr=findings.get("diabetic_retinopathy", False),
                has_amd=findings.get("amd", False),
                has_rvo=findings.get("rvo", False),
                medgemma_report=medgemma_report,
                gpt5_report=gpt5_report,
                medgemma_detected_dr=detected_dr,
                medgemma_detected_amd=detected_amd,
                medgemma_detected_rvo=detected_rvo,
                findings_accuracy=accuracy,
            )
            results.append(result)

            print(f"  Ground truth: DR={result.has_dr}, AMD={result.has_amd}, RVO={result.has_rvo}")
            print(f"  MedGemma detected: DR={detected_dr}, AMD={detected_amd}, RVO={detected_rvo}")
            print(f"  Findings accuracy: {accuracy:.1%}")

        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    output_dir = Path('./outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    if results:
        avg_accuracy = sum(r.findings_accuracy for r in results) / len(results)
        dr_sensitivity = sum(1 for r in results if r.has_dr and r.medgemma_detected_dr) / max(1, sum(1 for r in results if r.has_dr))
        dr_specificity = sum(1 for r in results if not r.has_dr and not r.medgemma_detected_dr) / max(1, sum(1 for r in results if not r.has_dr))

        summary = {
            "num_evaluated": len(results),
            "average_findings_accuracy": avg_accuracy,
            "diabetic_retinopathy": {
                "sensitivity": dr_sensitivity,
                "specificity": dr_specificity,
            },
            "model_comparison": {
                "medgemma": "Image input (no findings provided)",
                "gpt5": "Text input (findings provided)",
            }
        }

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Examples evaluated: {len(results)}")
        print(f"Average findings accuracy: {avg_accuracy:.1%}")
        print(f"DR sensitivity: {dr_sensitivity:.1%}")
        print(f"DR specificity: {dr_specificity:.1%}")

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        detailed = [
            {
                "example_id": r.example_id,
                "person_id": r.person_id,
                "ground_truth": {"dr": r.has_dr, "amd": r.has_amd, "rvo": r.has_rvo},
                "medgemma_detected": {"dr": r.medgemma_detected_dr, "amd": r.medgemma_detected_amd, "rvo": r.medgemma_detected_rvo},
                "findings_accuracy": r.findings_accuracy,
                "medgemma_report": r.medgemma_report,
                "gpt5_report": r.gpt5_report,
            }
            for r in results
        ]
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(detailed, f, indent=2)

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
