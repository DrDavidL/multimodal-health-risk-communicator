#!/usr/bin/env python3
"""Evaluate pre-trained DR model with grade-level probabilities.

Extract probabilities for each DR grade (A-E) to enable fine-grained threshold tuning.
"""

import sys
from pathlib import Path
import json
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from src.loaders.dicom_loader import DICOMLoader

# Grade descriptions from model card
GRADES = {
    "A": "No apparent retinopathy",
    "B": "Mild nonproliferative diabetic retinopathy",
    "C": "Moderate nonproliferative diabetic retinopathy",
    "D": "Severe nonproliferative diabetic retinopathy",
    "E": "Proliferative diabetic retinopathy",
}

DR_GRADES_PROMPT = """Based on the fundus image, what is the stage of diabetic retinopathy?
A: No apparent retinopathy (No DR)
B: Mild nonproliferative diabetic retinopathy (Mild NPDR)
C: Moderate nonproliferative diabetic retinopathy (Moderate NPDR)
D: Severe nonproliferative diabetic retinopathy (Severe NPDR)
E: Proliferative diabetic retinopathy (PDR)"""


def get_grade_probabilities(model, processor, image, device="mps"):
    """Get probability distribution over DR grades A-E."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DR_GRADES_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Get token IDs for each grade letter
        grade_probs = {}
        for grade in ["A", "B", "C", "D", "E"]:
            tokens = processor.tokenizer.encode(grade, add_special_tokens=False)
            token_id = tokens[0]
            grade_probs[grade] = F.softmax(logits, dim=-1)[0, token_id].item()

        # Normalize across grades
        total = sum(grade_probs.values())
        if total > 0:
            grade_probs = {k: v/total for k, v in grade_probs.items()}

        # Calculate P(DR) = P(B) + P(C) + P(D) + P(E)
        p_dr = grade_probs["B"] + grade_probs["C"] + grade_probs["D"] + grade_probs["E"]

    return grade_probs, p_dr


def main():
    print("=" * 60)
    print("PRE-TRAINED DR MODEL: Grade-Level Analysis")
    print("=" * 60)

    test_manifest = Path("./data/training/stage1_visual/test_manifest.json")

    with open(test_manifest) as f:
        test_data = json.load(f)
    print(f"\nTest set: {test_data['num_examples']} examples")

    with open("./data/training/id_mapping.json") as f:
        id_mapping = json.load(f)

    # Load model
    print("\nLoading pre-trained DR model...")
    base_model_id = "google/medgemma-4b-it"
    adapter_id = "qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy"

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()

    torch.mps.synchronize()
    print(f"  Model loaded, using {torch.mps.current_allocated_memory()/1e9:.1f}GB")

    dicom_loader = DICOMLoader()

    # Collect results
    results = []

    print("\nExtracting grade probabilities...")
    for i, example in enumerate(test_data["examples"]):
        print(f"\n[{i+1}/{len(test_data['examples'])}] {example['person_id']} ({example['eye']} eye)")

        original_id = id_mapping.get(example["person_id"], example["person_id"].replace("P", ""))
        eye_letter = "l" if example["eye"] == "left" else "r"
        pattern = f"./data/retinal_photography/cfp/icare_eidon/{original_id}/*_{eye_letter}_*.dcm"
        matches = glob.glob(pattern)

        if not matches:
            print("  ⚠ Image not found, skipping")
            continue

        image = dicom_loader.load(matches[0])

        grade_probs, p_dr = get_grade_probabilities(model, processor, image)
        torch.mps.empty_cache()

        predicted_grade = max(grade_probs, key=grade_probs.get)

        print(f"  Ground truth: DR={example['has_dr']}")
        print(f"  P(DR) = {p_dr:.3f}  |  Grades: A={grade_probs['A']:.3f} B={grade_probs['B']:.3f} C={grade_probs['C']:.3f} D={grade_probs['D']:.3f} E={grade_probs['E']:.3f}")
        print(f"  Predicted grade: {predicted_grade}")

        results.append({
            "person_id": example["person_id"],
            "eye": example["eye"],
            "original_id": original_id,
            "ground_truth_dr": example["has_dr"],
            "p_dr": p_dr,
            "grade_probs": grade_probs,
            "predicted_grade": predicted_grade,
        })

    # Analyze
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Sort by P(DR)
    print("\nResults sorted by P(DR):")
    for r in sorted(results, key=lambda x: -x["p_dr"]):
        marker = "DR+" if r["ground_truth_dr"] else "DR-"
        print(f"  {r['person_id']:5} {r['eye']:5}: P(DR)={r['p_dr']:.3f}, Grade={r['predicted_grade']} [{marker}]")

    # Calculate metrics at different P(DR) thresholds
    print("\n--- Performance at different P(DR) thresholds ---")
    print(f"{'Threshold':<12} {'Sens':<8} {'Spec':<8} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4}")
    print("-" * 48)

    for thresh in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        tp = sum(1 for r in results if r["ground_truth_dr"] and r["p_dr"] >= thresh)
        fn = sum(1 for r in results if r["ground_truth_dr"] and r["p_dr"] < thresh)
        fp = sum(1 for r in results if not r["ground_truth_dr"] and r["p_dr"] >= thresh)
        tn = sum(1 for r in results if not r["ground_truth_dr"] and r["p_dr"] < thresh)

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        marker = " ← 100% sens" if sens == 1.0 else ""
        print(f"{thresh:<12.3f} {sens*100:<8.1f} {spec*100:<8.1f} {tp:<4} {fp:<4} {fn:<4} {tn:<4}{marker}")

    # Find optimal threshold for ≥80% sensitivity
    dr_positive = [r for r in results if r["ground_truth_dr"]]
    if dr_positive:
        min_dr_prob = min(r["p_dr"] for r in dr_positive)
        print(f"\n** Minimum P(DR) among true DR+ cases: {min_dr_prob:.4f}")
        print(f"** For 100% sensitivity, use threshold ≤ {min_dr_prob:.4f}")

    # Patient-level analysis
    print("\n--- Patient-Level Analysis (either eye positive → patient positive) ---")
    patients = {}
    for r in results:
        pid = r["person_id"]
        if pid not in patients:
            patients[pid] = {"max_p_dr": 0, "ground_truth": r["ground_truth_dr"], "eyes": []}
        patients[pid]["max_p_dr"] = max(patients[pid]["max_p_dr"], r["p_dr"])
        patients[pid]["eyes"].append(r)

    for pid, p in sorted(patients.items(), key=lambda x: -x[1]["max_p_dr"]):
        marker = "DR+" if p["ground_truth"] else "DR-"
        print(f"  {pid}: max P(DR)={p['max_p_dr']:.3f} [{marker}]")

    # Save results
    output_path = Path("./outputs/medgemma-stage1/pretrained_dr_grades.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": adapter_id,
            "results": results,
            "patient_level": {pid: {"max_p_dr": p["max_p_dr"], "ground_truth": p["ground_truth"]}
                            for pid, p in patients.items()}
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
