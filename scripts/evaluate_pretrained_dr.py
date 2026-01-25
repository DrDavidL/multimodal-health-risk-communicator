#!/usr/bin/env python3
"""Evaluate pre-trained DR LoRA from HuggingFace on AI-READI test set.

Uses: qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy
"""

import sys
from pathlib import Path
import json
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from src.loaders.dicom_loader import DICOMLoader

# DR classification prompt from the model card
SEMANTIC_CLASS_DESCRIPTIONS = [
    "A: No apparent retinopathy (No DR)",
    "B: Mild nonproliferative diabetic retinopathy (Mild NPDR)",
    "C: Moderate nonproliferative diabetic retinopathy (Moderate NPDR)",
    "D: Severe nonproliferative diabetic retinopathy (Severe NPDR)",
    "E: Proliferative diabetic retinopathy (PDR)"
]

OPTIONS_FOR_PROMPT = "\n".join(SEMANTIC_CLASS_DESCRIPTIONS)
DR_PROMPT = f"Based on the fundus image, what is the stage of diabetic retinopathy?\n{OPTIONS_FOR_PROMPT}"


def parse_dr_grade(response: str) -> tuple[str, bool]:
    """Parse DR grade from model response.

    Returns (grade, has_dr) where has_dr is True for grades B-E.
    """
    response_upper = response.upper()

    # Look for grade letters
    if "A:" in response_upper or "NO APPARENT" in response_upper or "NO DR" in response_upper:
        return "A", False
    elif "B:" in response_upper or "MILD NPDR" in response_upper or "MILD NONPROLIFERATIVE" in response_upper:
        return "B", True
    elif "C:" in response_upper or "MODERATE NPDR" in response_upper or "MODERATE NONPROLIFERATIVE" in response_upper:
        return "C", True
    elif "D:" in response_upper or "SEVERE NPDR" in response_upper or "SEVERE NONPROLIFERATIVE" in response_upper:
        return "D", True
    elif "E:" in response_upper or "PDR" in response_upper or "PROLIFERATIVE" in response_upper:
        return "E", True

    # Default to checking for any DR mention
    if "RETINOPATHY" in response_upper and "NO" not in response_upper:
        return "?", True

    return "?", False


def main():
    print("=" * 60)
    print("EVALUATING PRE-TRAINED DR MODEL")
    print("Model: qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy")
    print("=" * 60)

    # Paths
    test_manifest = Path("./data/training/stage1_visual/test_manifest.json")

    # Load test manifest
    with open(test_manifest) as f:
        test_data = json.load(f)
    print(f"\nTest set: {test_data['num_examples']} examples")

    # Load ID mapping
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

    # Load the pre-trained DR LoRA adapter
    print(f"  Loading adapter from {adapter_id}...")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()

    torch.mps.synchronize()
    print(f"  Model loaded, using {torch.mps.current_allocated_memory()/1e9:.1f}GB")

    # Initialize loader
    dicom_loader = DICOMLoader()

    # Evaluate
    results = []
    tp, fp, tn, fn = 0, 0, 0, 0

    print("\nRunning inference on test set...")
    for i, example in enumerate(test_data["examples"]):
        print(f"\n[{i+1}/{len(test_data['examples'])}] {example['person_id']} ({example['eye']} eye)")

        # Load image
        original_id = id_mapping.get(example["person_id"], example["person_id"].replace("P", ""))
        eye_letter = "l" if example["eye"] == "left" else "r"
        pattern = f"./data/retinal_photography/cfp/icare_eidon/{original_id}/*_{eye_letter}_*.dcm"
        matches = glob.glob(pattern)

        if not matches:
            print("  ⚠ Image not found, skipping")
            continue

        image = dicom_loader.load(matches[0])

        # Generate prediction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DR_PROMPT},
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
        inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response
        if "model" in response.lower():
            response = response.split("model")[-1]

        # Parse prediction
        grade, has_dr_pred = parse_dr_grade(response)
        has_dr_true = example["has_dr"]

        print(f"  Ground truth: DR={has_dr_true}")
        print(f"  Predicted:    DR={has_dr_pred} (Grade: {grade})")
        print(f"  Response: {response[:200]}...")

        # Score
        if has_dr_true and has_dr_pred:
            tp += 1
        elif has_dr_true and not has_dr_pred:
            fn += 1
        elif not has_dr_true and has_dr_pred:
            fp += 1
        else:
            tn += 1

        results.append({
            "person_id": example["person_id"],
            "eye": example["eye"],
            "ground_truth_dr": has_dr_true,
            "predicted_dr": has_dr_pred,
            "predicted_grade": grade,
            "response": response[:500],
        })

        torch.mps.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Pre-trained DR Model)")
    print("=" * 60)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")

    print(f"\nMetrics:")
    print(f"  Sensitivity (Recall): {sensitivity*100:.1f}% ({tp}/{tp+fn})")
    print(f"  Specificity:          {specificity*100:.1f}% ({tn}/{tn+fp})")
    print(f"  Accuracy:             {accuracy*100:.1f}%")

    # Grade distribution
    grades = [r["predicted_grade"] for r in results]
    print(f"\nPredicted Grade Distribution:")
    for g in ["A", "B", "C", "D", "E", "?"]:
        count = grades.count(g)
        if count > 0:
            print(f"  {g}: {count}")

    # Save results
    output_path = Path("./outputs/medgemma-stage1/pretrained_dr_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": adapter_id,
            "metrics": {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            },
            "details": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
