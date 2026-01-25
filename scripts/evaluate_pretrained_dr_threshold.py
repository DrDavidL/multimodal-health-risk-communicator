#!/usr/bin/env python3
"""Evaluate pre-trained DR model with threshold tuning.

Extract confidence scores to enable sensitivity-optimized detection.
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

# Simple yes/no prompt for DR detection
DR_BINARY_PROMPT = """Look at this retinal fundus image.

Does this image show ANY signs of diabetic retinopathy (including mild, moderate, severe, or proliferative)?

Answer with just "Yes" or "No"."""


def get_yes_probability(model, processor, image, prompt, device="mps"):
    """Get probability of 'Yes' response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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

        # Get token IDs
        yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)

        yes_id = yes_tokens[0]
        no_id = no_tokens[0]

        probs = F.softmax(logits, dim=-1)
        yes_prob = probs[0, yes_id].item()
        no_prob = probs[0, no_id].item()

        # Normalize
        total = yes_prob + no_prob
        if total > 0:
            yes_prob_normalized = yes_prob / total
        else:
            yes_prob_normalized = 0.5

    return yes_prob_normalized, yes_prob, no_prob


def main():
    print("=" * 60)
    print("PRE-TRAINED DR MODEL: Threshold Analysis")
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

    print("\nExtracting confidence scores...")
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

        # Get probability
        prob, yes_raw, no_raw = get_yes_probability(model, processor, image, DR_BINARY_PROMPT)
        torch.mps.empty_cache()

        print(f"  Ground truth: DR={example['has_dr']}")
        print(f"  P(DR):        {prob:.3f}  (raw: yes={yes_raw:.4f}, no={no_raw:.4f})")

        results.append({
            "person_id": example["person_id"],
            "eye": example["eye"],
            "original_id": original_id,
            "ground_truth_dr": example["has_dr"],
            "probability": prob,
            "raw_yes": yes_raw,
            "raw_no": no_raw,
        })

    # Analyze thresholds
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    # Show individual results sorted by probability
    print("\nResults sorted by P(DR):")
    for r in sorted(results, key=lambda x: -x["probability"]):
        marker = "DR+" if r["ground_truth_dr"] else "DR-"
        print(f"  {r['person_id']} {r['eye']}: P={r['probability']:.3f} [{marker}]")

    # Find threshold for 100% sensitivity (need to catch both DR+ cases)
    dr_positive = [r for r in results if r["ground_truth_dr"]]
    dr_negative = [r for r in results if not r["ground_truth_dr"]]

    if dr_positive:
        min_dr_prob = min(r["probability"] for r in dr_positive)
        print(f"\nMinimum P(DR) among true DR+ cases: {min_dr_prob:.3f}")
        print(f"  → Set threshold ≤ {min_dr_prob:.3f} for 100% sensitivity")

    # Calculate metrics at different thresholds
    print("\n--- Performance at different thresholds ---")
    print(f"{'Threshold':<12} {'Sens':<8} {'Spec':<8} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4}")
    print("-" * 48)

    for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = sum(1 for r in results if r["ground_truth_dr"] and r["probability"] >= thresh)
        fn = sum(1 for r in results if r["ground_truth_dr"] and r["probability"] < thresh)
        fp = sum(1 for r in results if not r["ground_truth_dr"] and r["probability"] >= thresh)
        tn = sum(1 for r in results if not r["ground_truth_dr"] and r["probability"] < thresh)

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        print(f"{thresh:<12.2f} {sens*100:<8.1f} {spec*100:<8.1f} {tp:<4} {fp:<4} {fn:<4} {tn:<4}")

    # Save results
    output_path = Path("./outputs/medgemma-stage1/pretrained_dr_threshold.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": adapter_id,
            "results": results,
            "recommendation": f"Use threshold ≤ {min_dr_prob:.3f} for 100% sensitivity" if dr_positive else "No DR+ cases"
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
