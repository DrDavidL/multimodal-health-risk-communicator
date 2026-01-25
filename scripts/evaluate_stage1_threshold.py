#!/usr/bin/env python3
"""Evaluate Stage 1 model with threshold tuning for sensitivity.

Extracts confidence scores from model logits to enable threshold-based
detection that prioritizes sensitivity (not missing disease).
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Direct yes/no prompts for each finding
DR_PROMPT = """Examine this retinal fundus image carefully.

Does this image show signs of diabetic retinopathy (microaneurysms, hemorrhages, exudates, or neovascularization)?

Answer with just "Yes" or "No"."""

AMD_PROMPT = """Examine this retinal fundus image carefully.

Does this image show signs of age-related macular degeneration (drusen, pigment changes, or geographic atrophy)?

Answer with just "Yes" or "No"."""

RVO_PROMPT = """Examine this retinal fundus image carefully.

Does this image show signs of retinal vascular occlusion (flame hemorrhages, cotton wool spots, or venous dilation)?

Answer with just "Yes" or "No"."""


def get_yes_probability(model, processor, image, prompt, device="mps"):
    """Get probability of 'Yes' response for a given prompt."""
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
        # Get logits for next token prediction
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last position logits

        # Get token IDs for "Yes" and "No"
        yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)

        # Use first token of each (usually the full word)
        yes_id = yes_tokens[0]
        no_id = no_tokens[0]

        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        yes_prob = probs[0, yes_id].item()
        no_prob = probs[0, no_id].item()

        # Normalize to Yes/No only
        total = yes_prob + no_prob
        if total > 0:
            yes_prob_normalized = yes_prob / total
        else:
            yes_prob_normalized = 0.5

    return yes_prob_normalized, yes_prob, no_prob


def evaluate_at_threshold(results, threshold):
    """Evaluate predictions at a given threshold."""
    metrics = {"dr": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
               "amd": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
               "rvo": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}}

    for r in results:
        for condition in ["dr", "amd", "rvo"]:
            prob = r["probabilities"][condition]
            pred = prob >= threshold
            truth = r["ground_truth"][condition]

            if truth and pred:
                metrics[condition]["tp"] += 1
            elif truth and not pred:
                metrics[condition]["fn"] += 1
            elif not truth and pred:
                metrics[condition]["fp"] += 1
            else:
                metrics[condition]["tn"] += 1

    # Calculate sensitivity and specificity
    output = {}
    for condition in ["dr", "amd", "rvo"]:
        m = metrics[condition]
        sens = m["tp"] / max(m["tp"] + m["fn"], 1)
        spec = m["tn"] / max(m["tn"] + m["fp"], 1)
        acc = (m["tp"] + m["tn"]) / max(m["tp"] + m["tn"] + m["fp"] + m["fn"], 1)
        output[condition] = {"sensitivity": sens, "specificity": spec, "accuracy": acc, **m}

    return output


def find_threshold_for_sensitivity(results, target_sensitivity=0.8, condition="dr"):
    """Find threshold that achieves target sensitivity."""
    # Get all probabilities for the condition
    probs = [r["probabilities"][condition] for r in results]
    truths = [r["ground_truth"][condition] for r in results]

    # If no positive cases, can't find threshold
    if not any(truths):
        return None, None

    # Try different thresholds
    best_threshold = 0.5
    best_spec = 0.0

    for threshold in [i/100 for i in range(1, 100)]:
        preds = [p >= threshold for p in probs]

        # Calculate sensitivity
        tp = sum(1 for p, t in zip(preds, truths) if p and t)
        fn = sum(1 for p, t in zip(preds, truths) if not p and t)
        sensitivity = tp / max(tp + fn, 1)

        if sensitivity >= target_sensitivity:
            # Calculate specificity at this threshold
            tn = sum(1 for p, t in zip(preds, truths) if not p and not t)
            fp = sum(1 for p, t in zip(preds, truths) if p and not t)
            specificity = tn / max(tn + fp, 1)

            if specificity > best_spec:
                best_spec = specificity
                best_threshold = threshold

    return best_threshold, best_spec


def main():
    print("=" * 60)
    print("STAGE 1 EVALUATION: Threshold Analysis")
    print("=" * 60)

    # Paths
    adapter_path = Path("./outputs/medgemma-stage1/adapter")
    test_manifest = Path("./data/training/stage1_visual/test_manifest.json")

    if not adapter_path.exists():
        print("ERROR: Adapter not found. Run training first.")
        return

    # Load test manifest
    with open(test_manifest) as f:
        test_data = json.load(f)
    print(f"\nTest set: {test_data['num_examples']} examples")

    # Load ID mapping
    with open("./data/training/id_mapping.json") as f:
        id_mapping = json.load(f)

    # Load model
    print("\nLoading fine-tuned model...")
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    torch.mps.synchronize()
    print(f"  Model loaded, using {torch.mps.current_allocated_memory()/1e9:.1f}GB")

    # Create test dataset
    from src.loaders.dicom_loader import DICOMLoader
    import glob
    dicom_loader = DICOMLoader()

    # Collect results with probabilities
    results = []

    print("\nExtracting confidence scores...")
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

        # Get probabilities for each condition
        dr_prob, dr_yes, dr_no = get_yes_probability(model, processor, image, DR_PROMPT)
        torch.mps.empty_cache()

        amd_prob, amd_yes, amd_no = get_yes_probability(model, processor, image, AMD_PROMPT)
        torch.mps.empty_cache()

        rvo_prob, rvo_yes, rvo_no = get_yes_probability(model, processor, image, RVO_PROMPT)
        torch.mps.empty_cache()

        print(f"  Ground truth: DR={example['has_dr']}, AMD={example['has_amd']}, RVO={example['has_rvo']}")
        print(f"  P(Yes|DR):  {dr_prob:.3f}  (raw: yes={dr_yes:.4f}, no={dr_no:.4f})")
        print(f"  P(Yes|AMD): {amd_prob:.3f}  (raw: yes={amd_yes:.4f}, no={amd_no:.4f})")
        print(f"  P(Yes|RVO): {rvo_prob:.3f}  (raw: yes={rvo_yes:.4f}, no={rvo_no:.4f})")

        results.append({
            "person_id": example["person_id"],
            "eye": example["eye"],
            "ground_truth": {
                "dr": example["has_dr"],
                "amd": example["has_amd"],
                "rvo": example["has_rvo"]
            },
            "probabilities": {
                "dr": dr_prob,
                "amd": amd_prob,
                "rvo": rvo_prob
            },
            "raw_probs": {
                "dr": {"yes": dr_yes, "no": dr_no},
                "amd": {"yes": amd_yes, "no": amd_no},
                "rvo": {"yes": rvo_yes, "no": rvo_no},
            }
        })

    # Analyze thresholds
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    # Default threshold (0.5)
    print("\n--- At default threshold (0.50) ---")
    metrics_50 = evaluate_at_threshold(results, 0.50)
    for cond in ["dr", "amd", "rvo"]:
        m = metrics_50[cond]
        print(f"  {cond.upper()}: Sens={m['sensitivity']*100:.1f}%, Spec={m['specificity']*100:.1f}%, Acc={m['accuracy']*100:.1f}%")

    # Find thresholds for 80% sensitivity
    print("\n--- Thresholds for 80% sensitivity ---")
    for cond in ["dr", "amd", "rvo"]:
        threshold, spec = find_threshold_for_sensitivity(results, 0.80, cond)
        if threshold is not None:
            print(f"  {cond.upper()}: threshold={threshold:.2f} → Spec={spec*100:.1f}%")
        else:
            print(f"  {cond.upper()}: No positive cases in test set")

    # Analyze across multiple thresholds
    print("\n--- Performance across thresholds ---")
    print(f"{'Threshold':<12} {'DR Sens':<10} {'DR Spec':<10} {'AMD Sens':<10} {'AMD Spec':<10}")
    print("-" * 52)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = evaluate_at_threshold(results, thresh)
        print(f"{thresh:<12.1f} {m['dr']['sensitivity']*100:<10.1f} {m['dr']['specificity']*100:<10.1f} "
              f"{m['amd']['sensitivity']*100:<10.1f} {m['amd']['specificity']*100:<10.1f}")

    # Save results
    output_path = Path("./outputs/medgemma-stage1/threshold_analysis.json")
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "metrics_at_50": metrics_50,
            "analysis": {
                "note": "Use lower threshold to increase sensitivity at cost of specificity"
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
