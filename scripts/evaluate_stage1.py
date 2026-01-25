#!/usr/bin/env python3
"""Evaluate Stage 1 model on held-out test set.

Measures whether the fine-tuned model actually learned to detect
retinal findings (DR, AMD, RVO) from images.
"""

import sys
from pathlib import Path
import json
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from scripts.run_stage1_training import Stage1Dataset, STAGE1_PROMPT


def parse_findings(text: str) -> dict:
    """Extract Yes/No findings from model output."""
    text_lower = text.lower()

    # Look for explicit Yes/No patterns
    dr = None
    amd = None
    rvo = None

    # Diabetic retinopathy
    if re.search(r"diabetic retinopathy[:\s]*yes", text_lower):
        dr = True
    elif re.search(r"diabetic retinopathy[:\s]*no", text_lower):
        dr = False
    elif "diabetic retinopathy" in text_lower or "retinopathy" in text_lower:
        # Mentioned but not clear yes/no - check context
        dr = "no signs" not in text_lower and "not present" not in text_lower

    # AMD
    if re.search(r"macular degeneration[:\s]*yes", text_lower):
        amd = True
    elif re.search(r"macular degeneration[:\s]*no", text_lower):
        amd = False
    elif "macular degeneration" in text_lower or "amd" in text_lower:
        amd = "no signs" not in text_lower and "not present" not in text_lower

    # RVO
    if re.search(r"vascular occlusion[:\s]*yes", text_lower):
        rvo = True
    elif re.search(r"vascular occlusion[:\s]*no", text_lower):
        rvo = False
    elif "vascular occlusion" in text_lower or "rvo" in text_lower:
        rvo = "no signs" not in text_lower and "not present" not in text_lower

    return {"dr": dr, "amd": amd, "rvo": rvo}


def main():
    print("=" * 60)
    print("STAGE 1 EVALUATION: Visual Understanding")
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
    dicom_loader = DICOMLoader()

    # Evaluate
    results = []
    correct_dr, correct_amd, correct_rvo = 0, 0, 0
    total_dr, total_amd, total_rvo = 0, 0, 0

    print("\nRunning inference on test set...")
    for i, example in enumerate(test_data["examples"]):
        print(f"\n[{i+1}/{len(test_data['examples'])}] {example['person_id']} ({example['eye']} eye)")

        # Load image
        import glob
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
                    {"type": "text", "text": STAGE1_PROMPT},
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
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response (after the prompt)
        if "model" in response.lower():
            response = response.split("model")[-1]

        print(f"  Ground truth: DR={example['has_dr']}, AMD={example['has_amd']}, RVO={example['has_rvo']}")

        # Parse predictions
        predictions = parse_findings(response)
        print(f"  Predicted:    DR={predictions['dr']}, AMD={predictions['amd']}, RVO={predictions['rvo']}")

        # Score
        if predictions['dr'] is not None:
            total_dr += 1
            if predictions['dr'] == example['has_dr']:
                correct_dr += 1

        if predictions['amd'] is not None:
            total_amd += 1
            if predictions['amd'] == example['has_amd']:
                correct_amd += 1

        if predictions['rvo'] is not None:
            total_rvo += 1
            if predictions['rvo'] == example['has_rvo']:
                correct_rvo += 1

        results.append({
            "person_id": example["person_id"],
            "eye": example["eye"],
            "ground_truth": {"dr": example["has_dr"], "amd": example["has_amd"], "rvo": example["has_rvo"]},
            "predicted": predictions,
            "response": response[:500],
        })

        torch.mps.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nDiabetic Retinopathy:")
    print(f"  Accuracy: {correct_dr}/{total_dr} = {correct_dr/max(total_dr,1)*100:.1f}%")

    print(f"\nAge-related Macular Degeneration:")
    print(f"  Accuracy: {correct_amd}/{total_amd} = {correct_amd/max(total_amd,1)*100:.1f}%")

    print(f"\nRetinal Vascular Occlusion:")
    print(f"  Accuracy: {correct_rvo}/{total_rvo} = {correct_rvo/max(total_rvo,1)*100:.1f}%")

    overall = correct_dr + correct_amd + correct_rvo
    total = total_dr + total_amd + total_rvo
    print(f"\nOverall Accuracy: {overall}/{total} = {overall/max(total,1)*100:.1f}%")

    # Save results
    output_path = Path("./outputs/medgemma-stage1/evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "metrics": {
                "dr_accuracy": correct_dr / max(total_dr, 1),
                "amd_accuracy": correct_amd / max(total_amd, 1),
                "rvo_accuracy": correct_rvo / max(total_rvo, 1),
                "overall_accuracy": overall / max(total, 1),
            },
            "details": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
