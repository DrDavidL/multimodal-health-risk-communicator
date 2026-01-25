#!/usr/bin/env python3
"""Run LoRA fine-tuning on MedGemma using prepared training data.

This script:
1. Loads training examples from manifests
2. Prepares MedGemma with LoRA adapters
3. Runs fine-tuning
4. Validates for memorization
5. Saves adapter weights
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset import load_training_examples_from_manifest
from src.training.trainer import MedGemmaFineTuner, LoRAConfig, run_memorization_check


def load_id_mapping() -> dict[str, str]:
    """Load ID mapping from training directory."""
    with open('./data/training/id_mapping.json') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("MEDGEMMA LORA FINE-TUNING")
    print("=" * 60)

    cache_dir = Path('./data')
    training_dir = Path('./data/training')
    output_dir = Path('./outputs/medgemma-lora')

    # Load ID mapping
    print("\nStep 1: Loading ID mapping...")
    id_mapping = load_id_mapping()
    print(f"  Mapped {len(id_mapping)} participants")

    # Load training examples
    print("\nStep 2: Loading training examples...")
    train_examples = load_training_examples_from_manifest(
        manifest_path=training_dir / "train_manifest.json",
        cache_dir=cache_dir,
        person_id_mapping=id_mapping,
    )
    print(f"  Loaded {len(train_examples)} training examples")

    val_examples = load_training_examples_from_manifest(
        manifest_path=training_dir / "val_manifest.json",
        cache_dir=cache_dir,
        person_id_mapping=id_mapping,
    )
    print(f"  Loaded {len(val_examples)} validation examples")

    if len(train_examples) < 10:
        print("ERROR: Not enough training examples. Check ID mapping.")
        return

    # Configure LoRA (optimized for M3 Mac with 64GB RAM)
    print("\nStep 3: Configuring LoRA...")
    lora_config = LoRAConfig(
        r=8,  # Lower rank for stability
        lora_alpha=16,
        lora_dropout=0.1,
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=1e-4,  # Conservative LR
        gradient_checkpointing=True,
        bf16=True,
        output_dir=str(output_dir),
        save_steps=50,
        eval_steps=25,
        logging_steps=5,
    )

    print(f"  LoRA rank: {lora_config.r}")
    print(f"  Learning rate: {lora_config.learning_rate}")
    print(f"  Epochs: {lora_config.num_epochs}")
    print(f"  Effective batch size: {lora_config.batch_size * lora_config.gradient_accumulation_steps}")

    # Initialize fine-tuner
    print("\nStep 4: Initializing MedGemma with LoRA...")
    finetuner = MedGemmaFineTuner(lora_config=lora_config)
    finetuner.prepare_model()

    # Train
    print("\nStep 5: Starting training...")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")
    print()

    finetuner.train(train_examples, val_examples)

    # Save adapter
    print("\nStep 6: Saving adapter...")
    adapter_path = output_dir / "adapter"
    finetuner.save_adapter(adapter_path)

    # Run memorization check on a subset
    print("\nStep 7: Running memorization check...")
    test_examples = load_training_examples_from_manifest(
        manifest_path=training_dir / "test_manifest.json",
        cache_dir=cache_dir,
        person_id_mapping=id_mapping,
    )

    if len(test_examples) > 0:
        results = run_memorization_check(
            finetuner,
            test_examples[:10],
            similarity_threshold=0.85,
        )
        print(f"  Examples tested: {results['total_tested']}")
        print(f"  Max similarity: {results['max_similarity']:.2%}")
        print(f"  Memorization rate: {results['memorization_rate']:.2%}")

        # Save results
        with open(output_dir / "memorization_check.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
