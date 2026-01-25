#!/usr/bin/env python3
"""Fine-tune MedGemma on AI-READI multimodal data.

This script orchestrates the full fine-tuning pipeline:
1. Load participant data
2. Generate training examples (optionally with GPT-5 annotations)
3. Fine-tune MedGemma with LoRA
4. Validate for memorization
5. Save adapter weights

Usage:
    # Prepare dataset only (for manual annotation)
    python scripts/finetune.py prepare --output ./data/training

    # Train with existing annotations
    python scripts/finetune.py train --data ./data/training

    # Full pipeline with GPT-5 annotation
    python scripts/finetune.py full --participants 100

Per AI-READI DUA Section 3.D, this is permitted as a "Licensee Model".
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import ParticipantLoader
from src.training.dataset import (
    TrainingConfig,
    create_training_examples,
    split_examples,
    save_dataset_manifest,
)
from src.training.trainer import MedGemmaFineTuner, LoRAConfig, run_memorization_check


def get_participant_ids(loader: ParticipantLoader, n: int) -> list[str]:
    """Get list of participant IDs with complete data.

    Args:
        loader: ParticipantLoader instance.
        n: Maximum number of participants.

    Returns:
        List of person_id strings.
    """
    # Ensure metadata is available
    loader.ensure_metadata()
    loader.ensure_clinical_data()

    # Get participants with complete data
    complete = loader.get_complete_participants()

    # Limit to requested number
    return complete[:n]


def generate_target_with_gpt5(image, clinical_context: str, cgm_context: str) -> str:
    """Generate target response using GPT-5 via Azure.

    This creates high-quality training targets for fine-tuning.

    Args:
        image: PIL Image (not used directly by GPT-5, but included for API consistency).
        clinical_context: Clinical summary text.
        cgm_context: CGM summary text.

    Returns:
        Generated patient-friendly response.
    """
    from scripts.azure_query import query, load_env

    load_env()

    prompt = f"""You are creating training data for a medical AI model. Generate a patient-friendly
health explanation based on the following clinical data. The explanation should be:
- Written for someone with no medical background
- Warm and supportive in tone
- Clear about what findings mean
- Practical with actionable insights

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Generate a response with these sections:
1. What Your Health Data Shows (2-3 paragraphs)
2. Key Points to Remember (3-4 bullet points)
3. Questions to Discuss with Your Doctor (2-3 questions)

Keep the total response under 500 words."""

    return query(prompt)


def prepare_dataset(args):
    """Prepare training dataset from participant data."""
    print("=" * 60)
    print("PREPARING TRAINING DATASET")
    print("=" * 60)

    # Initialize loader
    loader = ParticipantLoader(
        cache_dir=args.cache_dir,
        auto_download=args.auto_download,
    )

    # Get participant IDs
    print(f"\nFinding participants with complete data...")
    person_ids = get_participant_ids(loader, args.max_participants)
    print(f"Found {len(person_ids)} complete participants")

    # Configure dataset creation
    config = TrainingConfig(
        min_cgm_days=args.min_cgm_days,
        require_complete=True,
        anonymize_ids=True,
        jitter_values=True,
        use_both_eyes=args.both_eyes,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=args.seed,
    )

    # Create examples
    print(f"\nCreating training examples...")

    # Optionally use GPT-5 to generate target responses
    response_generator = None
    if args.generate_targets:
        print("Using GPT-5 to generate target responses (this may take a while)...")
        response_generator = generate_target_with_gpt5

    examples = create_training_examples(
        loader=loader,
        person_ids=person_ids,
        config=config,
        response_generator=response_generator,
    )

    print(f"Created {len(examples)} training examples")

    # Split into train/val/test
    train_examples, val_examples, test_examples = split_examples(examples, config)

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")

    # Save manifests
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_manifest(train_examples, output_dir / "train_manifest.json")
    save_dataset_manifest(val_examples, output_dir / "val_manifest.json")
    save_dataset_manifest(test_examples, output_dir / "test_manifest.json")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "num_participants": len(person_ids),
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "num_test": len(test_examples),
            "config": {
                "min_cgm_days": config.min_cgm_days,
                "anonymize_ids": config.anonymize_ids,
                "jitter_values": config.jitter_values,
                "use_both_eyes": config.use_both_eyes,
            }
        }, f, indent=2)

    print(f"\nDataset prepared and saved to: {output_dir}")

    return train_examples, val_examples, test_examples


def train_model(args, train_examples=None, val_examples=None):
    """Fine-tune MedGemma on the prepared dataset."""
    print("\n" + "=" * 60)
    print("FINE-TUNING MEDGEMMA")
    print("=" * 60)

    # Load examples if not provided
    if train_examples is None:
        print("Loading prepared dataset...")
        # In practice, we'd load serialized examples here
        raise NotImplementedError("Loading serialized examples not yet implemented")

    # Configure LoRA
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # Initialize fine-tuner
    finetuner = MedGemmaFineTuner(lora_config=lora_config)

    # Prepare model with LoRA
    print("\nPreparing model...")
    finetuner.prepare_model()

    # Train
    print("\nStarting training...")
    finetuner.train(train_examples, val_examples)

    # Save adapter
    adapter_path = Path(args.output) / "adapter"
    finetuner.save_adapter(adapter_path)

    return finetuner


def validate_model(args, finetuner, test_examples):
    """Validate the fine-tuned model for memorization."""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)

    print("\nRunning memorization check...")
    results = run_memorization_check(
        finetuner,
        test_examples[:20],  # Test on subset
        similarity_threshold=0.85,
    )

    print(f"\nMemorization Check Results:")
    print(f"  Examples tested: {results['total_tested']}")
    print(f"  Potential memorization: {results['potential_memorization']}")
    print(f"  Max similarity: {results['max_similarity']:.2%}")
    print(f"  Memorization rate: {results['memorization_rate']:.2%}")

    if results['memorization_rate'] > 0.05:
        print("\n⚠️  WARNING: High memorization rate detected!")
        print("Consider: more diverse training data, higher dropout, or differential privacy")
    else:
        print("\n✓ Memorization check passed")

    # Save results
    output_path = Path(args.output) / "memorization_check.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma on AI-READI data"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training dataset")
    prepare_parser.add_argument("--output", "-o", default="./data/training",
                                help="Output directory for dataset")
    prepare_parser.add_argument("--cache-dir", default="./data",
                                help="Data cache directory")
    prepare_parser.add_argument("--max-participants", type=int, default=100,
                                help="Maximum participants to include")
    prepare_parser.add_argument("--min-cgm-days", type=float, default=7.0,
                                help="Minimum CGM recording days")
    prepare_parser.add_argument("--both-eyes", action="store_true",
                                help="Include both eyes per participant")
    prepare_parser.add_argument("--generate-targets", action="store_true",
                                help="Generate target responses with GPT-5")
    prepare_parser.add_argument("--auto-download", action="store_true",
                                help="Auto-download from Azure")
    prepare_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", "-d", required=True,
                              help="Path to prepared dataset")
    train_parser.add_argument("--output", "-o", default="./outputs/medgemma-lora",
                              help="Output directory for model")
    train_parser.add_argument("--lora-r", type=int, default=16,
                              help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=32,
                              help="LoRA alpha")
    train_parser.add_argument("--epochs", type=int, default=3,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=1,
                              help="Batch size per device")
    train_parser.add_argument("--lr", type=float, default=2e-4,
                              help="Learning rate")

    # Full command
    full_parser = subparsers.add_parser("full", help="Full pipeline")
    full_parser.add_argument("--participants", type=int, default=50,
                             help="Number of participants")
    full_parser.add_argument("--output", "-o", default="./outputs/medgemma-lora",
                             help="Output directory")
    full_parser.add_argument("--cache-dir", default="./data",
                             help="Data cache directory")
    full_parser.add_argument("--auto-download", action="store_true",
                             help="Auto-download from Azure")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(args)

    elif args.command == "train":
        train_model(args)

    elif args.command == "full":
        # Set up args for full pipeline
        args.max_participants = args.participants
        args.min_cgm_days = 7.0
        args.both_eyes = True
        args.generate_targets = True
        args.seed = 42
        args.lora_r = 16
        args.lora_alpha = 32
        args.epochs = 3
        args.batch_size = 1
        args.lr = 2e-4

        # Prepare dataset
        train_examples, val_examples, test_examples = prepare_dataset(args)

        # Train
        finetuner = train_model(args, train_examples, val_examples)

        # Validate
        validate_model(args, finetuner, test_examples)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
