#!/usr/bin/env python3
"""Stage 1 Training: Simple Manual Loop.

Uses a manual training loop instead of HF Trainer for better MPS compatibility.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Import from the main script
from scripts.run_stage1_training import (
    Stage1Dataset,
    collate_fn,
    get_device_config,
)


def main():
    print("=" * 60)
    print("STAGE 1 TRAINING: Simple Manual Loop")
    print("=" * 60)

    # Device config
    print("\nDetecting hardware...")
    device_config = get_device_config()
    device = device_config.get("device", "cpu")

    # Paths
    cache_dir = Path("./data")
    training_dir = Path("./data/training/stage1_visual")
    output_dir = Path("./outputs/medgemma-stage1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ID mapping
    print("\nLoading ID mapping...")
    with open("./data/training/id_mapping.json") as f:
        id_mapping = json.load(f)

    # Load model
    print("\nLoading MedGemma...")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, TaskType

    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=device_config["torch_dtype"],
        device_map=device_config.get("device_map"),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if device == "mps":
        torch.mps.synchronize()
        print(f"  Model on MPS, using {torch.mps.current_allocated_memory()/1e9:.1f}GB")

    # LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=8,  # Lower rank for faster training
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # torch.compile for 10-20% speedup on MPS
    if device == "mps":
        print("  Compiling model for MPS (aot_eager backend)...")
        model = torch.compile(model, backend="aot_eager")

    # Dataset
    print("\nCreating dataset...")
    train_dataset = Stage1Dataset(
        manifest_path=training_dir / "train_manifest.json",
        cache_dir=cache_dir,
        processor=processor,
        id_mapping=id_mapping,
    )
    print(f"  {len(train_dataset)} examples")

    # DataLoader - optimized for M4 (batch_size=1 due to compiled model memory)
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # Keep at 1 - torch.compile uses more memory
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Parallel data loading
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
        weight_decay=0.01,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    model.train()
    num_epochs = 3
    grad_accum_steps = 4  # batch_size=1 × 4 = effective batch of 4

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress):
            if batch is None:
                continue

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

            # Backward
            loss.backward()

            # Update weights every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if device == "mps":
                    torch.mps.synchronize()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            progress.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

            # Periodic memory cleanup for MPS
            if device == "mps" and step % 16 == 0:
                torch.mps.empty_cache()

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Average loss: {avg_loss:.4f}")

    # Save
    print("\nSaving adapter...")
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    print(f"\n✓ Saved to {adapter_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
