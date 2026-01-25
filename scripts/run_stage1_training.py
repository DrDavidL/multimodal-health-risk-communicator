#!/usr/bin/env python3
"""Stage 1 Training: Visual Understanding.

Fine-tunes MedGemma to identify retinal findings from images.
- Input: Retinal fundus image
- Output: Retinal findings description (DR, AMD, RVO)

This is a classification/detection task where the model learns to
recognize diabetic retinopathy and other conditions from the image.

Supports:
- Apple Silicon (M1/M2/M3) via MPS
- CUDA GPUs
- CPU fallback
"""

import sys
from pathlib import Path
import json
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


class MPSMemoryCallback(TrainerCallback):
    """Callback to manage MPS memory during training."""

    def __init__(self, sync_every_n_steps: int = 4):
        self.sync_every_n_steps = sync_every_n_steps
        self.is_mps = torch.backends.mps.is_available()

    def on_step_end(self, args, state, control, **kwargs):
        if self.is_mps and state.global_step % self.sync_every_n_steps == 0:
            # Synchronize to flush pending operations
            torch.mps.synchronize()
            # Empty cache periodically
            if state.global_step % (self.sync_every_n_steps * 4) == 0:
                torch.mps.empty_cache()


def get_device_config():
    """Get optimal device configuration for current hardware.

    Optimized for Apple Silicon (M1/M2/M3/M4) with unified memory.
    Set FORCE_CPU=1 environment variable to force CPU training.
    """
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

    if force_cpu:
        print("  Using CPU (forced via FORCE_CPU=1)")
        return {
            "device_map": None,
            "torch_dtype": torch.float32,
            "use_bf16": False,
            "use_fp16": False,
            "device": "cpu",
        }

    if torch.cuda.is_available():
        print("  Using CUDA GPU")
        return {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "use_bf16": True,
            "use_fp16": False,
        }
    elif torch.backends.mps.is_available():
        print("  Using Apple Silicon MPS (optimized for M-series)")
        # M4 with 64GB unified memory can handle float32 for stability
        # Load directly to MPS to avoid CPU→MPS memory doubling
        return {
            "device_map": {"": "mps"},  # Load directly to MPS
            "torch_dtype": torch.float32,
            "use_bf16": False,
            "use_fp16": False,
            "device": "mps",
        }
    else:
        print("  Using CPU")
        return {
            "device_map": None,
            "torch_dtype": torch.float32,
            "use_bf16": False,
            "use_fp16": False,
            "device": "cpu",
        }


# Stage 1 prompt template
STAGE1_PROMPT = """Analyze this retinal fundus image and identify any abnormalities.

Report your findings in this format:
- Diabetic retinopathy: [Yes/No] - [brief description if present]
- Age-related macular degeneration: [Yes/No] - [brief description if present]
- Retinal vascular occlusion: [Yes/No] - [brief description if present]

If any condition is detected, explain what you observe in the image."""


class Stage1Dataset(Dataset):
    """Dataset for Stage 1: Image → Findings."""

    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        processor,
        id_mapping: dict[str, str],
        max_length: int = 1024,
    ):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.examples = self.manifest["examples"]
        self.cache_dir = cache_dir
        self.processor = processor
        self.id_mapping = id_mapping
        self.max_length = max_length

        # Load DICOM loader
        from src.loaders.dicom_loader import DICOMLoader
        self.dicom_loader = DICOMLoader()

    def __len__(self):
        return len(self.examples)

    def _find_image(self, anon_id: str, eye: str) -> Image.Image | None:
        """Find and load the retinal image for this example."""
        import glob

        original_id = self.id_mapping.get(anon_id, anon_id.replace("P", ""))
        eye_letter = "l" if eye == "left" else "r"

        # Search for DICOM file
        pattern = str(
            self.cache_dir
            / f"retinal_photography/cfp/icare_eidon/{original_id}/*_{eye_letter}_*.dcm"
        )
        matches = glob.glob(pattern)

        if not matches:
            return None

        return self.dicom_loader.load(matches[0])

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Load image
        image = self._find_image(example["person_id"], example["eye"])
        if image is None:
            # Return a placeholder - will be filtered in collator
            return None

        # Create input message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": STAGE1_PROMPT},
                ],
            }
        ]

        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Tokenize target
        target_tokens = self.processor.tokenizer(
            example["target_findings"],
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Concatenate
        full_input_ids = torch.cat(
            [inputs["input_ids"], target_tokens["input_ids"]], dim=1
        )
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], target_tokens["attention_mask"]], dim=1
        )

        # Labels: -100 for input (no loss), actual ids for target
        labels = torch.cat(
            [
                torch.full_like(inputs["input_ids"], -100),
                target_tokens["input_ids"],
            ],
            dim=1,
        )

        # Truncate if needed
        if full_input_ids.shape[1] > self.max_length:
            full_input_ids = full_input_ids[:, : self.max_length]
            full_attention_mask = full_attention_mask[:, : self.max_length]
            labels = labels[:, : self.max_length]

        result = {
            "input_ids": full_input_ids.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)

        return result


def collate_fn(batch):
    """Filter None items and collate."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Pad sequences
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

        if "pixel_values" in item:
            pixel_values.append(item["pixel_values"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

    if pixel_values:
        result["pixel_values"] = torch.stack(pixel_values)

    return result


def main():
    print("=" * 60)
    print("STAGE 1 TRAINING: Visual Understanding")
    print("Task: Retinal Image → Findings Detection")
    print("=" * 60)

    # Get device configuration
    print("\nDetecting hardware...")
    device_config = get_device_config()

    # Paths
    cache_dir = Path("./data")
    training_dir = Path("./data/training/stage1_visual")
    output_dir = Path("./outputs/medgemma-stage1")

    # Load ID mapping
    print("\nLoading ID mapping...")
    with open("./data/training/id_mapping.json") as f:
        id_mapping = json.load(f)
    print(f"  Mapped {len(id_mapping)} participants")

    # Load model and processor
    print("\nLoading MedGemma...")
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load model with appropriate device configuration
    load_kwargs = {
        "torch_dtype": device_config["torch_dtype"],
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_config.get("device_map"):
        load_kwargs["device_map"] = device_config["device_map"]

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    # Move to device only if not using device_map
    if not device_config.get("device_map") and device_config.get("device"):
        print(f"  Moving model to {device_config['device']}...")
        model = model.to(device_config["device"])

    # For MPS, synchronize to ensure loading is complete
    if device_config.get("device") == "mps":
        torch.mps.synchronize()
        print(f"  Model loaded to MPS, using ~{torch.mps.current_allocated_memory() / 1e9:.1f}GB")
    else:
        print("  Model loaded successfully")

    # Configure LoRA
    print("\nConfiguring LoRA adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    # Enable gradients for the base model first
    model.train()
    for param in model.parameters():
        param.requires_grad = False  # Freeze all first

    model = get_peft_model(model, lora_config)

    # Ensure LoRA parameters have gradients
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.print_trainable_parameters()

    # torch.compile for 10-20% speedup on MPS
    if device_config.get("device") == "mps":
        print("  Compiling model for MPS (aot_eager backend)...")
        model = torch.compile(model, backend="aot_eager")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = Stage1Dataset(
        manifest_path=training_dir / "train_manifest.json",
        cache_dir=cache_dir,
        processor=processor,
        id_mapping=id_mapping,
    )
    val_dataset = Stage1Dataset(
        manifest_path=training_dir / "val_manifest.json",
        cache_dir=cache_dir,
        processor=processor,
        id_mapping=id_mapping,
    )
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")

    # Test loading one example
    print("\nTesting data loading...")
    test_sample = None
    try:
        sample = train_dataset[0]
        if sample is not None:
            print(f"  ✓ Sample loaded: input_ids shape = {sample['input_ids'].shape}")
            if "pixel_values" in sample:
                print(f"  ✓ Image loaded: pixel_values shape = {sample['pixel_values'].shape}")
            test_sample = sample
        else:
            print("  ⚠ First sample returned None, checking next...")
            for i in range(1, min(5, len(train_dataset))):
                sample = train_dataset[i]
                if sample is not None:
                    print(f"  ✓ Sample {i} loaded successfully")
                    test_sample = sample
                    break
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        return

    # Test forward pass
    if test_sample is not None:
        print("\nTesting forward pass (this may take a moment)...")
        try:
            with torch.no_grad():
                # Prepare batch
                batch = {
                    "input_ids": test_sample["input_ids"].unsqueeze(0),
                    "attention_mask": test_sample["attention_mask"].unsqueeze(0),
                    "labels": test_sample["labels"].unsqueeze(0),
                }
                if "pixel_values" in test_sample:
                    batch["pixel_values"] = test_sample["pixel_values"].unsqueeze(0)

                # Move to device
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                print(f"  ✓ Forward pass successful, loss = {outputs.loss.item():.4f}")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            print("\n  Tip: Try running with FORCE_CPU=1 if MPS is causing issues:")
            print("       FORCE_CPU=1 python scripts/run_stage1_training.py")
            return

    # Training arguments - optimized for M4 Mac
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Increased - 64GB has headroom
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # Halved to keep effective batch = 4
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=10,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=device_config["use_bf16"],
        fp16=device_config["use_fp16"],
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_prefetch_factor=2,  # Prefetch batches
    )

    # Trainer with MPS memory management
    callbacks = []
    if device_config.get("device") == "mps":
        callbacks.append(MPSMemoryCallback(sync_every_n_steps=4))
        print("  Added MPS memory management callback")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Stage 1 Training...")
    print("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current checkpoint...")

    # Save
    print("\nSaving Stage 1 adapter...")
    adapter_path = output_dir / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
