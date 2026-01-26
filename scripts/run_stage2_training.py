#!/usr/bin/env python3
"""Stage 2 Training: Probabilistic Report Generation.

Fine-tunes MedGemma to generate patient-friendly reports that communicate
DR findings probabilistically using natural frequencies.

- Input: P(DR) probability + Clinical context + CGM context
- Output: Patient-friendly report explaining findings with uncertainty

This is a text-to-text task (no images) - we're training the model to:
1. Interpret DR probability values
2. Communicate uncertainty using natural frequencies ("7 out of 10 people")
3. Provide appropriate recommendations based on urgency level
4. Connect eye health to glucose control when CGM data is available

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
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
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
            torch.mps.synchronize()
            if state.global_step % (self.sync_every_n_steps * 4) == 0:
                torch.mps.empty_cache()


def get_device_config():
    """Get optimal device configuration for current hardware."""
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
        print("  Using Apple Silicon MPS")
        return {
            "device_map": {"": "mps"},
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


# Stage 2 prompt template - focuses on probabilistic communication
STAGE2_PROMPT_TEMPLATE = """You are a health communication specialist helping patients understand their diabetic retinopathy screening results.

SCREENING RESULTS:
- Probability of diabetic retinopathy: {p_dr:.1%}
- Screening assessment: {certainty}
- Predicted severity if present: {grade_description}
- Urgency level: {urgency}

CLINICAL INFORMATION:
{clinical_context}

GLUCOSE MONITORING:
{cgm_context}

Generate a patient-friendly report that:
1. Explains the probability using natural frequencies (e.g., "X out of 10 people with similar results...")
2. Clearly states this is a SCREENING result, not a definitive diagnosis
3. Provides appropriate recommendations based on urgency
4. Connects eye health to glucose control when relevant
5. Uses simple language (8th grade reading level)
6. Is warm and supportive, not alarming

Include these sections:
- Understanding Your Retinal Screening Results
- Connecting Your Eye Health to Your Diabetes
- What You Should Do Next
- Key Points to Remember
- Questions to Ask Your Eye Doctor"""


def build_prompt(example: dict) -> str:
    """Build the input prompt from example data."""
    p_dr = example["p_dr"]

    # Determine certainty language
    if p_dr >= 0.7:
        certainty = "Diabetic retinopathy is likely"
    elif p_dr >= 0.3:
        certainty = "Diabetic retinopathy is possible"
    else:
        certainty = "Diabetic retinopathy is unlikely"

    # Grade description
    grade_descriptions = {
        "A": "no apparent retinopathy",
        "B": "mild early-stage changes",
        "C": "moderate changes",
        "D": "more advanced changes",
        "E": "advanced proliferative changes",
    }
    grade_description = grade_descriptions.get(example["dr_grade"], "some changes")

    return STAGE2_PROMPT_TEMPLATE.format(
        p_dr=p_dr,
        certainty=certainty,
        grade_description=grade_description,
        urgency=example["urgency"].upper(),
        clinical_context=example["clinical_context"],
        cgm_context=example["cgm_context"],
    )


class Stage2Dataset(Dataset):
    """Dataset for Stage 2: (P(DR) + Clinical + CGM) → Probabilistic Report."""

    def __init__(
        self,
        manifest_path: Path,
        processor,
        max_length: int = 2048,
    ):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.examples = self.manifest["examples"]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Build input prompt
        prompt = build_prompt(example)
        target = example["target_report"]

        # Create chat messages
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Apply chat template for input
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize input
        input_tokens = self.processor.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Tokenize target
        target_tokens = self.processor.tokenizer(
            target + self.processor.tokenizer.eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Concatenate
        full_input_ids = torch.cat(
            [input_tokens["input_ids"], target_tokens["input_ids"]], dim=1
        )
        full_attention_mask = torch.cat(
            [input_tokens["attention_mask"], target_tokens["attention_mask"]], dim=1
        )

        # Labels: -100 for input (no loss), actual ids for target
        labels = torch.cat(
            [
                torch.full_like(input_tokens["input_ids"], -100),
                target_tokens["input_ids"],
            ],
            dim=1,
        )

        # Truncate if needed
        if full_input_ids.shape[1] > self.max_length:
            full_input_ids = full_input_ids[:, :self.max_length]
            full_attention_mask = full_attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]

        return {
            "input_ids": full_input_ids.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def collate_fn(batch):
    """Collate and pad batch."""
    if not batch:
        return None

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

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

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def main():
    print("=" * 60)
    print("STAGE 2 TRAINING: Probabilistic Report Generation")
    print("Task: (P(DR) + Clinical + CGM) → Patient-Friendly Report")
    print("=" * 60)

    # Get device configuration
    print("\nDetecting hardware...")
    device_config = get_device_config()

    # Paths
    training_dir = Path("./data/training/stage2_probabilistic")
    output_dir = Path("./outputs/medgemma-stage2-probabilistic")

    # Load model and processor
    print("\nLoading MedGemma (text-only mode for Stage 2)...")
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # For Stage 2, we use the language model only (no vision)
    # MedGemma-4b-it is based on Gemma-2, so we can use AutoModelForCausalLM
    load_kwargs = {
        "torch_dtype": device_config["torch_dtype"],
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_config.get("device_map"):
        load_kwargs["device_map"] = device_config["device_map"]

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Move to device if needed
    if not device_config.get("device_map") and device_config.get("device"):
        print(f"  Moving model to {device_config['device']}...")
        model = model.to(device_config["device"])

    if device_config.get("device") == "mps":
        torch.mps.synchronize()
        print(f"  Model loaded to MPS, using ~{torch.mps.current_allocated_memory() / 1e9:.1f}GB")
    else:
        print("  Model loaded successfully")

    # Configure LoRA
    print("\nConfiguring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,  # Higher rank for more expressive text generation
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.print_trainable_parameters()

    # torch.compile for speedup on MPS
    if device_config.get("device") == "mps":
        print("  Compiling model for MPS...")
        model = torch.compile(model, backend="aot_eager")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = Stage2Dataset(
        manifest_path=training_dir / "train_manifest.json",
        processor=processor,
    )
    val_dataset = Stage2Dataset(
        manifest_path=training_dir / "val_manifest.json",
        processor=processor,
    )
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")

    # Test loading one example
    print("\nTesting data loading...")
    try:
        sample = train_dataset[0]
        print(f"  ✓ Sample loaded: input_ids shape = {sample['input_ids'].shape}")

        # Decode a snippet to verify
        input_snippet = processor.tokenizer.decode(sample['input_ids'][:100], skip_special_tokens=True)
        print(f"  ✓ Input starts with: {input_snippet[:80]}...")
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        return

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            batch = {
                "input_ids": sample["input_ids"].unsqueeze(0),
                "attention_mask": sample["attention_mask"].unsqueeze(0),
                "labels": sample["labels"].unsqueeze(0),
            }

            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            print(f"  ✓ Forward pass successful, loss = {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,  # More epochs for small dataset
        per_device_train_batch_size=1,  # Small batch for long sequences
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch = 4
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=20,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="no",  # Disable eval to avoid memory issues on MPS
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        bf16=device_config["use_bf16"],
        fp16=device_config["use_fp16"],
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=False,  # Use final checkpoint
    )

    # Trainer
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
    print("Starting Stage 2 Training...")
    print("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current checkpoint...")

    # Save
    print("\nSaving Stage 2 adapter...")
    adapter_path = output_dir / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    # Save training info
    info = {
        "stage": 2,
        "task": "probabilistic_report_generation",
        "base_model": model_id,
        "description": "Fine-tuned to generate patient-friendly reports with probabilistic DR communication",
        "input_format": "P(DR) + Clinical context + CGM context",
        "output_format": "Patient-friendly report with natural frequency explanations",
        "num_train_examples": len(train_dataset),
        "num_val_examples": len(val_dataset),
    }
    with open(adapter_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
