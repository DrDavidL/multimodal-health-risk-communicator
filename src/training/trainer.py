"""MedGemma fine-tuning with LoRA adapters.

Implements efficient fine-tuning using LoRA (Low-Rank Adaptation)
suitable for running on M3 Mac with 64GB RAM.

Per AI-READI DUA Section 3.D:
- Models trained on the Data are permitted
- Must not contain the Data itself
- Must minimize memorization risk
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

from .dataset import MultimodalTrainingDataset, TrainingExample


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    # LoRA hyperparameters
    r: int = 16  # Rank of update matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # MLP
    ])

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1  # Small for memory efficiency
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False  # Use bf16 on M3
    bf16: bool = True

    # Memorization mitigation
    max_grad_norm: float = 1.0  # Gradient clipping
    label_smoothing: float = 0.1  # Reduces overconfident memorization

    # Output
    output_dir: str = "./outputs/medgemma-lora"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10


class MedGemmaFineTuner:
    """Fine-tune MedGemma with LoRA adapters.

    Example:
        finetuner = MedGemmaFineTuner()
        finetuner.prepare_model()
        finetuner.train(train_examples, val_examples)
        finetuner.save_adapter("./adapters/patient-friendly-v1")
    """

    DEFAULT_MODEL_ID = "google/medgemma-4b-it"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize fine-tuner.

        Args:
            model_id: HuggingFace model identifier.
            lora_config: LoRA configuration (uses defaults if None).
            device: Device to train on (auto-detected if None).
        """
        self.model_id = model_id
        self.config = lora_config or LoRAConfig()
        self.device = device or self._detect_device()

        self.model = None
        self.processor = None
        self.peft_model = None

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def prepare_model(self) -> None:
        """Load base model and apply LoRA adapters."""
        print(f"Loading base model: {self.model_id}")
        print(f"Device: {self.device}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load base model
        model_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=model_dtype,
            device_map=self.device,
        )

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

        # Apply LoRA adapters
        self.peft_model = get_peft_model(self.model, peft_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def train(
        self,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample],
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Fine-tune the model on training data.

        Args:
            train_examples: Training examples.
            val_examples: Validation examples.
            resume_from_checkpoint: Path to checkpoint to resume from.
        """
        if self.peft_model is None:
            raise RuntimeError("Call prepare_model() before training")

        # Create datasets
        train_dataset = MultimodalTrainingDataset(
            train_examples, self.processor
        )
        val_dataset = MultimodalTrainingDataset(
            val_examples, self.processor
        )

        print(f"Training examples: {len(train_dataset)}")
        print(f"Validation examples: {len(val_dataset)}")

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            label_smoothing_factor=self.config.label_smoothing,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Disable wandb etc. by default
            dataloader_pin_memory=False,  # For MPS compatibility
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        print("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print("Training complete!")

    def save_adapter(self, output_path: str | Path) -> None:
        """Save LoRA adapter weights.

        Only saves the adapter weights, not the full model.
        This is what will be distributed (per DUA Section 3.D).

        Args:
            output_path: Directory to save adapter.
        """
        if self.peft_model is None:
            raise RuntimeError("No model to save")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        self.peft_model.save_pretrained(output_path)

        # Save config
        config_dict = {
            "base_model": self.model_id,
            "lora_r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
        }
        with open(output_path / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Adapter saved to: {output_path}")

    def load_adapter(self, adapter_path: str | Path) -> None:
        """Load a previously saved adapter.

        Args:
            adapter_path: Path to saved adapter directory.
        """
        from peft import PeftModel

        if self.model is None:
            # Load base model first
            model_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=model_dtype,
                device_map=self.device,
            )

        # Load adapter
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"Adapter loaded from: {adapter_path}")

    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate text using the fine-tuned model.

        Args:
            image: PIL Image.
            prompt: Text prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        model = self.peft_model or self.model
        if model is None:
            raise RuntimeError("No model loaded")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()

        return response


def run_memorization_check(
    model: MedGemmaFineTuner,
    test_examples: list[TrainingExample],
    similarity_threshold: float = 0.9,
) -> dict:
    """Check for potential data memorization.

    Tests whether the model reproduces training data verbatim.

    Args:
        model: Fine-tuned model.
        test_examples: Examples to test (should be from training set).
        similarity_threshold: Threshold for considering output as memorized.

    Returns:
        Dict with memorization statistics.
    """
    from difflib import SequenceMatcher

    results = {
        "total_tested": len(test_examples),
        "potential_memorization": 0,
        "max_similarity": 0.0,
        "flagged_examples": [],
    }

    for example in test_examples:
        # Generate response
        prompt = f"""Based on the following health information, provide a patient-friendly explanation:

{example.clinical_context}

{example.cgm_context}"""

        generated = model.generate(example.image, prompt)

        # Check similarity to target
        similarity = SequenceMatcher(
            None, generated.lower(), example.target_response.lower()
        ).ratio()

        results["max_similarity"] = max(results["max_similarity"], similarity)

        if similarity > similarity_threshold:
            results["potential_memorization"] += 1
            results["flagged_examples"].append({
                "example_id": example.example_id,
                "similarity": similarity,
            })

    results["memorization_rate"] = (
        results["potential_memorization"] / results["total_tested"]
        if results["total_tested"] > 0 else 0
    )

    return results
