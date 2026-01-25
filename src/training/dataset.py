"""Training dataset creation for MedGemma fine-tuning.

Creates multimodal training examples from AI-READI participant data.
Each example consists of:
- Input: Retinal image + clinical context + CGM summary
- Output: Patient-friendly health explanation

Includes data augmentation and privacy-preserving techniques.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from ..loaders import ParticipantData, ParticipantLoader


@dataclass
class TrainingExample:
    """Single training example for fine-tuning."""

    # Unique identifier (for tracking, not included in training)
    example_id: str

    # Inputs
    image: Image.Image
    clinical_context: str
    cgm_context: str

    # Target output
    target_response: str

    # Metadata
    person_id: str
    eye: str  # "left" or "right"

    # Optional fields (with defaults)
    retinal_context: str = ""  # Retinal findings (DR, AMD, RVO)


@dataclass
class TrainingConfig:
    """Configuration for training dataset creation."""

    # Data selection
    min_cgm_days: float = 7.0  # Minimum CGM recording days
    require_complete: bool = True  # Require all 3 modalities

    # Privacy/memorization mitigation
    anonymize_ids: bool = True  # Replace person_id with random ID
    jitter_values: bool = True  # Add small noise to numeric values
    jitter_amount: float = 0.02  # 2% jitter

    # Augmentation
    use_both_eyes: bool = True  # Create examples for both eyes
    augment_prompts: bool = True  # Vary prompt wording

    # Output
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


# Prompt variations for augmentation
PROMPT_TEMPLATES = [
    """You are a compassionate health educator. Based on the retinal image and health data below,
provide a patient-friendly explanation of the findings.

{retinal_context}

{clinical_context}

{cgm_context}

Explain what you see in the eye image and how it relates to the patient's overall health.""",

    """As a medical educator, help this patient understand their health status.
Review the eye image along with their clinical information:

{retinal_context}

{clinical_context}

{cgm_context}

Provide a clear, simple explanation suitable for someone without medical training.""",

    """Review this patient's multimodal health data and provide an accessible summary.

Retinal Eye Exam:
{retinal_context}

Clinical Information:
{clinical_context}

Glucose Monitoring:
{cgm_context}

Looking at the retinal image, explain the findings and their health implications in plain language.""",
]

# Response structure guidance (appended to prompts)
RESPONSE_GUIDANCE = """

Structure your response with:
1. What the eye exam shows (simple terms)
2. How glucose patterns relate to overall health
3. Key takeaways (2-3 bullet points)
4. Questions to discuss with healthcare provider"""


def jitter_numeric_value(value: float, amount: float = 0.02) -> float:
    """Add small random noise to a numeric value.

    This helps prevent exact memorization of patient data.

    Args:
        value: Original numeric value.
        amount: Maximum jitter as fraction of value (default 2%).

    Returns:
        Jittered value.
    """
    if value == 0:
        return value
    jitter = random.uniform(-amount, amount) * abs(value)
    return round(value + jitter, 2)


def anonymize_summary(summary: str, original_id: str, anon_id: str) -> str:
    """Replace identifiable information in summary text.

    Args:
        summary: Original summary text.
        original_id: Original person_id to replace.
        anon_id: Anonymized ID to use.

    Returns:
        Anonymized summary text.
    """
    # Replace person_id references
    result = summary.replace(f"Participant {original_id}", f"Participant {anon_id}")
    result = result.replace(f"participant {original_id}", f"participant {anon_id}")
    result = result.replace(original_id, anon_id)

    return result


def create_training_examples(
    loader: ParticipantLoader,
    person_ids: list[str],
    config: TrainingConfig,
    response_generator: Optional[callable] = None,
) -> list[TrainingExample]:
    """Create training examples from participant data.

    Args:
        loader: ParticipantLoader instance.
        person_ids: List of participant IDs to process.
        config: Training configuration.
        response_generator: Optional function to generate target responses.
                          If None, examples will have empty targets (for later annotation).

    Returns:
        List of TrainingExample objects.
    """
    random.seed(config.random_seed)
    examples = []
    anon_counter = 1000

    for person_id in person_ids:
        try:
            data = loader.load(person_id)

            # Skip incomplete data if required
            if config.require_complete and not data.is_complete():
                continue

            # Skip if CGM recording too short
            if data.cgm_metrics and data.cgm_metrics.duration_days < config.min_cgm_days:
                continue

            # Generate anonymized ID
            anon_id = f"P{anon_counter}" if config.anonymize_ids else person_id
            anon_counter += 1

            # Get context strings
            clinical_context = ""
            if data.clinical:
                clinical_context = data.clinical.to_summary()
                if config.anonymize_ids:
                    clinical_context = anonymize_summary(clinical_context, person_id, anon_id)

            cgm_context = ""
            if data.cgm_metrics:
                cgm_context = data.cgm_metrics.to_summary()

            # Apply value jittering if enabled
            if config.jitter_values and data.clinical:
                # Note: In production, we'd jitter individual values before summary generation
                # For now, this is a placeholder for the concept
                pass

            # Create examples for each available eye
            eyes_to_process = []
            if data.fundus_left:
                eyes_to_process.append(("left", data.fundus_left))
            if data.fundus_right and config.use_both_eyes:
                eyes_to_process.append(("right", data.fundus_right))
            elif data.fundus_right and not data.fundus_left:
                eyes_to_process.append(("right", data.fundus_right))

            for eye, image in eyes_to_process:
                # Select prompt template
                if config.augment_prompts:
                    prompt_template = random.choice(PROMPT_TEMPLATES)
                else:
                    prompt_template = PROMPT_TEMPLATES[0]

                prompt = prompt_template.format(
                    clinical_context=clinical_context,
                    cgm_context=cgm_context,
                ) + RESPONSE_GUIDANCE

                # Generate target response if generator provided
                target = ""
                if response_generator:
                    target = response_generator(image, clinical_context, cgm_context)

                example = TrainingExample(
                    example_id=f"{anon_id}_{eye}",
                    image=image,
                    clinical_context=clinical_context,
                    cgm_context=cgm_context,
                    target_response=target,
                    person_id=anon_id,
                    eye=eye,
                )
                examples.append(example)

        except Exception as e:
            print(f"Warning: Could not process {person_id}: {e}")
            continue

    return examples


class MultimodalTrainingDataset(Dataset):
    """PyTorch Dataset for MedGemma fine-tuning.

    Wraps TrainingExample objects for use with PyTorch DataLoader.
    """

    def __init__(
        self,
        examples: list[TrainingExample],
        processor,  # AutoProcessor from transformers
        max_length: int = 2048,
    ):
        """Initialize dataset.

        Args:
            examples: List of TrainingExample objects.
            processor: HuggingFace processor for MedGemma.
            max_length: Maximum sequence length.
        """
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training example.

        Returns dict with:
            - input_ids: Tokenized input
            - attention_mask: Attention mask
            - labels: Target token IDs
            - pixel_values: Processed image
        """
        example = self.examples[idx]

        # Construct message in MedGemma chat format
        prompt = PROMPT_TEMPLATES[0].format(
            retinal_context=example.retinal_context or "No documented retinal findings.",
            clinical_context=example.clinical_context,
            cgm_context=example.cgm_context,
        ) + RESPONSE_GUIDANCE

        # First, process the user message with image to get input
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example.image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Get the input portion with generation prompt
        inputs = self.processor.apply_chat_template(
            user_message,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Tokenize the target response
        target_tokens = self.processor.tokenizer(
            example.target_response,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Concatenate input_ids with target tokens
        import torch
        full_input_ids = torch.cat([
            inputs["input_ids"],
            target_tokens["input_ids"],
        ], dim=1)

        full_attention_mask = torch.cat([
            inputs["attention_mask"],
            target_tokens["attention_mask"],
        ], dim=1)

        # Create labels: -100 for input tokens (don't compute loss), actual ids for target
        labels = torch.cat([
            torch.full_like(inputs["input_ids"], -100),
            target_tokens["input_ids"],
        ], dim=1)

        # Truncate if too long
        if full_input_ids.shape[1] > self.max_length:
            full_input_ids = full_input_ids[:, :self.max_length]
            full_attention_mask = full_attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]

        result = {
            "input_ids": full_input_ids.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

        # Add pixel values if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)

        return result


def split_examples(
    examples: list[TrainingExample],
    config: TrainingConfig,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Split examples into train/val/test sets.

    Ensures no participant appears in multiple splits.

    Args:
        examples: Full list of examples.
        config: Training configuration with split ratios.

    Returns:
        Tuple of (train_examples, val_examples, test_examples).
    """
    random.seed(config.random_seed)

    # Group by participant to prevent data leakage
    by_participant: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        if ex.person_id not in by_participant:
            by_participant[ex.person_id] = []
        by_participant[ex.person_id].append(ex)

    # Shuffle participants
    participants = list(by_participant.keys())
    random.shuffle(participants)

    # Calculate split indices
    n = len(participants)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)

    train_participants = participants[:train_end]
    val_participants = participants[train_end:val_end]
    test_participants = participants[val_end:]

    # Collect examples for each split
    train_examples = [ex for p in train_participants for ex in by_participant[p]]
    val_examples = [ex for p in val_participants for ex in by_participant[p]]
    test_examples = [ex for p in test_participants for ex in by_participant[p]]

    return train_examples, val_examples, test_examples


def load_training_examples_from_manifest(
    manifest_path: Path,
    cache_dir: Path,
    person_id_mapping: dict[str, str] | None = None,
) -> list[TrainingExample]:
    """Load training examples from a saved manifest.

    Args:
        manifest_path: Path to manifest JSON file.
        cache_dir: Base directory for cached data.
        person_id_mapping: Optional mapping from anonymized IDs back to original IDs
                          for loading images. If None, assumes mapping file exists.

    Returns:
        List of TrainingExample objects with images loaded.
    """
    import glob

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Try to load ID mapping if not provided
    if person_id_mapping is None:
        mapping_path = manifest_path.parent / "id_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                person_id_mapping = json.load(f)
        else:
            person_id_mapping = {}

    examples = []
    for ex_data in manifest["examples"]:
        # Skip if no target response (metadata-only manifest)
        if "target_response" not in ex_data:
            continue

        # Find the original person_id for loading images
        anon_id = ex_data["person_id"]
        original_id = person_id_mapping.get(anon_id, anon_id.replace("P", ""))
        eye = ex_data["eye"]

        # Find the DICOM file
        eye_letter = "l" if eye == "left" else "r"
        search_pattern = str(cache_dir / f"retinal_photography/cfp/icare_eidon/{original_id}/*_{eye_letter}_*.dcm")
        matches = glob.glob(search_pattern)

        if not matches:
            # Try alternate location
            search_pattern = str(cache_dir / f"participants/{original_id}/retinal/*_{eye}.dcm")
            matches = glob.glob(search_pattern)

        if not matches:
            print(f"Warning: No image found for {anon_id} {eye} (original: {original_id})")
            continue

        # Load the image
        from ..loaders.dicom_loader import DICOMLoader
        try:
            loader = DICOMLoader()
            image = loader.load(matches[0])
        except Exception as e:
            print(f"Warning: Could not load image for {anon_id}: {e}")
            continue

        example = TrainingExample(
            example_id=ex_data["example_id"],
            image=image,
            clinical_context=ex_data.get("clinical_context", ""),
            cgm_context=ex_data.get("cgm_context", ""),
            retinal_context=ex_data.get("retinal_context", ""),
            target_response=ex_data["target_response"],
            person_id=anon_id,
            eye=eye,
        )
        examples.append(example)

    return examples


def save_dataset_manifest(
    examples: list[TrainingExample],
    output_path: Path,
    save_full_content: bool = True,
) -> None:
    """Save dataset manifest for reproducibility.

    Args:
        examples: List of training examples.
        output_path: Path to save manifest JSON.
        save_full_content: If True, save full contexts and targets (for re-use).
    """
    manifest = {
        "num_examples": len(examples),
        "examples": [
            {
                "example_id": ex.example_id,
                "person_id": ex.person_id,
                "eye": ex.eye,
                "has_clinical": bool(ex.clinical_context),
                "has_cgm": bool(ex.cgm_context),
                "target_length": len(ex.target_response),
                "has_retinal": bool(ex.retinal_context),
                # Full content for training data re-use
                **(
                    {
                        "clinical_context": ex.clinical_context,
                        "cgm_context": ex.cgm_context,
                        "retinal_context": ex.retinal_context,
                        "target_response": ex.target_response,
                    }
                    if save_full_content
                    else {}
                ),
            }
            for ex in examples
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
