"""Fine-tuning pipeline for MedGemma on AI-READI data.

This module provides tools for:
- Creating training datasets from multimodal participant data
- Fine-tuning MedGemma with LoRA adapters
- Evaluating and validating fine-tuned models
- Extracting retinal findings metadata

Per AI-READI DUA Section 3.D, Licensee Models are permitted provided
reasonable efforts are made to prevent data memorization/reconstruction.

## 3-Stage Training Pipeline

Stage 1: Visual Understanding
  - Input: Retinal image
  - Output: Findings (DR, AMD, RVO)
  - Ground truth: condition_occurrence.csv

Stage 2: Report Generation (Text-Only)
  - Input: Findings + Clinical + CGM
  - Output: Patient-friendly report
  - Ground truth: GPT-5.2 generated

Stage 3: End-to-End Evaluation
  - Input: Image + Clinical + CGM (no findings)
  - Compare: MedGemma vs GPT-5.2
"""

from .dataset import (
    MultimodalTrainingDataset,
    TrainingExample,
    create_training_examples,
    load_training_examples_from_manifest,
)
from .trainer import MedGemmaFineTuner
from .retinal_findings import (
    load_retinal_findings,
    format_retinal_findings,
    format_retinal_findings_for_target,
    get_retinal_findings_summary,
)

__all__ = [
    # Dataset
    "MultimodalTrainingDataset",
    "TrainingExample",
    "create_training_examples",
    "load_training_examples_from_manifest",
    # Trainer
    "MedGemmaFineTuner",
    # Retinal findings
    "load_retinal_findings",
    "format_retinal_findings",
    "format_retinal_findings_for_target",
    "get_retinal_findings_summary",
]
