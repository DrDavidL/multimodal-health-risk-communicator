---
library_name: peft
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
base_model: google/medgemma-4b-it
tags:
  - medgemma
  - lora
  - health
  - diabetes
  - diabetic-retinopathy
  - patient-communication
  - health-literacy
  - medgemma-impact-challenge
datasets:
  - aireadi
pipeline_tag: text-generation
---

# MedGemma Stage 2 — Probabilistic Health Report Generator

A LoRA adapter fine-tuned on `google/medgemma-4b-it` for generating **patient-friendly health reports** that communicate diabetic retinopathy screening results using probabilistic framing and health-literate language.

## Model Description

This adapter was trained as **Model 2** in a dual-adapter pipeline for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge):

1. **Model 1 (DR Detection):** Community LoRA → retinal image → P(DR) grade probabilities
2. **Model 2 (This adapter):** P(DR) + clinical labs + CGM data → patient-friendly report

### What It Does

Given structured clinical context (DR probability, labs, CGM metrics, demographics), this adapter generates a ~5,000-character patient report that:

- Communicates risk using **natural frequencies** ("about 8 out of 10 people...")
- Clearly distinguishes **screening vs diagnosis**
- Provides **urgency-appropriate recommendations** (routine / moderate / urgent)
- Connects eye health to glucose control using the patient's own data
- Uses **8th-grade reading level** language

### Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `google/medgemma-4b-it` |
| PEFT method | LoRA |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 0.76% of base model |
| Training data | 43 participants from AI-READI dataset |
| Training targets | GPT-5.2 generated reports with probabilistic framing |
| Hardware | Apple Silicon M4 (MPS backend), float32 |
| Epochs | ~55 steps with gradient accumulation=4 |

### Evaluation

Evaluated on 7 held-out test participants using GPT-5.2 as automated judge (1-5 scale):

| Criterion | Fine-tuned (this) | Base MedGemma | Improvement |
|-----------|-------------------|---------------|-------------|
| Probability communication | **4.3** | 3.3 | +1.0 |
| Actionability | **3.7** | 2.6 | +1.1 |
| Clinical accuracy | 3.0 | 3.0 | 0 |
| Readability | 4.1 | 4.1 | 0 |
| Completeness | **5.0** | 4.9 | +0.1 |
| Overall quality | 3.4 | 3.1 | +0.3 |
| **Overall mean** | **3.9/5** | **3.5/5** | **+0.4** |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
model = PeftModel.from_pretrained(base_model, "drdavidl/medgemma-stage2-report")
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")

prompt = """You are a health communication specialist...
SCREENING RESULTS:
- Probability of diabetic retinopathy: 42.0% (about 4 out of 10)
...
"""

messages = [{"role": "user", "content": prompt}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.3)
report = processor.decode(outputs[0], skip_special_tokens=True)
```

## Privacy & Compliance

- Trained under AI-READI Data Use Agreement (DUA Section 3.D)
- LoRA adapters contain <1% of base model parameters — minimizes memorization risk
- All participant IDs anonymized (P1000–P1054)
- Numeric values jittered by 2% during training
- No patient data is stored in adapter weights

## Part Of

**Multimodal Health Risk Communicator** — [GitHub Repository](https://github.com/drdavidl/multimodal-health-risk-communicator)

MedGemma Impact Challenge submission by David Liebovitz, MD

## License

This adapter follows the [Health AI Developer Foundations Terms of Service](https://developers.google.com/health-ai-developer-foundations/terms), consistent with the base MedGemma model license.
