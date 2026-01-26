# Architecture

## System Overview

The Multimodal Health Risk Communicator uses a **dual-adapter architecture** on a single MedGemma 4B base model. Two LoRA adapters are swapped at inference time using PEFT's `model.set_adapter()` to perform distinct tasks: (1) diabetic retinopathy detection from fundus images, and (2) patient-friendly probabilistic report generation from structured clinical inputs.

```
                           ┌──────────────────────────────────────┐
                           │          DATA LAYER (Azure)           │
                           ├──────────┬───────────┬───────────────┤
                           │ Retinal  │   CGM     │   Clinical    │
                           │ DICOM    │   JSON    │   CSV         │
                           │ (YBR422) │ (mHealth) │  (OMOP CDM)   │
                           └────┬─────┴─────┬─────┴───────┬───────┘
                                │           │             │
                        lazy-download-and-cache pattern (Azure CLI)
                                │           │             │
                                ▼           ▼             ▼
                           ┌──────────────────────────────────────┐
                           │          LOADER LAYER                 │
                           ├──────────┬───────────┬───────────────┤
                           │ DICOM    │   CGM     │  Clinical     │
                           │ Loader   │  Loader   │  Loader       │
                           │ YBR→RGB  │  metrics  │  OMOP parse   │
                           └────┬─────┴─────┬─────┴───────┬───────┘
                                │           │             │
                                ▼           ▼             ▼
                           ┌──────────────────────────────────────┐
                           │       ParticipantLoader               │
                           │   (unified multimodal orchestrator)   │
                           └───────────────┬──────────────────────┘
                                           │
             ┌─────────────────────────────┼────────────────────────────┐
             │                             │                            │
             ▼                             ▼                            ▼
   ┌─────────────────┐       ┌────────────────────────┐    ┌──────────────────┐
   │  Fundus Image   │       │  Clinical Context      │    │  CGM Context     │
   │  (PIL Image)    │       │  (diabetes type,       │    │  (avg glucose,   │
   │                 │       │   HbA1c, duration)     │    │   TIR, GMI)      │
   └────────┬────────┘       └───────────┬────────────┘    └────────┬─────────┘
            │                            │                          │
            ▼                            │                          │
  ┌─────────────────────┐                │                          │
  │  MedGemma 4B Base   │◄───── loaded once ──────────────────────────────────┐
  │  (AutoModelFor      │                │                          │         │
  │   ImageTextToText)  │                │                          │         │
  ├─────────────────────┤                │                          │         │
  │                     │                │                          │         │
  │  ┌───────────────┐  │                │                          │         │
  │  │  DR Detection  │  │  set_adapter("dr")                      │         │
  │  │  LoRA Adapter  │──┼──►  fundus image  ──►  grade probs     │         │
  │  │  (Stage 1)     │  │     (A/B/C/D/E)       P(DR)            │         │
  │  └───────────────┘  │                  │                       │         │
  │                     │                  │                       │         │
  │  ┌───────────────┐  │                  ▼                       │         │
  │  │  Report Gen    │  │  set_adapter("report")                  │         │
  │  │  LoRA Adapter  │──┼──►  P(DR) + clinical + CGM  ──►  report│         │
  │  │  (Stage 2)     │  │     text-only input             text    │         │
  │  └───────────────┘  │                                          │         │
  │                     │                                          │         │
  └─────────────────────┘                                          │         │
            │                                                      │         │
            ▼                                                      │         │
  ┌─────────────────────┐                                          │         │
  │   OUTPUT             │                                          │         │
  │   Patient-friendly   │                                          │         │
  │   probabilistic      │◄─────── lifestyle Q&A uses same ────────┘         │
  │   health report      │         Stage 2 adapter with                      │
  │   + lifestyle Q&A    │         grounded prompting (ADA/NEI/AAO)          │
  └─────────────────────┘                                                    │
                                                                             │
                                             ~8GB base + ~60MB adapters ─────┘
```

---

## Component Details

### 1. Data Layer

Data is stored in Azure Blob Storage under a DUA-protected container. Three modalities per participant, linked by `person_id`:

| Modality | Format | Schema | Location |
|----------|--------|--------|----------|
| Retinal images | DICOM (`.dcm`) | YBR_FULL_422 color space | `retinal_photography/cfp/icare_eidon/{person_id}/` |
| CGM | JSON (`.json`) | Open mHealth | `wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{person_id}/` |
| Clinical | CSV | OMOP CDM | `clinical_data/measurement.csv`, `condition_occurrence.csv`, etc. |

Data availability: 2,244 of 2,280 participants have complete data across all three modalities.

### 2. Loader Layer

Each loader handles one modality and produces a standardized output. The `ParticipantLoader` orchestrates all three with a lazy-download-and-cache pattern via Azure CLI.

#### DICOMLoader (`src/loaders/dicom_loader.py`)

Loads retinal fundus DICOM images with automatic color space conversion:

- Reads DICOM files using `pydicom`
- Converts YBR_FULL_422 to RGB using `convert_color_space()`
- Returns PIL Image for direct model input

#### CGMLoader (`src/loaders/cgm_loader.py`)

Parses Open mHealth JSON and computes glycemic metrics:

- Extracts readings from `body.cgm[].blood_glucose.value` and `effective_time_frame.time_interval.start_date_time`
- Computes: mean glucose, time in range (70-180 mg/dL), time below/above range, GMI (estimated A1c), coefficient of variation
- Returns `CGMMetrics` dataclass with `.to_summary()` for prompt construction

#### ClinicalLoader (`src/loaders/clinical_loader.py`)

Parses OMOP CDM CSV files for clinical measurements:

- Reads `measurement.csv` filtered by `person_id`
- Extracts key variables: HbA1c, fasting glucose, BMI, blood pressure, lipid panel, creatinine, visual acuity, MoCA scores
- Returns `ClinicalData` dataclass with `.to_summary()` for prompt construction

#### AzureBlobDownloader (`src/loaders/azure_storage.py`)

Manages Azure Blob Storage access:

- Clinical CSVs downloaded once (~150MB, shared across all participants)
- CGM and retinal data downloaded on-demand per participant, cached locally
- Uses Azure CLI (`az storage blob download`) with key-based authentication
- Minimizes egress costs: 50 participants ~1.1GB ~$0.09

#### ParticipantLoader (`src/loaders/participant.py`)

Unified orchestrator:

- Coordinates all three sub-loaders
- Returns `ParticipantData` with `has_retinal()`, `has_cgm()`, `has_clinical()`, `is_complete()`
- Supports graceful degradation when modalities are missing
- Provides `to_prompt_context()` for combined text prompt generation

### 3. Model Layer

#### Base Model

`google/medgemma-4b-it` (MedGemma 4B Instruct) loaded via `AutoModelForImageTextToText` from Hugging Face Transformers. Supports both vision+text and text-only input through the same architecture.

#### DR Detection Adapter (`src/models/dr_detector.py`)

Community LoRA from `qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy`:

- **Input**: Retinal fundus image + standardized DR grading prompt
- **Output**: Token logits for grades A through E
- **Processing**: Extracts softmax probabilities for each grade token, normalizes across grades, computes P(DR) = P(B) + P(C) + P(D) + P(E)
- **Detection**: Threshold-based binary classification with configurable sensitivity presets

**Sensitivity Presets** (`SensitivityPreset` enum):

| Preset | Threshold | Use Case |
|--------|-----------|----------|
| SCREENING | 0.03 | Population screening, minimize false negatives |
| HIGH | 0.05 | Primary care (recommended, 100% sensitivity, 67% specificity) |
| BALANCED | 0.15 | Specialist settings with follow-up capacity |
| SPECIFIC | 0.30 | Research or when false positives are costly |

**Urgency Levels** (`UrgencyLevel` enum):

| Level | P(DR) Range | Recommendation |
|-------|-------------|----------------|
| URGENT | >= 0.7 | See specialist within 2 weeks |
| MODERATE | 0.3 - 0.7 | Schedule exam within 1-2 months |
| ROUTINE | < 0.3 | Continue annual exams |

#### Report Generation Adapter

Our novel LoRA fine-tuned on AI-READI data:

- **Input**: P(DR) probability + clinical context + CGM context (text-only, no images)
- **Output**: Patient-friendly probabilistic health report
- **Key behavior**: Communicates uncertainty using natural frequencies ("7 out of 10 people"), connects eye health to glucose control, provides urgency-appropriate recommendations
- **Also used for**: Lifestyle Q&A via grounded prompting with ADA/NEI/AAO clinical guidelines

### 4. Report Generator (`src/pipeline/report_generator.py`)

Orchestrates the full pipeline from data loading through report generation:

- Loads participant data via `ParticipantLoader`
- Selects appropriate prompt template based on available modalities (full multimodal, retinal-only, or text-only)
- Runs MedGemma inference with structured prompts
- Outputs `HealthReport` with markdown formatting, modality tracking, and warnings
- Supports batch processing with automatic file output

---

## Inference Pipeline

### Step 1: DR Detection (Stage 1 Adapter)

```python
# Load base model once
base_model = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it")
model = PeftModel.from_pretrained(base_model, "qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy")

# Process fundus image
messages = [{"role": "user", "content": [
    {"type": "image", "image": fundus_pil_image},
    {"type": "text", "text": DR_GRADES_PROMPT}
]}]
inputs = processor.apply_chat_template(messages, ...)
outputs = model(**inputs)
logits = outputs.logits[:, -1, :]

# Extract grade probabilities from next-token logits
grade_probs = {}
for grade in ["A", "B", "C", "D", "E"]:
    token_id = processor.tokenizer.encode(grade, add_special_tokens=False)[0]
    grade_probs[grade] = F.softmax(logits, dim=-1)[0, token_id].item()

# Normalize and compute P(DR)
total = sum(grade_probs.values())
grade_probs = {k: v / total for k, v in grade_probs.items()}
p_dr = grade_probs["B"] + grade_probs["C"] + grade_probs["D"] + grade_probs["E"]
```

### Step 2: Build Context

```python
# Clinical context from OMOP CDM
clinical_context = data.clinical.to_summary()
# Includes: diabetes type, HbA1c, years since diagnosis, BMI, BP, etc.

# CGM context from Open mHealth JSON
cgm_context = data.cgm_metrics.to_summary()
# Includes: avg glucose, time in range, GMI, coefficient of variation
```

### Step 3: Report Generation (Stage 2 Adapter)

```python
# Switch adapter on the same base model
model.set_adapter("report")

# Text-only input: P(DR) + clinical + CGM
prompt = STAGE2_PROMPT_TEMPLATE.format(
    p_dr=p_dr,
    certainty="likely" if p_dr >= 0.7 else "possible" if p_dr >= 0.3 else "unlikely",
    grade_description=grade_descriptions[predicted_grade],
    urgency=urgency_level,
    clinical_context=clinical_context,
    cgm_context=cgm_context,
)

# Generate patient-friendly report
report = model.generate(prompt, max_new_tokens=2000)
```

### Step 4: Lifestyle Q&A (Same Stage 2 Adapter)

Uses the same report generation adapter with grounded prompting that references ADA, NEI, and AAO clinical guidelines. This ensures recommendations are evidence-based and appropriate for the patient's risk level.

---

## Training Pipeline

### Stage 1: DR Detection (Community LoRA)

Used the pre-trained community adapter `qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy` without additional fine-tuning. This adapter was trained on fundus image classification and achieves 100% sensitivity at threshold 0.05 on our test set.

Ground truth labels come from `condition_occurrence.csv` in the AI-READI clinical data, which records clinical diagnoses (not automated image analysis):
- `mhoccur_pdr`: Diabetic retinopathy
- `mhoccur_amd`: Age-related macular degeneration
- `mhoccur_rvo`: Retinal vascular occlusion

### Stage 2: Report Generation (Our Novel LoRA)

Fine-tuned on MedGemma 4B (`AutoModelForCausalLM`, text-only mode) to generate patient-friendly probabilistic health reports.

| Parameter | Value |
|-----------|-------|
| Base model | `google/medgemma-4b-it` |
| Model class | `AutoModelForCausalLM` |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Training examples | 43 |
| Validation examples | 5 |
| Epochs | 5 |
| Batch size | 1 (gradient accumulation 4, effective batch 4) |
| Learning rate | 1e-4 |
| Max grad norm | 1.0 |
| Optimizer | AdamW (weight decay 0.01) |
| Warmup steps | 20 |
| Platform | Apple Silicon MPS (with `torch.compile(backend="aot_eager")`) |

**Training data format**: Each example consists of a structured prompt containing P(DR), clinical context, and CGM context, paired with a target patient-friendly report generated by Azure GPT-5.2 (BAA-compliant).

**Post-training conversion**: `torch.compile()` adds `_orig_mod.` prefix to adapter weight keys. The `scripts/convert_adapter.py` script strips this prefix to restore compatibility with PEFT's `PeftModel.from_pretrained()` for dual-adapter loading on `AutoModelForImageTextToText`.

### Stage 3: End-to-End Evaluation

Compares the full pipeline against a GPT-5.2 baseline:

| System | Image Input | Findings Input | Text Input |
|--------|-------------|----------------|------------|
| MedGemma Pipeline | Fundus image | Inferred via Stage 1 (P(DR)) | Clinical + CGM |
| GPT-5.2 Baseline | None | Ground truth from clinical data | Clinical + CGM |

Evaluation metrics: findings detection accuracy (DR sensitivity/specificity), report quality scoring.

### Privacy Safeguards (DUA Section 3.D)

1. **LoRA adapters**: Only ~0.1% of parameters trained, reducing memorization risk
2. **Anonymization**: Person IDs replaced with P1000, P1001, etc.
3. **Value jittering**: 2% noise added to numeric values during training data preparation
4. **Memorization testing**: Post-training similarity check with threshold < 5%
5. **Participant split isolation**: Train/validation/test sets are disjoint by participant (no data leakage)

---

## Deployment

### Local Development (M4 Mac, 64GB RAM)

| Metric | Value |
|--------|-------|
| Model load | ~30s |
| DR detection per image | ~5-10s |
| Report generation | ~10-15s |
| Base model memory | ~8GB |
| Adapter memory | ~60MB total |
| Device | Apple Silicon MPS (float32) |

MPS-specific optimizations:
- `torch.float32` (MPS does not support bfloat16)
- `torch.mps.synchronize()` and `torch.mps.empty_cache()` for memory management
- Custom `MPSMemoryCallback` during training

### HuggingFace Spaces (ZeroGPU)

Target deployment for competition judges:

| Parameter | Value |
|-----------|-------|
| GPU | T4 16GB (via ZeroGPU) |
| Model dtype | bfloat16 (CUDA) |
| Total memory | ~8GB base + ~60MB adapters (fits T4) |
| Interface | Gradio |

The dual-adapter architecture is efficient for deployment because both adapters share the same base model weights, requiring only one model load into GPU memory.

### CUDA Configuration

```python
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else {"": "mps"}
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Missing fundus images | Skip DR detection, generate report from clinical + CGM only |
| Missing CGM data | Generate report from retinal + clinical only |
| Missing clinical data | Generate report with available data, add warning |
| DICOM color space mismatch | Auto-detect and convert via `convert_color_space()` |
| Azure download failure | Log error, continue with locally cached data |
| MPS memory pressure | Periodic `torch.mps.synchronize()` + `torch.mps.empty_cache()` |
| `_orig_mod.` key prefix | `scripts/convert_adapter.py` strips prefix for PEFT compatibility |

---

## Key Design Principles

1. **Probabilistic communication**: Reports use natural frequencies ("7 out of 10 people with similar results") rather than raw percentages. Research shows natural frequencies improve patient comprehension of risk.

2. **8th grade reading level**: All patient-facing output avoids medical jargon and uses simple, clear language.

3. **Screening, not diagnosis**: The system produces screening results with explicit uncertainty. Every report states this is not a definitive diagnosis and recommends professional follow-up.

4. **Warm, supportive tone**: Reports celebrate positive findings, frame concerns constructively, and avoid alarming language.

5. **Grounded recommendations**: Lifestyle Q&A responses are grounded in published clinical guidelines from the American Diabetes Association (ADA), National Eye Institute (NEI), and American Academy of Ophthalmology (AAO).

6. **DUA compliance**: Data never leaves Azure for analysis. Only LoRA adapter weights (not training data) are distributed. Privacy safeguards minimize memorization risk.
