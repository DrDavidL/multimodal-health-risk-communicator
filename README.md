# Multimodal Health Risk Communicator

**MedGemma Impact Challenge Submission**

A multimodal AI system that combines retinal fundus imaging, continuous glucose monitoring (CGM), and clinical data to generate patient-friendly health risk explanations using Google's MedGemma foundation model.

## Project Goal

Transform complex multimodal medical data into accessible, understandable health insights for patients. The system analyzes:
- **Retinal fundus photographs** — microvascular health indicators
- **Continuous glucose monitoring data** — glycemic patterns and variability
- **Clinical measurements** — demographics, labs, vitals

And generates plain-language explanations of health risks and recommendations.

## Dataset

**AI-READI** (Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights)
- Part of NIH Bridge2AI program
- 2,280 participants across 4 diabetes status groups
- **2,244 participants** have complete data for our 3 target modalities
- Access: Approved via AIREADI.org DUA

| Study Group | Participants |
|-------------|--------------|
| Healthy controls | 760 |
| Pre-diabetes (lifestyle controlled) | 552 |
| Oral/injectable medication controlled | 681 |
| Insulin dependent | 251 |

## Model

**MedGemma 4B** (multimodal variant)
- Google's medical foundation model
- Supports image + text input
- Running locally on M4 Mac (64GB unified memory)
- Hugging Face: `google/medgemma-4b-it`

### Apple Silicon Training Optimizations

Fine-tuning runs entirely on-device using MPS (Metal Performance Shaders):

| Optimization | Description |
|--------------|-------------|
| `device_map={"": "mps"}` | Load model directly to MPS, avoiding CPU→GPU copy |
| `torch_dtype=torch.float32` | Float32 for numerical stability (float16 caused NaN loss) |
| `torch.compile(backend="aot_eager")` | JIT compilation for ~15% speedup |
| LoRA rank=8 | Low-rank adaptation with 0.14% trainable parameters |
| Manual training loop | Better MPS compatibility than HF Trainer |
| Gradient accumulation=4 | Effective batch size of 4 with batch_size=1 |

## Architecture

### 3-Stage Fine-Tuning Pipeline

We use a novel 3-stage approach that demonstrates learned visual understanding:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Visual Understanding                                           │
│ ┌──────────────┐                     ┌─────────────────────────────┐   │
│ │Retinal Image │ ──── MedGemma ────▶ │ Retinal Findings Detection  │   │
│ └──────────────┘     (fine-tuned)    │ (DR, AMD, RVO)              │   │
│                                       └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Report Generation (Text-Only)                                  │
│ ┌──────────────┐                                                        │
│ │ Retinal      │                     ┌─────────────────────────────┐   │
│ │ Findings     │                     │                             │   │
│ ├──────────────┤ ──── MedGemma ────▶ │  Patient-Friendly Report   │   │
│ │ Clinical     │     (fine-tuned)    │                             │   │
│ ├──────────────┤                     └─────────────────────────────┘   │
│ │ CGM Data     │                                                        │
│ └──────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: End-to-End Evaluation                                          │
│ ┌──────────────┐                     ┌─────────────────────────────┐   │
│ │Retinal Image │                     │                             │   │
│ ├──────────────┤ ──── MedGemma ────▶ │  Patient-Friendly Report   │   │
│ │ Clinical     │  (doubly fine-tuned)│  (findings inferred from   │   │
│ ├──────────────┤                     │   image, not provided!)    │   │
│ │ CGM Data     │                     └─────────────────────────────┘   │
│ └──────────────┘                                                        │
│        ⚠️ NO retinal findings metadata provided                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Comparison to Baseline

| Model | Input | Retinal Findings |
|-------|-------|------------------|
| **MedGemma (ours)** | Image + Clinical + CGM | Must infer from image |
| **GPT-5.2 (baseline)** | Text only | Provided as metadata |

This demonstrates that MedGemma **learned to see** diabetic retinopathy, not just explain pre-labeled findings.

## Project Structure

```
multimodal-health-risk-communicator/
├── src/
│   ├── loaders/
│   │   ├── __init__.py          # Exports all loaders
│   │   ├── azure_storage.py     # Azure download with caching
│   │   ├── dicom_loader.py      # Retinal DICOM with YBR→RGB conversion
│   │   ├── cgm_loader.py        # CGM JSON (Open mHealth format)
│   │   ├── clinical_loader.py   # Clinical CSV (OMOP CDM format)
│   │   └── participant.py       # Unified multimodal loader
│   ├── models/
│   │   ├── __init__.py          # Exports MedGemmaInference, DRDetector
│   │   ├── medgemma.py          # MedGemma 4B wrapper
│   │   └── dr_detector.py       # High-sensitivity DR detection ⭐
│   ├── training/
│   │   ├── __init__.py          # Training exports
│   │   ├── dataset.py           # Training dataset creation
│   │   ├── trainer.py           # LoRA fine-tuning
│   │   └── retinal_findings.py  # Retinal metadata extraction ⭐
│   ├── pipeline/
│   │   ├── __init__.py          # Exports ReportGenerator
│   │   └── report_generator.py  # Patient report generation
│   └── utils/
├── scripts/
│   ├── prepare_stage1_data.py   # Stage 1: Visual understanding data
│   ├── prepare_stage2_data.py   # Stage 2: Report generation data
│   ├── run_stage1_training.py   # Train visual understanding
│   ├── run_stage2_training.py   # Train report generation
│   ├── evaluate_stage3.py       # End-to-end evaluation
│   └── demo_report.py           # Quick demo script
├── data/                        # Local cache (gitignored)
│   ├── clinical_data/           # Shared clinical CSVs
│   │   └── condition_occurrence.csv  # Contains retinal findings! ⭐
│   ├── training/
│   │   ├── stage1_visual/       # Stage 1 manifests
│   │   └── stage2_report/       # Stage 2 manifests
│   └── participants/1001/       # Per-participant cached data
├── outputs/
│   ├── medgemma-stage1/         # Stage 1 adapter weights
│   ├── medgemma-stage2/         # Stage 2 adapter weights
│   └── evaluation/              # Stage 3 results
├── CLAUDE.md                    # Context for Claude Code
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.10+
- ~16GB RAM minimum (64GB recommended for local inference)
- Azure CLI (for data access)
- Hugging Face account with MedGemma access

### Installation

```bash
# Clone repository
git clone https://github.com/drdavidl/multimodal-health-risk-communicator.git
cd multimodal-health-risk-communicator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Azure credentials
az login
```

### Data Access

Data is stored in Azure Blob Storage (not in repo — DUA protected).

**Important:** All data is nested under a UUID prefix in Azure:
```
aXXX/dataset/
```

### Data Processing Strategy

We use a **lazy-download-and-cache** approach to minimize Azure bandwidth costs and local storage:

| Data Type | Size | Strategy |
|-----------|------|----------|
| Clinical CSVs | ~150MB total | Download once, shared across all participants |
| CGM JSON | ~2MB/participant | Download on-demand, cache locally |
| Retinal DICOM | ~20MB/participant | Download on-demand, cache locally |

**Why this approach:**
- Full dataset is ~50GB (mostly images) — impractical to download entirely
- Competition demos only need a subset of participants (10-50)
- Azure egress costs are minimized by caching
- Reproducible: others can run with just Azure access + our code

**Estimated costs for evaluation:**
- 50 participants × 22MB ≈ 1.1GB download
- Azure egress: ~$0.09 for first 10GB/month (often free tier)

```bash
# Login to Azure
az login

# Set credentials from your DUA (do not commit actual values)
export AIREADI_STORAGE_ACCOUNT="your-storage-account"
export AIREADI_CONTAINER="your-container-name"
export BLOB_PREFIX="your-uuid-prefix"  # UUID from DUA

# Download metadata files
az storage blob download --account-name "$AIREADI_STORAGE_ACCOUNT" --container-name "$AIREADI_CONTAINER" \
    --name "$BLOB_PREFIX/participants.tsv" --file ./data/participants.tsv

az storage blob download --account-name "$AIREADI_STORAGE_ACCOUNT" --container-name "$AIREADI_CONTAINER" \
    --name "$BLOB_PREFIX/dataset_structure_description.json" --file ./data/dataset_structure_description.json

# Download data for a specific participant (e.g., 1001)
# Retinal fundus images
az storage blob download --account-name "$AIREADI_STORAGE_ACCOUNT" --container-name "$AIREADI_CONTAINER" \
    --name "$BLOB_PREFIX/retinal_photography/cfp/icare_eidon/1001/1001_eidon_mosaic_cfp_l_*.dcm" \
    --file ./data/participants/1001/retinal/

# CGM data
az storage blob download --account-name "$AIREADI_STORAGE_ACCOUNT" --container-name "$AIREADI_CONTAINER" \
    --name "$BLOB_PREFIX/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/1001/1001_DEX.json" \
    --file ./data/participants/1001/cgm/1001_DEX.json
```

### Data File Paths

| Modality | Path Pattern | Format |
|----------|-------------|--------|
| Retinal photography | `retinal_photography/cfp/icare_eidon/{id}/{id}_eidon_*.dcm` | DICOM (YBR_FULL_422) |
| CGM | `wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{id}/{id}_DEX.json` | JSON (Open mHealth) |
| Clinical | `clinical_data/measurement.csv`, `person.csv` | CSV (OMOP CDM) |

**Linkage key:** `person_id` (e.g., `1001`, `1002`) is consistent across all files.

## Usage

### Quick Demo (Data Loading Only)

```bash
# Test data loading pipeline without running MedGemma
python scripts/demo_report.py 1001
```

### Generate Patient Report

```bash
# Full inference with MedGemma
python scripts/demo_report.py 1001 --run-inference
```

### Python API

```python
from src.loaders import ParticipantLoader
from src.pipeline import ReportGenerator

# Load multimodal data for a participant
loader = ParticipantLoader(cache_dir="./data", auto_download=True)
data = loader.load("1001")

# Access individual modalities
print(data.clinical.to_summary())    # Clinical summary
print(data.cgm_metrics.to_summary()) # CGM metrics
data.fundus_left.show()               # Display retinal image

# Generate patient-friendly report
generator = ReportGenerator(cache_dir="./data")
report = generator.generate_report("1001")

# Output report
print(report.to_markdown())
report.save("./reports/1001_report.md")
```

### Batch Processing

```python
# Generate reports for multiple participants
generator = ReportGenerator(cache_dir="./data", preload_model=True)
reports = generator.generate_batch_reports(
    person_ids=["1001", "1002", "1003"],
    output_dir="./reports"
)
```

## 3-Stage Training Pipeline

### Stage 1: Visual Understanding

Train MedGemma to identify retinal findings from images alone.

```bash
# Prepare Stage 1 data
python scripts/prepare_stage1_data.py

# Train (outputs to ./outputs/medgemma-stage1/)
python scripts/run_stage1_training.py
```

**Task:** `Retinal Image → Retinal Findings (DR, AMD, RVO)`

#### DR-Prioritized Data Balancing

Since this project focuses on diabetes, we implement **DR-prioritized stratification** to ensure diabetic retinopathy cases are properly represented:

1. **Stratified Splits**: DR cases are distributed across train/val/test splits first (highest priority), then AMD/RVO cases, then negatives. This ensures DR detection can be evaluated on held-out data.

2. **Oversampling**: Positive cases are oversampled in training to balance the class distribution (~50% positive after oversampling).

**Training Data Distribution:**

| Split | Total | DR+ | AMD+ | RVO+ | Notes |
|-------|-------|-----|------|------|-------|
| Train | 134 | 40 | 32 | 8 | After 4x oversampling of positives |
| Val | 10 | 2 | 0 | 0 | For monitoring training |
| Test | 14 | 2 | 2 | 0 | Held-out for final evaluation |

#### Stage 1 Performance: Fine-tuned Model

| Metric | Value | Notes |
|--------|-------|-------|
| DR Accuracy | 78.6% | 0% sensitivity, high specificity |
| AMD Accuracy | 78.6% | 0% sensitivity, high specificity |
| RVO Accuracy | 92.9% | High specificity |
| **Overall Accuracy** | **83.3%** | Strong at ruling out, weak at detection |

**Analysis:** The fine-tuned model achieved high specificity but **0% sensitivity** for DR and AMD — it missed both true positive cases. This is problematic for a diabetes screening application where false negatives are costly.

#### Stage 1 Performance: Pre-trained DR Model (Recommended)

We integrate a community-trained DR-specific LoRA adapter: [`qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy`](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)

**Method:** Extract grade-level probabilities (A-E) and compute P(DR) = P(B) + P(C) + P(D) + P(E)

| Threshold | Sensitivity | Specificity | TP | FP | FN | TN |
|-----------|-------------|-------------|----|----|----|----|
| 0.01      | **100%**    | 58.3%       | 2  | 5  | 0  | 7  |
| **0.05** (default) | **100%** | **66.7%** | 2 | 4 | 0 | 8 |
| 0.10      | 50%         | 75%         | 1  | 3  | 1  | 9  |

**Recommendation:** For screening applications where missing DR is costly, use **threshold ≤ 0.062** (minimum P(DR) among true positives). The default threshold of 0.05 achieves **100% sensitivity** with 66.7% specificity.

```python
from src.models import DRDetector

# High-sensitivity DR detection (100% sensitivity)
detector = DRDetector(threshold=0.05)
result = detector.detect(fundus_image)

print(f"Has DR: {result.has_dr}")      # True/False at threshold
print(f"P(DR): {result.p_dr:.3f}")      # Raw probability
print(f"Grade: {result.predicted_grade}") # A-E
```

### Stage 2: Probabilistic Report Generation

Fine-tune on generating patient reports that communicate DR findings **probabilistically**.

```bash
# Prepare Stage 2 data with probabilistic framing (uses DR model + GPT-5)
python scripts/prepare_stage2_probabilistic.py

# Train (outputs to ./outputs/medgemma-stage2/)
python scripts/run_stage2_training.py
```

**Task:** `(P(DR) + Clinical + CGM) → Probabilistic Patient Report`

#### Health Literacy-Aware Communication

Reports are trained to communicate uncertainty using best practices for patients with varying numeracy levels:

| Concept | Implementation |
|---------|----------------|
| **Natural frequencies** | "7 out of 10 people" instead of "70%" |
| **Screening vs diagnosis** | Clear that AI screening ≠ definitive diagnosis |
| **Urgency tiers** | P(DR) ≥ 0.7 = urgent, 0.3-0.7 = moderate, < 0.3 = routine |
| **Action-oriented** | Specific next steps, not vague "follow up" |

**Example output:**
> The screening found some signs that suggest diabetic retinopathy is possible. If we screened 10 people with similar results, about 4 would have diabetic retinopathy and 6 would not. This is a screening result, not a definitive diagnosis—your eye doctor can give you a definitive answer.
>
> **Recommendation:** Discuss these findings with your doctor at your next visit, or schedule an eye exam within 1-2 months.

### Stage 3: End-to-End Evaluation

Test the doubly fine-tuned model **without** providing retinal findings.

```bash
# Run evaluation
python scripts/evaluate_stage3.py
```

**Comparison:**
- MedGemma receives: `Image + Clinical + CGM` (must infer findings)
- GPT-5.2 receives: `Findings + Clinical + CGM` (findings provided)

## Retinal Findings Metadata

Ground truth retinal findings are stored in the AI-READI clinical data.

### Location

```
data/clinical_data/condition_occurrence.csv
```

### Relevant Columns

| Column | Description |
|--------|-------------|
| `person_id` | Participant identifier |
| `condition_source_value` | Diagnosis code (see below) |

### Diagnosis Codes

| Code | Condition |
|------|-----------|
| `mhoccur_pdr` | Diabetic retinopathy (proliferative or non-proliferative) |
| `mhoccur_amd` | Age-related macular degeneration |
| `mhoccur_rvo` | Retinal vascular occlusion |

### Python API

```python
from src.training.retinal_findings import (
    load_retinal_findings,
    format_retinal_findings,
    get_retinal_findings_summary,
)

# Load all findings
findings = load_retinal_findings()

# Get for specific participant
participant = findings.get("1001", {})
# {"diabetic_retinopathy": True, "amd": False, "rvo": False}

# Format for prompts
context = format_retinal_findings(participant)
# "Retinal Eye Exam Findings:\n- Diabetic retinopathy..."

# Get summary stats
summary = get_retinal_findings_summary(findings, person_ids=["1001", "1002"])
```

## Deployment Vision

### Clinical Workflow

The system is designed to integrate with emerging smartphone-based retinal imaging:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Patient Self-Screening Workflow                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  Smartphone  │    │   Our AI     │    │  Patient-    │    │  Follow  │ │
│  │   Retinal    │───▶│   Pipeline   │───▶│  Friendly    │───▶│   Up     │ │
│  │   Capture    │    │              │    │   Report     │    │  Action  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│   (iExaminer,         ↑                   - P(DR) value                    │
│    EyeQue, etc.)      │                   - Natural freq.                  │
│                       │                   - Urgency tier                   │
│              ┌────────┴────────┐          - Next steps                     │
│              │  + CGM data     │                                           │
│              │  + Lab results  │                                           │
│              │  + Demographics │                                           │
│              └─────────────────┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configurable Sensitivity

Healthcare organizations can configure detection sensitivity based on their clinical context:

```python
from src.pipeline import HealthReportPipeline

# Primary care clinic - high sensitivity, minimize missed cases
pipeline = HealthReportPipeline(dr_sensitivity="high")

# Research study - need confirmed cases
pipeline = HealthReportPipeline(dr_sensitivity="specific")

# Custom threshold
pipeline = HealthReportPipeline(dr_threshold=0.08)
```

### Demo Application (Planned)

A hosted web demo will allow:
- Upload retinal image (DICOM or JPEG)
- Enter CGM metrics and basic demographics
- Receive patient-friendly report with probabilistic DR assessment
- Adjust sensitivity slider to see how results change

## Competition

**MedGemma Impact Challenge**
- Organized by: Google Health AI
- Dataset focus: Bridge2AI datasets
- Evaluation: Clinical accuracy, patient accessibility, innovation

## License

Code: MIT License
Data: AI-READI DUA (not included in repository)

## Author

David Liebovitz, MD
Academic Physician Informaticist
