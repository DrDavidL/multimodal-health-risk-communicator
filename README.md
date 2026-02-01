# Multimodal Health Risk Communicator

**MedGemma Impact Challenge Submission**

A multimodal AI system that chains **two fine-tuned MedGemma models** — a community-trained DR detection adapter and our locally fine-tuned report generation adapter — to transform retinal fundus images, continuous glucose monitoring (CGM), and clinical data into patient-friendly health risk reports.

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

## Models

The pipeline uses **two fine-tuned MedGemma 4B models** (LoRA adapters on the same base), chained sequentially:

| # | Model | Source | Input | Output |
|---|-------|--------|-------|--------|
| 1 | **DR Detection** | [Community LoRA](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy) (HuggingFace) | Retinal fundus image | DR grade probabilities (A–E) |
| 2 | **Report Generation** | Our LoRA (fine-tuned locally) | P(DR) + clinical labs + CGM + demographics | Patient-friendly health report |

Both are LoRA adapters (~30MB each) on the same `google/medgemma-4b-it` base model (~8GB). At inference, adapters are swapped via `model.set_adapter()` — total memory ~8GB, fits a T4 GPU (16GB).

### Apple Silicon Training Optimizations

Stage 2 fine-tuning runs entirely on-device using MPS (Metal Performance Shaders):

| Optimization | Description |
|--------------|-------------|
| `device_map={"": "mps"}` | Load model directly to MPS, avoiding CPU→GPU copy |
| `torch_dtype=torch.float32` | Float32 for numerical stability (float16 caused NaN loss) |
| `torch.compile(backend="aot_eager")` | JIT compilation for ~15% speedup |
| LoRA rank=16 | Low-rank adaptation with 0.76% trainable parameters |
| Manual training loop | Better MPS compatibility than HF Trainer |
| Gradient accumulation=4 | Effective batch size of 4 with batch_size=1 |

## Architecture

### Two-Model Pipeline

The system chains two fine-tuned MedGemma models — one for vision (DR detection from retinal images) and one for language (patient report generation from multimodal clinical data):

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MODEL 1: DR Detection (Community LoRA from HuggingFace)                 │
│ ┌──────────────┐                     ┌─────────────────────────────┐   │
│ │Retinal Image │ ──── MedGemma ────▶ │ DR Grade Probabilities      │   │
│ └──────────────┘   + DR LoRA adapter │ P(A), P(B), P(C), P(D), P(E)│  │
│                    (vision task)      └──────────────┬──────────────┘   │
└─────────────────────────────────────────────────────│───────────────────┘
                                                      │ P(DR) = P(B)+P(C)+P(D)+P(E)
                                            (swap adapter)
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODEL 2: Report Generation (Our LoRA, fine-tuned locally)               │
│ ┌──────────────┐                                                        │
│ │ P(DR) score  │                     ┌─────────────────────────────┐   │
│ ├──────────────┤ ──── MedGemma ────▶ │  Patient-Friendly Report   │   │
│ │ Clinical labs│  + Report LoRA      │  (probabilistic framing,   │   │
│ ├──────────────┤    adapter          │   health-literate language) │   │
│ │ CGM data     │  (text-only task)   └─────────────────────────────┘   │
│ ├──────────────┤                                                        │
│ │ Demographics │                                                        │
│ └──────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**End-to-end inference** chains both models on the same base:

```
┌──────────────┐   DR LoRA    ┌────────────┐  Report LoRA ┌─────────────┐
│Retinal Image │ ──adapter──▶ │ P(DR)=0.42 │ ──adapter──▶ │   Patient   │
├──────────────┤  (swap #1)   └────────────┘  (swap #2)   │   Report    │
│ Clinical labs│ ─────────────────────────────────────────▶│             │
├──────────────┤                                           │ • P(DR)     │
│ CGM data     │ ─────────────────────────────────────────▶│ • Next steps│
├──────────────┤                                           │ • Questions │
│ Demographics │ ─────────────────────────────────────────▶│   for doc   │
└──────────────┘                                           └─────────────┘
  Adapters swapped on SAME base model via model.set_adapter()
```

### Why Two Models?

The key insight is **adapter swapping, not stacked fine-tuning**. Each adapter was trained independently on the base model for a different task:

- **Model 1 (DR LoRA)** was trained by the community on the [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset to grade diabetic retinopathy severity from fundus images
- **Model 2 (Report LoRA)** was fine-tuned by us on GPT-5.2–generated training targets to translate clinical data + DR probabilities into patient-friendly reports

At inference, the DR adapter runs first to produce grade probabilities from the retinal image, then the Report adapter generates the patient report using those probabilities plus clinical and CGM data. This means the system can **detect and communicate** diabetic retinopathy end-to-end — it does not require pre-labeled findings.

| Component | Adapter | Size | Task |
|-----------|---------|------|------|
| DR Detection | Community LoRA | ~30MB | Image → P(DR) |
| Report Gen | Our LoRA | ~30MB | Text → Patient report |
| Lifestyle Q&A | Same as Report | — | Grounded Q&A |

### Comparison to Baseline

| Model | Input | Retinal Findings |
|-------|-------|------------------|
| **MedGemma (ours)** | Image + Clinical + CGM | Inferred via DR LoRA adapter |
| **GPT-5.2 (baseline)** | Text only | Provided as metadata |

## Project Structure

```
multimodal-health-risk-communicator/
├── app/
│   ├── demo.py                  # Gradio web demo application
│   ├── requirements.txt         # App dependencies
│   └── examples/                # Sample images for demo
├── src/
│   ├── loaders/
│   │   ├── azure_storage.py     # Azure download with caching
│   │   ├── dicom_loader.py      # Retinal DICOM with YBR→RGB conversion
│   │   ├── cgm_loader.py        # CGM JSON (Open mHealth format)
│   │   ├── clinical_loader.py   # Clinical CSV (OMOP CDM format)
│   │   └── participant.py       # Unified multimodal loader
│   ├── models/
│   │   ├── medgemma.py          # MedGemma 4B wrapper
│   │   └── dr_detector.py       # High-sensitivity DR detection
│   ├── training/
│   │   ├── dataset.py           # Training dataset creation
│   │   ├── trainer.py           # LoRA fine-tuning
│   │   └── retinal_findings.py  # Retinal metadata extraction
│   └── pipeline/
│       └── report_generator.py  # Patient report generation
├── scripts/
│   ├── prepare_stage1_data.py   # Stage 1 data preparation
│   ├── prepare_stage2_data.py   # Stage 2 data preparation
│   ├── prepare_stage2_probabilistic.py  # Probabilistic report data
│   ├── run_stage1_training.py   # Train visual understanding
│   ├── run_stage2_training.py   # Train report generation
│   ├── evaluate_stage3.py       # End-to-end evaluation
│   ├── convert_adapter.py       # Strip torch.compile prefix for adapter compatibility
│   └── azure_query.py           # BAA-compliant Azure GPT-5.2 queries
├── docs/
│   ├── ARCHITECTURE.md          # System architecture
│   ├── DATA.md                  # Dataset documentation
│   └── EVALUATION_PLAN.md       # Evaluation methodology
├── outputs/                     # Model weights and results
│   ├── medgemma-stage1/         # Stage 1 adapter + evaluation
│   └── medgemma-stage2-probabilistic/  # Stage 2 adapter (post-training)
├── configs/default.yaml         # Configuration
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

**Important:** All data is nested under a UUID prefix in Azure (provided with your DUA credentials).

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

## Training & Evaluation Pipeline

### Model 1: DR Detection (Community LoRA)

We use a community-trained DR-specific LoRA adapter from HuggingFace: [`qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy`](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)

**Task:** `Retinal fundus image → DR grade probabilities (A–E)`

**Method:** Extract grade-level probabilities and compute P(DR) = P(B) + P(C) + P(D) + P(E)

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

<details>
<summary>Why not train our own DR model?</summary>

We initially attempted to fine-tune MedGemma on AI-READI retinal images for DR/AMD/RVO detection. The model achieved 83.3% overall accuracy but **0% sensitivity** for DR and AMD — it missed all true positive cases. This is unacceptable for a diabetes screening application where false negatives are costly. The community-trained adapter, trained on the larger APTOS 2019 dataset, provides far better DR sensitivity.
</details>

### Model 2: Report Generation (Our LoRA, fine-tuned locally)

Fine-tuned on generating patient reports that communicate DR findings **probabilistically**, using clinical context from all three modalities.

```bash
# Prepare training data with probabilistic framing (uses DR model + GPT-5.2)
python scripts/prepare_stage2_probabilistic.py

# Train (outputs to ./outputs/medgemma-stage2/)
python scripts/run_stage2_training.py
```

**Task:** `P(DR) + clinical labs + CGM + demographics → patient-friendly report`

This same adapter also powers **interactive Q&A** in the deployed app — after a report is generated, patients can ask follow-up questions about diabetes management, and the Report LoRA responds with answers grounded in ADA/NEI/AAO clinical guidelines.

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

#### Technical Notes

- **Text-only training**: This model uses no images. Loaded with `AutoModelForCausalLM` (not `AutoModelForVision2Seq`) since the task is purely text-to-text (structured clinical context → patient report).
- **Post-training conversion**: Training with `torch.compile()` prepends `_orig_mod.` to all parameter keys. Before uploading the adapter to Hugging Face Hub, run `scripts/convert_adapter.py` to strip this prefix and ensure compatibility with standard PEFT loading.

```bash
# Convert adapter after training
python scripts/convert_adapter.py \
    --input ./outputs/medgemma-stage2-probabilistic/adapter \
    --output ./outputs/medgemma-stage2-converted/
```

### Evaluation

Evaluates Model 2 (Report LoRA) on held-out test participants using **text-only** pre-computed data — no images are processed. The evaluation has two phases:

1. **Report generation** — Model 2 generates reports from pre-computed P(DR) + clinical + CGM text
2. **Automated judging** — Azure GPT-5.2 scores each MedGemma report against the GPT-5.2 baseline on 6 criteria (1–5 scale): clinical accuracy, probability communication, actionability, readability, completeness, overall quality

```bash
# Run evaluation (requires Azure OpenAI credentials in .env)
python scripts/evaluate_stage3.py
```

**Inputs (from test manifest, no data download needed):**
- Pre-computed P(DR) values from Model 1
- Clinical context, CGM context (text)
- GPT-5.2 baseline reports (generated during training data prep)

**Outputs saved to `outputs/evaluation/`:**
- `summary.json` — aggregate scores and per-participant GPT-5.2 judgments
- `detailed_results.json` — full MedGemma vs GPT-5.2 reports side-by-side

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

### Demo Application

A Gradio-based web demo ([`app/demo.py`](app/demo.py)) runs the full two-model pipeline:

1. **Upload** a retinal fundus image (DICOM or JPEG) and enter CGM metrics + demographics
2. **Model 1 (DR LoRA)** processes the image → P(DR) grade probabilities
3. **Model 2 (Report LoRA)** takes P(DR) + clinical data → patient-friendly report
4. **Interactive Q&A** — after the report is generated, patients can ask follow-up questions about their results, diabetes management, and lifestyle changes. The same Report LoRA adapter responds with answers grounded in ADA/NEI/AAO clinical guidelines.

Additional features:
- Adjustable sensitivity slider to explore different DR detection thresholds
- Template-based fallback if model adapters are unavailable

**Deployed on HuggingFace Spaces** with ZeroGPU (free T4 GPU allocation on demand):

```bash
# Run locally
cd app && python demo.py

# Deployed at:
# https://huggingface.co/spaces/drdavidl/dr-screening-assistant
```

## Competition

**MedGemma Impact Challenge**
- Organized by: Google Health AI
- Dataset focus: Bridge2AI datasets
- Evaluation: Clinical accuracy, patient accessibility, innovation

## Acknowledgments

### Data

- **[AI-READI](https://aireadi.org)** — Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights. We gratefully acknowledge the AI-READI consortium for creating and curating this multimodal diabetes dataset.
- **[Bridge2AI](https://bridge2ai.org)** — This work uses data generated by the Bridge2AI program, supported by the **National Institutes of Health (NIH)** Common Fund. Bridge2AI aims to generate flagship datasets and best practices for the ethical use of AI in biomedical research.

**Required Citations (per AI-READI license):**

> AI-READI Consortium. (2024). "AI-READI: rethinking data collection, preparation and sharing for propelling AI-based discoveries in diabetes research and beyond." *Nature Metabolism*. https://doi.org/10.1038/s42255-024-01165-x

> AI-READI Consortium. (2025). Flagship Dataset of Type 2 Diabetes from the AI-READI Project (3.0.0) [Data set]. FAIRhub. https://doi.org/10.60775/fairhub.3

### Models

- **[Google Health AI](https://health.google)** — For developing and open-sourcing **MedGemma** (`google/medgemma-4b-it`), the medical foundation model at the core of this project, and for organizing the MedGemma Impact Challenge.
- **[qizunlee](https://huggingface.co/qizunlee)** — For the community-contributed DR detection LoRA adapter ([`medgemma-4b-it-sft-lora-diabetic-retinopathy`](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)), which serves as Model 1 in our two-model pipeline for detecting diabetic retinopathy from retinal images. This adapter was trained on data from the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) competition.

### Infrastructure

- **[Microsoft Azure](https://azure.microsoft.com)** — BAA-compliant cloud infrastructure for secure data storage and analysis via Azure OpenAI Service.
- **[Hugging Face](https://huggingface.co)** — Model hosting, Transformers, PEFT, and Spaces platform for deployment.

### Development Tools

- **[Anthropic Claude](https://www.anthropic.com)** — AI coding assistant used for software development, architecture design, and documentation. Per DUA terms, Claude was not exposed to any patient-level data — all data analysis was performed within the BAA-compliant Azure environment.

### Open-Source Software

This project builds on the work of many open-source communities:

[PyTorch](https://pytorch.org) | [Hugging Face Transformers](https://github.com/huggingface/transformers) | [PEFT](https://github.com/huggingface/peft) | [Gradio](https://gradio.app) | [pydicom](https://pydicom.github.io) | [pandas](https://pandas.pydata.org)

## License

Code: MIT License
Data: AI-READI DUA (not included in repository)
Model Adapters: Released under the same terms as MedGemma (see [model card](https://huggingface.co/google/medgemma-4b-it))

## Author

David Liebovitz, MD
Academic Physician Informaticist
