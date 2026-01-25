# CLAUDE.md

Context and instructions for Claude Code working on this project.

## Project Summary

Building a **Multimodal Health Risk Communicator** for the MedGemma Impact Challenge. The system takes three data modalities (retinal fundus images, CGM data, clinical data) from the AI-READI dataset and generates patient-friendly health explanations using MedGemma 4B.

## Current Status

### Completed
- [x] AI-READI dataset access approved and data transferred to Azure
- [x] Azure storage configured (Azure Blob Storage)
- [x] MedGemma 4B downloaded and running locally (M3 Mac, 64GB RAM)
- [x] Prompt formatting issues resolved — successful local inference on fundus images
- [x] DICOM loader built with YBR_FULL_422 → RGB color space conversion
- [x] Data structure explored — confirmed formats for all 3 modalities
- [x] Sample participant data downloaded (1001) for testing
- [x] CGM data loader (JSON Open mHealth format with glycemic metrics)
- [x] Clinical data loader (OMOP CDM CSV format with measurement mapping)
- [x] Azure blob downloader with lazy-download-and-cache pattern
- [x] Unified ParticipantLoader orchestrating all three modalities
- [x] Multimodal fusion pipeline (ReportGenerator)
- [x] Patient-friendly report generation prompts
- [x] Azure OpenAI (GPT-5.2) integration for BAA-compliant data analysis
- [x] LoRA fine-tuning pipeline for MedGemma (DUA Section 3.D compliant)

### In Progress
- [ ] Stage 1: Visual understanding fine-tuning (image → findings)
- [ ] Stage 2: Report generation fine-tuning (text → report)
- [ ] Stage 3: End-to-end evaluation (vs GPT-5.2 baseline)

### Not Started
- [ ] Memorization validation testing
- [ ] Final evaluation report for judges
- [ ] Testing suite

## Technical Context

### Environment
- **Local dev**: M3 Mac, 64GB RAM, Python 3.10+
- **Cloud storage**: Azure Blob Storage
- **Compute (optional)**: Colab Pro available if needed

### MedGemma Usage

The model requires specific prompt formatting. Working example:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

model_id = "google/medgemma-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id)

# Image must be PIL Image, prompt uses specific chat format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Describe any abnormalities in this retinal fundus image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True,
    return_tensors="pt"
)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

### DICOM Handling

AI-READI retinal images use YBR_FULL_422 color space. Critical conversion:

```python
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from PIL import Image

def load_fundus_dicom(path):
    dcm = pydicom.dcmread(path)
    pixels = dcm.pixel_array

    # Convert color space if needed
    if dcm.PhotometricInterpretation == "YBR_FULL_422":
        pixels = convert_color_space(pixels, "YBR_FULL_422", "RGB")

    return Image.fromarray(pixels)
```

**DICOM metadata available:** `PatientID`, `Modality` (OP), image dimensions (typically 1836×3293).

### CGM Data Format

CGM files are JSON using **Open mHealth** schema (not CSV as initially assumed):

```python
import json

def load_cgm_data(path):
    with open(path) as f:
        data = json.load(f)

    # Extract glucose readings
    readings = []
    for entry in data["body"]["cgm"]:
        readings.append({
            "timestamp": entry["effective_time_frame"]["time_interval"]["start_date_time"],
            "glucose_mg_dl": entry["blood_glucose"]["value"],
            "event_type": entry["event_type"]  # "EGV" for estimated glucose value
        })
    return readings
```

**CGM metadata:** `header.patient_id` (e.g., "AIREADI-1001"), ~10 days of 5-minute readings.

### Clinical Data Format

Clinical data uses **OMOP CDM** format (CSV files):

```python
import csv

def load_clinical_measurements(measurement_csv, person_id):
    measurements = {}
    with open(measurement_csv) as f:
        for row in csv.DictReader(f):
            if row["person_id"] == person_id:
                source = row["measurement_source_value"]
                # Extract variable name after comma
                name = source.split(", ")[1] if ", " in source else source
                measurements[name] = float(row["value_as_number"])
    return measurements
```

**Key clinical variables:** HbA1c, fasting glucose, BMI, blood pressure, lipid panel, creatinine, visual acuity scores, MoCA cognitive scores.

### Data Location (Azure)

**Critical:** All blobs are nested under a UUID prefix:
```
BLOB_PREFIX = "<UUID>/dataset"  # Actual UUID redacted - set via environment variable
```

**Confirmed structure:**
```
{BLOB_PREFIX}/
├── clinical_data/
│   ├── measurement.csv      # Labs, vitals, vision tests (OMOP CDM)
│   ├── person.csv           # Demographics
│   ├── condition_occurrence.csv
│   ├── observation.csv      # Survey responses
│   └── visit_occurrence.csv
├── retinal_photography/
│   └── cfp/icare_eidon/{person_id}/
│       ├── {id}_eidon_mosaic_cfp_l_*.dcm     # Left eye mosaic
│       ├── {id}_eidon_mosaic_cfp_r_*.dcm     # Right eye mosaic
│       ├── {id}_eidon_uwf_central_cfp_*.dcm  # Ultra-wide field
│       ├── {id}_eidon_uwf_nasal_cfp_*.dcm
│       └── {id}_eidon_uwf_temporal_cfp_*.dcm
├── wearable_blood_glucose/
│   └── continuous_glucose_monitoring/dexcom_g6/{person_id}/
│       └── {id}_DEX.json    # ~10 days CGM data (Open mHealth JSON)
├── participants.tsv         # Linkage + modality availability flags
└── participants.json        # Column definitions
```

**Linkage key:** `person_id` (e.g., `1001`) — consistent across all files.

**Data availability:** 2,244 of 2,280 participants have complete data for retinal + CGM + clinical.

**Data is DUA-protected** — never commit any data files, participant IDs, or derived datasets.

## Retinal Findings Metadata

**IMPORTANT:** Ground truth retinal diagnoses are in the clinical data, NOT detected from images.

### Location
```
data/clinical_data/condition_occurrence.csv
```

### How to Access
```python
from src.training.retinal_findings import load_retinal_findings, format_retinal_findings

# Load all findings (returns dict: person_id → findings)
findings = load_retinal_findings()

# Get for specific participant
participant_findings = findings.get("1001", {})
# Returns: {"diabetic_retinopathy": True, "amd": False, "rvo": False}

# Format for prompts
context = format_retinal_findings(participant_findings)
# "Retinal Eye Exam Findings:\n- Diabetic retinopathy..."
```

### Diagnosis Codes (in condition_source_value column)
| Code | Condition |
|------|-----------|
| `mhoccur_pdr` | Diabetic retinopathy |
| `mhoccur_amd` | Age-related macular degeneration |
| `mhoccur_rvo` | Retinal vascular occlusion |

### Note
- Presence of row = positive diagnosis (no value needed)
- Use `findings.get(person_id, {})` for safe access
- See `src/training/retinal_findings.py` for full API

## DUA-Compliant Data Access

**CRITICAL:** AI-READI data is protected under a Data Use Agreement. Follow this workflow:

| Task | Tool | Reason |
|------|------|--------|
| Write/edit code | Claude Code | Code doesn't contain data |
| Analyze patient data | Azure GPT-5.2 | BAA-compliant, DUA-approved |
| View data samples | Azure GPT-5.2 | Keep data in Azure environment |
| Generate training targets | Azure GPT-5.2 | Data stays in compliant environment |

**Claude Code should NOT:**
- Directly analyze or interpret patient-level data
- Display or summarize individual participant records
- Make clinical inferences from the data

**Azure GPT-5.2 handles:**
- Reading and analyzing clinical measurements
- Generating patient-friendly explanations from data
- Extracting patterns from CGM/retinal findings
- Creating training dataset annotations

**Query Azure GPT-5.2:**
```bash
python scripts/azure_query.py "Analyze the clinical data for participant..."
```

## Coding Conventions

### Style
- Python type hints throughout
- Docstrings for all public functions (Google style)
- `black` for formatting, `ruff` for linting

### File Organization
- Loaders in `src/loaders/` — one file per modality
- Model code in `src/models/` — keep MedGemma wrapper thin
- Pipeline logic in `src/pipeline/`
- Configuration via YAML in `configs/`

### Error Handling
- Graceful handling of missing modalities (not all participants have all data)
- Explicit validation of DICOM color spaces before processing
- Clear error messages for Azure connectivity issues

## Key Design Decisions

1. **Local-first development**: MedGemma runs locally to avoid API costs and latency during iteration. Cloud deployment deferred.

2. **Lazy-download-and-cache data strategy**:
   - Full dataset is ~50GB (mostly retinal images) — impractical to download entirely
   - Clinical CSVs downloaded once (~150MB shared across all participants)
   - CGM and retinal data downloaded on-demand per participant, cached locally
   - Minimizes Azure egress costs; 50 participants ≈ 1.1GB ≈ $0.09
   - Reproducible: others only need Azure access + code to replicate

3. **Modular loaders**: Each data modality has independent loader, enabling partial data processing when modalities are missing.

4. **Patient accessibility focus**: Output should be readable by a patient with no medical training. Avoid jargon; explain implications.

5. **Multimodal fusion strategy**: Structured prompt with all data
   - Clinical and CGM summaries formatted as text context
   - Retinal image(s) passed as vision input
   - Single comprehensive prompt requesting patient-friendly explanation
   - Prompt includes structured output sections (findings, meaning, takeaways, questions)

## Common Tasks

### Download participant data from Azure
```bash
# Set these from your environment (not committed to repo)
BLOB_PREFIX="$AIREADI_BLOB_PREFIX"  # UUID prefix from DUA
STORAGE_ACCOUNT="$AIREADI_STORAGE_ACCOUNT"
CONTAINER="$AIREADI_CONTAINER"
PERSON_ID="<person_id>"  # e.g., 1001

# Download retinal images
az storage blob download --account-name $STORAGE_ACCOUNT --container-name $CONTAINER \
    --name "$BLOB_PREFIX/retinal_photography/cfp/icare_eidon/$PERSON_ID/${PERSON_ID}_eidon_mosaic_cfp_l_*.dcm" \
    --file "./data/participants/$PERSON_ID/retinal/"

# Download CGM data
az storage blob download --account-name $STORAGE_ACCOUNT --container-name $CONTAINER \
    --name "$BLOB_PREFIX/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/$PERSON_ID/${PERSON_ID}_DEX.json" \
    --file "./data/participants/$PERSON_ID/cgm/${PERSON_ID}_DEX.json"

# Clinical data is in shared CSVs — filter by person_id after download
```

### Run demo (data loading only)
```bash
python scripts/demo_report.py 1001
```

### Run demo with MedGemma inference
```bash
python scripts/demo_report.py 1001 --run-inference
```

### Generate patient report
```bash
python -m src.pipeline.report_generator 1001
```

### Run MedGemma inference on single image
```bash
python -m src.models.medgemma path/to/fundus.dcm
```

### Run tests
```bash
pytest tests/ -v
```

### Fine-tune MedGemma
```bash
# Prepare training dataset (uses GPT-5.2 to generate target responses)
python scripts/finetune.py prepare --max-participants 100 --generate-targets --auto-download

# Train with LoRA
python scripts/finetune.py train --data ./data/training --epochs 3

# Full pipeline
python scripts/finetune.py full --participants 50
```

## 3-Stage Fine-Tuning Strategy

**Per DUA Section 3.D:** Licensee Models trained on the Data are permitted, provided:
- Models do not contain the Data itself
- Reasonable efforts to minimize memorization/reconstruction

### Stage 1: Visual Understanding
- **Input:** Retinal fundus image only
- **Output:** Retinal findings (DR, AMD, RVO detection)
- **Ground truth:** Clinical diagnoses from `condition_occurrence.csv`
- **Goal:** Teach MedGemma to identify diabetic retinopathy from images

### Stage 2: Report Generation (Text-Only)
- **Input:** Retinal findings + Clinical + CGM (no images)
- **Output:** Patient-friendly health report
- **Ground truth:** GPT-5.2 generated reports
- **Goal:** Teach text synthesis and patient communication

### Stage 3: End-to-End Evaluation
- **Input:** Image + Clinical + CGM (NO retinal findings provided)
- **Output:** Patient report (model must infer findings from image)
- **Comparison:** vs GPT-5.2 given explicit findings
- **Goal:** Demonstrate learned visual understanding

### Privacy Safeguards
1. **LoRA adapters**: Only ~0.1% of parameters trained → lower memorization
2. **Anonymization**: Person IDs replaced with P1000, P1001, etc.
3. **Value jittering**: 2% noise on numeric values
4. **Memorization testing**: Post-training similarity check < 5%

### Pipeline Commands
```bash
# Stage 1: Visual understanding
python scripts/prepare_stage1_data.py
python scripts/run_stage1_training.py

# Stage 2: Report generation
python scripts/prepare_stage2_data.py
python scripts/run_stage2_training.py

# Stage 3: Evaluation
python scripts/evaluate_stage3.py
```

## Questions to Resolve

1. ~~What is the exact CGM data format?~~ → **Resolved:** JSON with Open mHealth schema
2. ~~What clinical variables are available?~~ → **Resolved:** 90+ variables including HbA1c, glucose, lipids, vitals, vision, cognition
3. ~~How to handle participants with missing modalities?~~ → **Resolved:** Graceful degradation — generate partial reports with available data
4. ~~Best multimodal fusion strategy?~~ → **Resolved:** Structured prompt with all data + image
5. What evaluation metrics will judges use?
6. How to validate clinical accuracy of generated reports?

## Resources

- [AI-READI Documentation](https://docs.aireadi.org)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-4b-it)
- [Bridge2AI Overview](https://bridge2ai.org)
- [MedGemma Impact Challenge Rules](https://health.google/medgemma-challenge) (verify URL)
