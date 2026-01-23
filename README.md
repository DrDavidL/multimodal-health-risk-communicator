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
- ~4,000 participants with Type 2 diabetes and matched controls
- Multimodal: retinal imaging, wearables, clinical data
- Access: Approved via AIREADI.org DUA

## Model

**MedGemma 4B** (multimodal variant)
- Google's medical foundation model
- Supports image + text input
- Running locally on M3 Mac (64GB RAM)
- Hugging Face: `google/medgemma-4b-it`

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Retinal DICOM  │     │    CGM CSV      │     │  Clinical JSON  │
│     Loader      │     │    Loader       │     │     Loader      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Multimodal Fusion     │
                    │   & Prompt Construction │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      MedGemma 4B        │
                    │   (Image + Text Input)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Patient-Friendly       │
                    │  Health Report          │
                    └─────────────────────────┘
```

## Project Structure

```
multimodal-health-risk-communicator/
├── src/
│   ├── loaders/
│   │   ├── dicom_loader.py      # Retinal DICOM with YBR→RGB conversion
│   │   ├── cgm_loader.py        # Dexcom CGM data processing
│   │   └── clinical_loader.py   # Clinical data extraction
│   ├── models/
│   │   ├── medgemma.py          # MedGemma wrapper
│   │   └── prompts.py           # Prompt templates
│   ├── pipeline/
│   │   ├── fusion.py            # Multimodal data fusion
│   │   └── report_generator.py  # Patient report generation
│   └── utils/
│       └── azure_storage.py     # Azure blob helpers
├── notebooks/
│   └── exploration.ipynb
├── configs/
│   └── default.yaml
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   └── DATA.md
├── CLAUDE.md                    # Context for Claude Code
├── TODO.md                      # Current status and next steps
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
git clone https://github.com/YOUR_USERNAME/multimodal-health-risk-communicator.git
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

Data is stored in Azure Blob Storage (not in repo — DUA protected):
```bash
# Set environment variables
export AZURE_STORAGE_ACCOUNT=nwaireadi2026
export AZURE_CONTAINER=aireadi-data

# Download sample data for development
az storage blob download-batch \
    --destination ./data \
    --source aireadi-data \
    --pattern "participants.*"
```

## Usage

```python
from src.loaders import DICOMLoader, CGMLoader, ClinicalLoader
from src.models import MedGemmaInference
from src.pipeline import ReportGenerator

# Load multimodal data for a participant
participant_id = "sub-001"
fundus = DICOMLoader.load(f"data/retinal_photography/{participant_id}")
cgm = CGMLoader.load(f"data/wearable_blood_glucose/{participant_id}")
clinical = ClinicalLoader.load(f"data/clinical_data/{participant_id}")

# Generate patient-friendly report
generator = ReportGenerator(model="medgemma-4b")
report = generator.generate(fundus, cgm, clinical)
print(report)
```

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
