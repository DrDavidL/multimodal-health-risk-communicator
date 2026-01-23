# CLAUDE.md

Context and instructions for Claude Code working on this project.

## Project Summary

Building a **Multimodal Health Risk Communicator** for the MedGemma Impact Challenge. The system takes three data modalities (retinal fundus images, CGM data, clinical data) from the AI-READI dataset and generates patient-friendly health explanations using MedGemma 4B.

## Current Status

### Completed
- [x] AI-READI dataset access approved and data transferred to Azure
- [x] Azure storage configured (`nwaireadi2026` / `aireadi-data` container)
- [x] MedGemma 4B downloaded and running locally (M3 Mac, 64GB RAM)
- [x] Prompt formatting issues resolved — successful local inference on fundus images
- [x] DICOM loader built with YBR_FULL_422 → RGB color space conversion

### In Progress
- [ ] CGM data loader (likely Dexcom CSV format)
- [ ] Clinical data loader (JSON/TSV format TBD)
- [ ] Multimodal fusion pipeline
- [ ] Patient-friendly report generation prompts

### Not Started
- [ ] Evaluation metrics
- [ ] Testing suite
- [ ] Documentation for judges

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

### Data Location (Azure)

```
aireadi-data/
├── clinical_data/           # Clinical measurements
├── retinal_photography/     # DICOM fundus images
├── wearable_blood_glucose/  # CGM data (likely Dexcom CSV)
├── participants.json        # Subject-level linkage
├── participants.tsv         # Same, tabular format
├── dataset_structure_description.json  # Schema documentation
└── study_description.json   # Protocol details
```

**Data is DUA-protected** — never commit any data files, participant IDs, or derived datasets.

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

2. **Modular loaders**: Each data modality has independent loader, enabling partial data processing when modalities are missing.

3. **Patient accessibility focus**: Output should be readable by a patient with no medical training. Avoid jargon; explain implications.

4. **Multimodal fusion strategy**: TBD — options include:
   - Sequential prompting (describe each modality, then synthesize)
   - Structured prompt with all data
   - Multi-turn conversation

## Common Tasks

### Sync data from Azure
```bash
az storage blob download-batch \
    --destination ./data \
    --source aireadi-data \
    --account-name nwaireadi2026
```

### Run MedGemma inference
```bash
python -m src.models.medgemma --image path/to/fundus.dcm
```

### Run tests
```bash
pytest tests/ -v
```

## Questions to Resolve

1. What is the exact CGM data format? (Need to inspect files)
2. What clinical variables are available and relevant?
3. How to handle participants with missing modalities?
4. What evaluation metrics will judges use?

## Resources

- [AI-READI Documentation](https://docs.aireadi.org)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-4b-it)
- [Bridge2AI Overview](https://bridge2ai.org)
- [MedGemma Impact Challenge Rules](https://health.google/medgemma-challenge) (verify URL)
