# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  Azure Blob     │  Retinal DICOM  │  CGM CSV        │  Clinical JSON        │
│  Storage        │  Images         │  Time Series    │  Structured Data      │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOADER LAYER                                      │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  AzureStorage   │  DICOMLoader    │  CGMLoader      │  ClinicalLoader       │
│  Client         │  YBR→RGB conv   │  Stats calc     │  Variable extraction  │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         └─────────────────┴────────┬────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUSION LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MultimodalFusion                                                    │    │
│  │  - Validate data completeness                                        │    │
│  │  - Construct structured prompt                                       │    │
│  │  - Handle missing modalities                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MedGemmaInference                                                   │    │
│  │  - google/medgemma-4b-it                                             │    │
│  │  - Multimodal input (image + text)                                   │    │
│  │  - Local inference (M3 Mac)                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ReportGenerator                                                     │    │
│  │  - Patient-friendly language                                         │    │
│  │  - Risk stratification                                               │    │
│  │  - Actionable recommendations                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Loaders

Each loader is responsible for a single modality and produces a standardized output.

#### DICOMLoader

```python
class DICOMLoader:
    """Load and preprocess retinal fundus DICOM images."""
    
    def load(self, path: str) -> PIL.Image:
        """Load DICOM, convert color space, return PIL Image."""
        
    def load_participant(self, participant_id: str) -> dict[str, PIL.Image]:
        """Load all fundus images for a participant.
        
        Returns:
            {"OD": Image, "OS": Image} or subset if missing
        """
```

**Key transformations**:
- YBR_FULL_422 → RGB color space conversion
- Pixel array to PIL Image
- Optional: resize for model input

#### CGMLoader

```python
class CGMLoader:
    """Load and analyze continuous glucose monitoring data."""
    
    def load(self, path: str) -> pd.DataFrame:
        """Load raw CGM data."""
        
    def calculate_metrics(self, df: pd.DataFrame) -> CGMMetrics:
        """Calculate glycemic variability metrics.
        
        Returns:
            CGMMetrics(
                mean_glucose=...,
                std_glucose=...,
                time_in_range=...,  # 70-180 mg/dL
                time_below_range=...,  # <70 mg/dL
                time_above_range=...,  # >180 mg/dL
                gmi=...,  # Glucose Management Indicator
                cv=...,  # Coefficient of variation
            )
        """
```

#### ClinicalLoader

```python
class ClinicalLoader:
    """Load clinical data for a participant."""
    
    def load(self, participant_id: str) -> ClinicalData:
        """Load and validate clinical data.
        
        Returns:
            ClinicalData(
                age=...,
                sex=...,
                diabetes_duration=...,
                hba1c=...,
                bmi=...,
                blood_pressure=...,
                medications=[...],
            )
        """
```

### 2. Multimodal Fusion

The fusion layer combines outputs from all loaders into a structured prompt.

```python
class MultimodalFusion:
    """Combine multimodal data for model input."""
    
    def fuse(
        self,
        fundus_images: dict[str, PIL.Image],
        cgm_metrics: CGMMetrics,
        clinical: ClinicalData,
    ) -> ModelInput:
        """Create structured input for MedGemma.
        
        Returns:
            ModelInput(
                images=[...],
                text_prompt="...",
                metadata={...},
            )
        """
```

**Fusion strategies** (to be evaluated):

1. **Sequential prompting**: Describe each modality separately, then synthesize
2. **Integrated prompt**: Single comprehensive prompt with all data
3. **Hierarchical**: Low-level analysis → High-level synthesis

### 3. MedGemma Inference

Thin wrapper around the Hugging Face model.

```python
class MedGemmaInference:
    """MedGemma 4B multimodal inference."""
    
    def __init__(self, model_id: str = "google/medgemma-4b-it"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id)
    
    def generate(
        self,
        images: list[PIL.Image],
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text response from multimodal input."""
```

### 4. Report Generation

Transforms model output into patient-friendly format.

```python
class ReportGenerator:
    """Generate patient-friendly health reports."""
    
    def generate(
        self,
        fundus: dict[str, PIL.Image],
        cgm: CGMMetrics,
        clinical: ClinicalData,
    ) -> PatientReport:
        """End-to-end report generation.
        
        Returns:
            PatientReport(
                summary="...",
                findings=[...],
                risks=[...],
                recommendations=[...],
                follow_up=[...],
            )
        """
```

---

## Data Flow Example

```python
# 1. Load data
participant_id = "sub-001"

fundus = dicom_loader.load_participant(
    f"data/retinal_photography/{participant_id}"
)
cgm = cgm_loader.load(
    f"data/wearable_blood_glucose/{participant_id}"
)
cgm_metrics = cgm_loader.calculate_metrics(cgm)
clinical = clinical_loader.load(participant_id)

# 2. Fuse modalities
model_input = fusion.fuse(fundus, cgm_metrics, clinical)

# 3. Generate with MedGemma
raw_output = medgemma.generate(
    images=model_input.images,
    prompt=model_input.text_prompt,
)

# 4. Format report
report = report_generator.format(raw_output, model_input.metadata)
```

---

## Prompt Engineering

### Base Template

```
You are a compassionate health educator explaining medical findings to a patient.
The patient has no medical training. Use simple language and avoid jargon.

## Patient Information
- Age: {age}
- Diabetes duration: {duration} years
- Recent HbA1c: {hba1c}%

## Glucose Patterns (from continuous monitor)
- Average glucose: {mean_glucose} mg/dL
- Time in healthy range (70-180): {tir}%
- Time low (<70): {tbr}%
- Time high (>180): {tar}%

## Retinal Image
[IMAGE]

Please provide:
1. A brief summary of what the retinal image shows
2. How this relates to the patient's diabetes management
3. Key takeaways in simple terms
4. Suggested questions to ask their doctor
```

### Considerations

- **Tone**: Warm, reassuring, not alarming
- **Specificity**: Actionable insights, not vague statements
- **Safety**: Always recommend professional follow-up
- **Accessibility**: Reading level ~8th grade

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Missing fundus images | Generate report from CGM + clinical only |
| Missing CGM data | Generate report from fundus + clinical only |
| Missing clinical data | Require minimum demographics, warn about limitations |
| DICOM read failure | Log error, skip participant, continue batch |
| Model inference OOM | Reduce batch size, retry |
| Invalid color space | Attempt auto-detection, fallback to raw |

---

## Performance Considerations

### Local Inference (M3 Mac, 64GB RAM)
- Model load time: ~30s
- Inference per image: ~5-10s
- Memory usage: ~12GB

### Batch Processing
- Process participants sequentially to avoid OOM
- Consider chunking for large cohorts
- Cache model in memory across participants

### Optimization Opportunities
- Quantization (4-bit) for faster inference
- Image preprocessing pipeline optimization
- Parallel loader execution
