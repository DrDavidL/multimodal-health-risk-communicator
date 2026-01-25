# Demo Application Plan

## Submission Requirements

Per [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge):
- 3-minute video demonstration
- Working demo application
- Reproducible source code
- 3-page technical writeup

## Bonus Prize Categories (Target These!)

| Category | Prize | Our Approach |
|----------|-------|--------------|
| **Novel fine-tuned model adaptations** | Additional award | 2-stage LoRA (vision + text), probabilistic communication |
| **Effective edge AI deployment** | Additional award | Local M4 Mac inference, no cloud required |
| **Agent-based workflows** | Additional award | Optional: add multi-step reasoning agent |

**Key Emphasis in Demo:**
- Show local inference (no cloud API calls)
- Highlight novel LoRA training approach
- Demonstrate configurable sensitivity (edge deployment flexibility)

## App Architecture

### Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Frontend** | Gradio | Fast prototyping, built-in hosting, medical-friendly |
| **Backend** | Python + MedGemma | Direct model integration |
| **Hosting** | Hugging Face Spaces | Free GPU, easy deployment, Gradio native |
| **Alternative** | Streamlit on Railway | If HF Spaces GPU unavailable |

### Why Gradio over Streamlit
- Native Hugging Face Spaces support with free GPU
- Built-in image upload components
- Easier model loading in Spaces
- Better for ML demos specifically

## User Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEMO APPLICATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Upload Retinal Image                                         │   │
│  │ ┌──────────────────────────────────────────────────────────────────┐│   │
│  │ │  [  Drop DICOM or JPG here  ]                                    ││   │
│  │ │                                                                   ││   │
│  │ │  Or use example: [Sample 1] [Sample 2] [Sample 3]                ││   │
│  │ └──────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Enter Patient Context (Optional)                            │   │
│  │                                                                      │   │
│  │  Diabetes Status: [Dropdown: Type 2 / Pre-diabetes / None]          │   │
│  │  HbA1c (%):       [  7.2  ]                                          │   │
│  │  Years with DM:   [  8    ]                                          │   │
│  │                                                                      │   │
│  │  CGM Data Available?  ☐ Yes  ☑ No                                   │   │
│  │  └─ If yes: Avg glucose [___] mg/dL, Time in range [___]%           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Configure Detection Sensitivity                             │   │
│  │                                                                      │   │
│  │  Sensitivity Preset:                                                 │   │
│  │  ○ Screening (highest sensitivity, some false positives)            │   │
│  │  ● High Sensitivity (recommended for primary care)                  │   │
│  │  ○ Balanced (moderate sensitivity/specificity)                      │   │
│  │  ○ High Specificity (fewest false positives)                        │   │
│  │                                                                      │   │
│  │  [──────●────────────] Threshold: 0.05                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         [ Generate Report ]                                 │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ RESULTS                                                              │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ DR Analysis                                         [?] Help │   │   │
│  │  │                                                              │   │   │
│  │  │  P(DR): ████████░░ 78%                                       │   │   │
│  │  │                                                              │   │   │
│  │  │  Interpretation: Diabetic retinopathy is LIKELY              │   │   │
│  │  │                                                              │   │   │
│  │  │  "If we screened 10 people with similar results,             │   │   │
│  │  │   about 8 would have diabetic retinopathy and 2 would not." │   │   │
│  │  │                                                              │   │   │
│  │  │  Urgency: 🔴 URGENT - See eye doctor within 2 weeks          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ Patient-Friendly Report                        [📋 Copy]     │   │   │
│  │  │                                                              │   │   │
│  │  │  ## Understanding Your Retinal Screening Results             │   │   │
│  │  │                                                              │   │   │
│  │  │  The screening found clear signs that suggest diabetic       │   │   │
│  │  │  retinopathy is likely. If we screened 10 people with        │   │   │
│  │  │  similar results, about 8 would have diabetic retinopathy... │   │   │
│  │  │                                                              │   │   │
│  │  │  ## What You Should Do Next                                  │   │   │
│  │  │  ...                                                         │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features to Showcase

### 1. Probabilistic Communication
- Natural frequency display ("X out of 10")
- Visual probability bar
- Clear screening vs diagnosis distinction

### 2. Configurable Sensitivity
- Named presets with clinical guidance
- Real-time threshold adjustment
- See how results change at different thresholds

### 3. Multimodal Integration
- Image analysis (retinal fundus)
- Clinical context integration
- CGM data when available

### 4. Accessibility Focus
- Simple language (8th grade reading level)
- Clear urgency indicators
- Actionable recommendations

## Technical Implementation

### File Structure

```
app/
├── app.py                 # Main Gradio application
├── requirements.txt       # Dependencies for Spaces
├── README.md             # Hugging Face Spaces card
├── examples/             # Sample images for demo
│   ├── sample_dr_positive.jpg
│   ├── sample_dr_negative.jpg
│   └── sample_borderline.jpg
└── models/               # LoRA adapters (or load from HF)
    ├── stage1_adapter/
    └── stage2_adapter/
```

### Core Components

```python
# app.py skeleton
import gradio as gr
from src.models import DRDetector, SensitivityPreset
from src.pipeline import ReportGenerator

def analyze_retina(
    image,
    diabetes_status,
    hba1c,
    years_dm,
    cgm_avg_glucose,
    cgm_tir,
    sensitivity_preset
):
    """Main analysis function."""

    # 1. Load image
    if image is None:
        return None, "Please upload an image"

    # 2. Configure detector
    preset = SensitivityPreset[sensitivity_preset.upper()]
    detector = DRDetector(sensitivity=preset)

    # 3. Analyze retinal image
    dr_result = detector.detect(image)

    # 4. Build clinical context
    clinical_context = build_clinical_context(
        diabetes_status, hba1c, years_dm
    )
    cgm_context = build_cgm_context(cgm_avg_glucose, cgm_tir)

    # 5. Generate report
    generator = ReportGenerator()
    report = generator.generate(
        p_dr=dr_result.p_dr,
        dr_grade=dr_result.predicted_grade,
        urgency=dr_result.urgency_level,
        clinical_context=clinical_context,
        cgm_context=cgm_context,
    )

    # 6. Format output
    analysis_summary = format_analysis(dr_result)

    return analysis_summary, report

# Gradio interface
with gr.Blocks(title="DR Screening Assistant") as demo:
    gr.Markdown("# Diabetic Retinopathy Screening Assistant")
    gr.Markdown("Upload a retinal image to receive a patient-friendly health report.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Retinal Image", type="pil")

            with gr.Accordion("Patient Context (Optional)", open=False):
                diabetes_status = gr.Dropdown(
                    ["Type 2 Diabetes", "Type 1 Diabetes", "Pre-diabetes", "None/Unknown"],
                    label="Diabetes Status"
                )
                hba1c = gr.Number(label="HbA1c (%)", value=7.0)
                years_dm = gr.Number(label="Years with Diabetes", value=5)
                cgm_avg = gr.Number(label="CGM Avg Glucose (mg/dL)", value=None)
                cgm_tir = gr.Number(label="Time in Range (%)", value=None)

            sensitivity = gr.Radio(
                ["Screening", "High", "Balanced", "Specific"],
                label="Detection Sensitivity",
                value="High"
            )

            analyze_btn = gr.Button("Generate Report", variant="primary")

        with gr.Column():
            analysis_output = gr.Markdown(label="Analysis")
            report_output = gr.Markdown(label="Patient Report")

    analyze_btn.click(
        analyze_retina,
        inputs=[image_input, diabetes_status, hba1c, years_dm, cgm_avg, cgm_tir, sensitivity],
        outputs=[analysis_output, report_output]
    )

    gr.Examples(
        examples=[
            ["examples/sample_dr_negative.jpg", "Type 2 Diabetes", 6.8, 3, None, None, "High"],
            ["examples/sample_dr_positive.jpg", "Type 2 Diabetes", 8.5, 12, 165, 55, "High"],
        ],
        inputs=[image_input, diabetes_status, hba1c, years_dm, cgm_avg, cgm_tir, sensitivity],
    )

demo.launch()
```

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended)

**Pros:**
- Free GPU (T4 or A10G for approved models)
- Native Gradio support
- Easy model loading from HF Hub
- Public URL immediately

**Requirements:**
- Upload LoRA adapters to HF Hub
- Create Space with `sdk: gradio`
- Request GPU if needed (may take 24-48h)

**Steps:**
```bash
# 1. Push adapters to HF Hub
huggingface-cli upload drdavidl/medgemma-dr-lora outputs/medgemma-stage1/adapter
huggingface-cli upload drdavidl/medgemma-report-lora outputs/medgemma-stage2-probabilistic/adapter

# 2. Create Space
huggingface-cli repo create --type space --space-sdk gradio dr-screening-assistant

# 3. Push app code
cd app && git init && git remote add origin https://huggingface.co/spaces/drdavidl/dr-screening-assistant
git add . && git commit -m "Initial commit" && git push
```

### Option 2: Railway/Render (Backup)

**Pros:**
- More control over environment
- Better for CPU-only if no GPU approval

**Cons:**
- Costs money for sustained usage
- More setup required

### Option 3: Local Demo for Video

**Pros:**
- Full control
- No deployment delays
- M4 Mac has sufficient power

**Process:**
- Run Gradio locally with `share=True` for temporary public URL
- Record 3-min video showing complete flow

## Sample Images

Need 3-5 example retinal images for demo:

| Example | DR Status | P(DR) Range | Purpose |
|---------|-----------|-------------|---------|
| Healthy eye | DR- | < 0.1 | Show normal result |
| Mild DR | DR+ | 0.4-0.6 | Show moderate finding |
| Clear DR | DR+ | 0.7-0.9 | Show urgent finding |
| Borderline | DR- | 0.2-0.3 | Show uncertainty handling |

**Note:** Must use publicly available images or synthetic examples. Cannot use AI-READI data in public demo.

### Public Retinal Image Sources
- [APTOS 2019 Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) - Public DR dataset
- [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) - Large public dataset
- [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) - Indian DR dataset

## Video Script Outline (3 minutes)

### 0:00-0:30 - Problem Statement
- 1 in 3 diabetics develop DR
- Early detection prevents vision loss
- Current screening requires specialist visit
- Health literacy barrier in understanding results

### 0:30-1:15 - Solution Demo
- Show image upload
- Show clinical context entry
- Click "Generate Report"
- Highlight: natural frequency explanation

### 1:15-1:45 - Technical Innovation
- MedGemma fine-tuning approach
- Stage 1: Visual understanding
- Stage 2: Probabilistic communication
- "The model learned to SEE DR, not just explain labels"

### 1:45-2:15 - Sensitivity Configuration
- Show preset options
- Adjust slider
- Explain clinical implications
- "Healthcare orgs can configure for their context"

### 2:15-2:45 - Impact Potential
- Smartphone-based screening
- Underserved populations
- Primary care integration
- Reduce specialist bottleneck

### 2:45-3:00 - Closing
- Summary of key innovations
- Thank judges
- Links to code/demo

## Timeline

| Task | Duration | Priority |
|------|----------|----------|
| Create basic Gradio app | 2 hrs | High |
| Add sensitivity controls | 1 hr | High |
| Find/prepare sample images | 1 hr | High |
| Deploy to HF Spaces | 1-2 hrs | Medium |
| Test and refine | 1 hr | Medium |
| Record video | 2-3 hrs | High |
| Write technical overview | 2 hrs | High |

## Deliverables Checklist

- [ ] Working Gradio demo (local or hosted)
- [ ] Public URL for judges to try
- [ ] 3-minute video demonstration
- [ ] 3-page technical writeup
- [ ] GitHub repo with reproducible code
- [ ] Sample images for demo
- [ ] LoRA adapters uploaded to HF Hub
