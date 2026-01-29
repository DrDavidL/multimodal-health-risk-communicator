# Demo Examples

Sample retinal fundus images and synthetic clinical data for testing the DR Screening Assistant.

## Important Disclaimer

**Images:** These fundus images are from the **JSIEC 1000 Fundus Images Dataset** (Joint Shantou International Eye Centre). They are used for demonstration purposes only and should NOT be used for clinical assessment.

**Citation Required:** If using these images, please cite:
> Li, N., et al. "Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks." *Nature Communications* 12, 4828 (2021). https://www.nature.com/articles/s41467-021-25138-w

**Clinical Data:** All clinical values in `synthetic_examples.json` are **synthetic (fictional)** and were created for demonstration purposes only. They do NOT represent real patient data.

**Do NOT use AI-READI data** in this public demo — that data is protected under a Data Use Agreement.

## Files

| File | Description | Source |
|------|-------------|--------|
| `fundus_normal.jpg` | Normal fundus | JSIEC Dataset (Category 0.0) |
| `fundus_dr_mild.jpg` | Mild NPDR (DR1) | JSIEC Dataset (Category 0.3) |
| `fundus_dr_severe.jpg` | Severe NPDR (DR3) | JSIEC Dataset (Category 1.1) |
| `synthetic_examples.json` | Fictional clinical + CGM data | Created for demo |

## Example Scenarios

### 1. Healthy Adult (Low Risk)
- **Image:** `fundus_normal.jpg`
- **HbA1c:** 5.9% (pre-diabetic)
- **CGM Time in Range:** 94.5%
- **Expected:** Routine follow-up

### 2. Diabetic with Mild Changes (Moderate Risk)
- **Image:** `fundus_dr_mild.jpg`
- **HbA1c:** 7.8% (diabetic range)
- **CGM Time in Range:** 68.2%
- **Expected:** Eye exam within 1-2 months

### 3. Diabetic with Advanced Changes (High Risk)
- **Image:** `fundus_dr_severe.jpg`
- **HbA1c:** 9.2% (poor control)
- **CGM Time in Range:** 42.1%
- **Expected:** URGENT referral within 1-2 weeks

## Usage in Demo

These examples can be loaded automatically in the Gradio demo to test the full pipeline without needing real patient data.

## License

- **Images:** Open Database License (JSIEC Dataset) - requires citation
- **Synthetic data:** MIT License (part of this repository)
