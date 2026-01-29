# Demo Examples

Sample retinal fundus images and synthetic clinical data for testing the DR Screening Assistant.

## Important Disclaimer

**Images:** These retinal images are from the [National Eye Institute (NEI)](https://www.nei.nih.gov/), part of the U.S. National Institutes of Health, and are in the public domain.

**Clinical Data:** All clinical values in `synthetic_examples.json` are **synthetic (fictional)** and were created for demonstration purposes only. They do NOT represent real patient data.

**Do NOT use AI-READI data** in this public demo — that data is protected under a Data Use Agreement.

## Files

| File | Description | Source |
|------|-------------|--------|
| `fundus_normal.jpg` | Normal retina | NEI public domain |
| `fundus_dr_mild.jpg` | Mild diabetic retinopathy | NEI public domain |
| `fundus_dr_severe.jpg` | Proliferative diabetic retinopathy | NEI public domain |
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

- **Images:** Public domain (U.S. government work)
- **Synthetic data:** MIT License (part of this repository)
