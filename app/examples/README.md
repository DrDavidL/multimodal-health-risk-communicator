# Demo Examples

Sample retinal fundus images and synthetic clinical data for testing the DR Screening Assistant.

## Important Disclaimer

**Images:** These are **synthetic placeholder images** created programmatically for demonstration purposes. They are NOT real medical images and should NOT be used for any clinical assessment. They are designed to visually resemble fundus photographs but are entirely artificial.

**Clinical Data:** All clinical values in `synthetic_examples.json` are **synthetic (fictional)** and were created for demonstration purposes only. They do NOT represent real patient data.

**Do NOT use AI-READI data** in this public demo — that data is protected under a Data Use Agreement.

## Files

| File | Description | Source |
|------|-------------|--------|
| `fundus_normal.jpg` | Normal retina (synthetic) | Programmatically generated |
| `fundus_dr_mild.jpg` | Mild DR appearance (synthetic) | Programmatically generated |
| `fundus_dr_severe.jpg` | Severe DR appearance (synthetic) | Programmatically generated |
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

- **Images:** MIT License (synthetic, created for this demo)
- **Synthetic data:** MIT License (part of this repository)
