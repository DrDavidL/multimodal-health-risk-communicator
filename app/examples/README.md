# Sample Retinal Images for Demo

This directory should contain sample retinal fundus images for the demo application.

## Obtaining Sample Images

### Option 1: APTOS 2019 Dataset (Recommended)
The APTOS 2019 Blindness Detection dataset contains publicly available retinal images with DR grades.

1. Visit: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
2. Accept the competition rules
3. Download a few sample images from the training set
4. Rename them according to our convention below

### Option 2: Academic Torrents
Alternative download without Kaggle account:
- https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd

## Required Files

Please add the following sample images:

| Filename | DR Grade | Purpose |
|----------|----------|---------|
| `sample_healthy.jpg` | Grade 0 (No DR) | Show normal/healthy result |
| `sample_mild.jpg` | Grade 1-2 (Mild/Moderate) | Show moderate finding |
| `sample_severe.jpg` | Grade 3-4 (Severe/PDR) | Show urgent finding |

## DR Grade Reference

The APTOS dataset uses this grading scale:
- **0**: No diabetic retinopathy
- **1**: Mild nonproliferative DR
- **2**: Moderate nonproliferative DR
- **3**: Severe nonproliferative DR
- **4**: Proliferative DR

## Image Selection Tips

When selecting sample images:
1. Choose clear, well-lit fundus photographs
2. Select one from each severity category
3. Avoid images with artifacts or poor quality
4. Ensure images are appropriate for public demonstration

## License

APTOS 2019 dataset images are released under the competition rules for research and educational purposes.

## Note

**Do NOT use AI-READI data** in this public demo - that data is protected under a Data Use Agreement.
