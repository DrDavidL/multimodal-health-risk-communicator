# Data Documentation

## Overview

This project uses the **AI-READI** (Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights) dataset, part of the NIH Bridge2AI program.

**Access**: Data is accessed via Azure Blob Storage under an approved Data Use Agreement. Data files are never committed to this repository.

## Storage Location

```
Azure Storage Account: nwaireadi2026
Container: aireadi-data
```

## Data Structure

Based on Azure Storage Explorer view (January 2026):

```
aireadi-data/
├── clinical_data/               # Clinical measurements and metadata
├── retinal_photography/         # DICOM fundus images
├── wearable_blood_glucose/      # CGM time-series data
├── CHANGELOG.md
├── dataset_description.json
├── dataset_structure_description.json   # ⭐ Key schema documentation
├── healthsheet.md
├── LICENSE.txt
├── participants.json            # ⭐ Subject linkage (JSON format)
├── participants.tsv             # ⭐ Subject linkage (tabular format)
├── README.md
└── study_description.json
```

## Modality Details

### 1. Retinal Photography

**Location**: `retinal_photography/`

**Format**: DICOM (.dcm)

**Expected structure** (to be confirmed):
```
retinal_photography/
├── sub-001/
│   ├── ses-01/
│   │   ├── OD/          # Right eye (Oculus Dexter)
│   │   │   └── fundus.dcm
│   │   └── OS/          # Left eye (Oculus Sinister)
│   │       └── fundus.dcm
│   └── ses-02/
│       └── ...
├── sub-002/
│   └── ...
```

**Technical notes**:
- Color space: Likely YBR_FULL_422 (requires conversion to RGB)
- Resolution: TBD
- Bit depth: Typically 8-bit color

**Loading code**: See `src/loaders/dicom_loader.py`

---

### 2. Continuous Glucose Monitoring (CGM)

**Location**: `wearable_blood_glucose/`

**Format**: Likely CSV (Dexcom export format)

**Expected columns** (to be confirmed):
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Measurement time |
| glucose | float | Glucose value (mg/dL) |
| trend | string | Arrow direction (optional) |
| device_id | string | Sensor identifier |

**Derived metrics to calculate**:
- Mean glucose
- Glucose standard deviation
- Time in range (TIR): 70-180 mg/dL
- Time below range (TBR): <70 mg/dL
- Time above range (TAR): >180 mg/dL
- Glucose Management Indicator (GMI)
- Coefficient of variation (CV)

**Loading code**: See `src/loaders/cgm_loader.py`

---

### 3. Clinical Data

**Location**: `clinical_data/`

**Format**: Likely JSON or CSV per participant

**Expected variables** (to be confirmed):
| Variable | Type | Description |
|----------|------|-------------|
| participant_id | string | Unique identifier |
| age | int | Age in years |
| sex | string | M/F/Other |
| diabetes_type | string | Type 2, Control |
| diabetes_duration | float | Years since diagnosis |
| hba1c | float | Most recent HbA1c (%) |
| bmi | float | Body mass index |
| systolic_bp | int | Systolic blood pressure |
| diastolic_bp | int | Diastolic blood pressure |
| medications | list | Current diabetes medications |
| comorbidities | list | Related conditions |

**Loading code**: See `src/loaders/clinical_loader.py`

---

## Participant Linkage

**Primary key**: `participant_id` (format: `sub-XXX`)

All three modalities link via participant_id. The `participants.tsv` file contains the master list with demographic information.

**Example** (hypothetical):
```tsv
participant_id	age	sex	diabetes_status
sub-001	58	F	Type 2
sub-002	62	M	Type 2
sub-003	55	M	Control
```

---

## Data Quality Considerations

### Missing Data
- Not all participants have all modalities
- Some CGM records may have gaps (sensor changes, signal loss)
- Clinical variables may be incomplete

### Handling Strategy
1. Identify participants with complete data for all three modalities
2. For partial data, generate reports for available modalities only
3. Clearly indicate data completeness in output

### Known Issues
- YBR_FULL_422 color space requires explicit conversion
- CGM timestamps may be in different timezones
- Clinical variable names may differ from OMOP/FHIR standards

---

## Data Access Commands

```bash
# List container contents
az storage blob list \
    --account-name nwaireadi2026 \
    --container-name aireadi-data \
    --output table

# Download metadata files
az storage blob download \
    --account-name nwaireadi2026 \
    --container-name aireadi-data \
    --name participants.tsv \
    --file ./data/participants.tsv

# Download structure documentation
az storage blob download \
    --account-name nwaireadi2026 \
    --container-name aireadi-data \
    --name dataset_structure_description.json \
    --file ./data/dataset_structure_description.json

# Download single participant's data (example)
az storage blob download-batch \
    --account-name nwaireadi2026 \
    --container-name aireadi-data \
    --destination ./data \
    --pattern "*/sub-001/*"
```

---

## References

- [AI-READI Documentation](https://docs.aireadi.org)
- [BIDS Standard](https://bids.neuroimaging.io) — AI-READI may follow BIDS-like conventions
- [DICOM Standard](https://www.dicomstandard.org)
- [Dexcom CGM Data Format](https://developer.dexcom.com)
