# TODO

## Immediate Priority (This Week)

### 1. Data Exploration
- [ ] Download and inspect `dataset_structure_description.json` — understand folder hierarchy
- [ ] Download and inspect `participants.tsv` — understand linkage keys and available subjects
- [ ] Sample 3-5 participants with complete data across all modalities
- [ ] Document actual file formats found in each modality folder

### 2. CGM Loader
- [ ] Inspect `wearable_blood_glucose/` folder structure
- [ ] Identify CGM file format (likely Dexcom CSV with timestamp, glucose columns)
- [ ] Build `CGMLoader` class:
  - Load raw data
  - Parse timestamps
  - Calculate summary statistics (mean, std, time-in-range, GMI)
  - Handle gaps in data

### 3. Clinical Data Loader
- [ ] Inspect `clinical_data/` folder structure
- [ ] Identify relevant variables (HbA1c, BMI, blood pressure, medications, etc.)
- [ ] Build `ClinicalLoader` class:
  - Load participant clinical data
  - Extract key diabetes-relevant metrics
  - Handle missing values

### 4. Validate DICOM Loader
- [ ] Test existing DICOM loader against actual AI-READI images
- [ ] Confirm folder structure assumptions (per-participant, per-eye, per-session?)
- [ ] Verify color space handling works correctly
- [ ] Add handling for multiple images per participant (if present)

---

## Next Phase (Week 2)

### 5. Multimodal Fusion
- [ ] Design prompt structure for combined modalities
- [ ] Implement `MultimodalFusion` class
- [ ] Test with 5-10 participants end-to-end

### 6. Report Generation
- [ ] Design patient-friendly output format
- [ ] Create prompt templates for different risk profiles
- [ ] Implement `ReportGenerator` class

### 7. Evaluation
- [ ] Define evaluation metrics (clinical accuracy, readability, completeness)
- [ ] Create small validation set with expected outputs
- [ ] Implement automated evaluation where possible

---

## Before Submission

- [ ] Write clear documentation for judges
- [ ] Create demo notebook with sample outputs
- [ ] Record demo video (if required)
- [ ] Ensure all code runs without data (graceful errors)
- [ ] Final README polish
- [ ] Make repository public (check challenge rules)
- [ ] Submit via challenge portal

---

## Technical Debt / Nice-to-Have

- [ ] Add unit tests for all loaders
- [ ] Add integration test with mock data
- [ ] Dockerize for reproducibility
- [ ] Add logging throughout pipeline
- [ ] Create config schema validation
- [ ] Performance profiling for inference

---

## Blockers / Questions

| Item | Status | Notes |
|------|--------|-------|
| Data format uncertainty | 🔄 Pending inspection | Need to download metadata files |
| Challenge evaluation criteria | ❓ Unknown | Review challenge documentation |
| Multi-image handling | ❓ Unknown | May have multiple fundus images per participant |

---

## Completed

- [x] Dataset access approved
- [x] Azure storage configured
- [x] Data transferred to container
- [x] MedGemma 4B running locally
- [x] DICOM loader with color space conversion
- [x] Prompt formatting for MedGemma resolved
