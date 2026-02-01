# Multimodal Health Risk Communicator

**MedGemma Impact Challenge Submission**

---

## Project Name

**Multimodal Health Risk Communicator: Probabilistic DR Screening with Patient-Friendly Reports**

---

## Your Team

| Name | Specialty | Role |
|------|-----------|------|
| David Liebovitz, MD | Internal Medicine Physician, Northwestern University Feinberg School of Medicine | Solo developer: problem identification, architecture design, data pipeline, model fine-tuning, evaluation, and UI development |

---

## Problem Statement

**The Health Literacy Gap in Diabetic Retinopathy Screening**

Diabetic retinopathy (DR) affects 1 in 3 people with diabetes and is the leading cause of blindness in working-age adults. While early screening can prevent 95% of vision loss, screening rates remain below 60% in the US.

A critical barrier is health communication. Current results are delivered as technical grades ("moderate NPDR") or percentages that patients struggle to interpret. Research shows 36% of US adults have basic or below-basic health literacy, and patients misinterpret probabilities by ~50% when presented as percentages versus natural frequencies.

**Impact Potential:** This tool bridges the communication gap by generating patient-friendly reports that explain screening results using evidence-based risk communication (natural frequencies like "2 out of 10 people with similar results..."). It integrates retinal imaging with CGM glucose patterns and clinical data to provide personalized, actionable health guidance—democratizing access to specialist-level interpretation.

---

## Overall Solution

**Effective Use of MedGemma (HAI-DEF Model)**

We designed a dual-adapter architecture where two LoRA adapters share one MedGemma 4B base model:

| Component | Adapter | Function |
|-----------|---------|----------|
| DR Detection | Community LoRA ([qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)) | Retinal image → DR probability grades |
| Report Generation | Our Novel LoRA ([dliebovi/medgemma-stage2-report](https://huggingface.co/dliebovi/medgemma-stage2-report)) | Probabilities + clinical context → Patient-friendly report |
| Lifestyle Q&A | Same Stage 2 LoRA | Interactive guidance grounded on ADA/NEI/AAO guidelines |

**Key Innovation:** We fine-tuned MedGemma specifically to communicate risk using natural frequencies rather than confusing percentages, trained on 43 AI-READI participants with GPT-5.2-generated gold-standard reports as training targets.

**Evaluation Results (7 held-out participants, GPT-5.2 as judge):**

| Metric | Fine-tuned | Base MedGemma | Δ |
|--------|------------|---------------|---|
| Probability communication | 4.3/5 | 3.3/5 | +1.0 |
| Actionability | 3.7/5 | 2.6/5 | +1.1 |
| Overall | 3.9/5 | 3.5/5 | +0.4 |

---

## Technical Details

**Product Feasibility**

- **Edge AI Ready:** Full pipeline runs on Apple M4 Mac (18GB VRAM) without cloud dependencies—suitable for offline clinic deployment
- **Multimodal Integration:** Combines retinal fundus images, CGM glucose data (Open mHealth JSON), and clinical measurements (OMOP CDM) via unified data loaders
- **Privacy Compliant:** LoRA adapters contain <1% of parameters, anonymized IDs, 2% value jittering—no patient data in released weights (AI-READI DUA Section 3.D compliant)
- **Configurable Sensitivity:** Healthcare organizations can adjust DR detection thresholds for their clinical context

**Architecture:**
```
Retinal Image → DR LoRA → P(DR grades)
                              ↓
Clinical + CGM data ────→ Stage 2 LoRA → Patient Report + Q&A
```

**Public Artifacts:**
- Model: [huggingface.co/dliebovi/medgemma-stage2-report](https://huggingface.co/dliebovi/medgemma-stage2-report)
- Demo: [huggingface.co/spaces/dliebovi/dr-screening-assistant](https://huggingface.co/spaces/dliebovi/dr-screening-assistant)
- Code: [github.com/DrDavidL/multimodal-health-risk-communicator](https://github.com/DrDavidL/multimodal-health-risk-communicator)

**Dataset:** AI-READI (Bridge2AI Program) — 2,280 participants with retinal images, CGM, and clinical data.

**References:**
- AI-READI Consortium. (2024). "AI-READI: rethinking data collection, preparation and sharing for propelling AI-based discoveries in diabetes research and beyond." *Nature Metabolism*. https://doi.org/10.1038/s42255-024-01165-x
- AI-READI Consortium. (2025). Flagship Dataset of Type 2 Diabetes from the AI-READI Project (3.0.0) [Data set]. FAIRhub. https://doi.org/10.60775/fairhub.3

---

*This submission demonstrates that fine-tuned MedGemma can bridge the health literacy gap in diabetic retinopathy screening through probabilistic communication, while running efficiently on edge hardware.*
