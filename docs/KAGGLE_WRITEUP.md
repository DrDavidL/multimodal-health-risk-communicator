# Multimodal Health Risk Communicator: Probabilistic DR Screening with Patient-Friendly Reports

**MedGemma Impact Challenge Submission**

David Liebovitz, MD | Northwestern University Feinberg School of Medicine

---

## 1. Problem Statement

### The Health Literacy Gap in Diabetic Retinopathy

Diabetic retinopathy (DR) affects 1 in 3 people with diabetes and is the leading cause of blindness in working-age adults. Early detection through retinal screening can prevent 95% of vision loss, yet screening rates remain below 60% in the US.

A critical barrier is **health communication**. Current screening results are delivered as technical grades (e.g., "moderate NPDR") or percentages that patients struggle to interpret. Research shows that:

- 36% of US adults have basic or below-basic health literacy (NCES)
- Patients misinterpret probabilities by an average of 50% when presented as percentages vs. natural frequencies (Hoffrage & Gigerenzer, 1998)
- Anxiety from poorly communicated results leads to screening avoidance

### Our Approach

We developed a **Multimodal Health Risk Communicator** that:

1. Analyzes retinal fundus images using MedGemma with a community-trained DR detection LoRA
2. Generates patient-friendly reports using our novel **Stage 2 LoRA adapter** trained specifically for probabilistic health communication
3. Provides an interactive Q&A agent for lifestyle guidance grounded on ADA/NEI/AAO guidelines

**Key Innovation:** Teaching MedGemma to communicate risk using **natural frequencies** ("About 2 out of 10 people with similar results...") rather than confusing percentages.

---

## 2. Technical Approach

### Dual-Adapter Architecture

We designed an efficient architecture where **two LoRA adapters share one MedGemma 4B base model**:

| Stage | Adapter | Task | Input |
|-------|---------|------|-------|
| **DR Detection** | Community DR LoRA | Image → P(DR) grade probabilities | Retinal fundus image |
| **Report Generation** | Our Novel LoRA | P(DR) + context → Patient report | Structured clinical data |
| **Lifestyle Q&A** | Same Stage 2 LoRA | Question → Grounded answer | Patient context + question |

This enables **efficient edge deployment** on a single GPU or Apple Silicon Mac—both adapters are swapped at inference time on the same base model.

### Stage 2 LoRA Training

Our probabilistic report generator was trained on 43 AI-READI participants:

| Parameter | Value |
|-----------|-------|
| Base model | `google/medgemma-4b-it` |
| PEFT method | LoRA (rank=16, alpha=32) |
| Trainable params | 0.76% of base |
| Target modules | All attention projections |
| Training targets | GPT-5.2 generated reports with natural frequency framing |
| Hardware | Apple M4 (64GB), MPS backend |

**Training Signal:** We used GPT-5.2 to generate gold-standard patient reports incorporating probabilistic communication best practices. MedGemma learned to reproduce this communication style.

### Privacy Safeguards (DUA Compliance)

Per AI-READI Data Use Agreement Section 3.D:

- **No patient data in adapters:** LoRA adapters contain <1% of base model parameters, minimizing memorization risk
- **Anonymized IDs:** Person IDs replaced with P1000-P1054
- **Value jittering:** 2% noise added to all numeric values during training
- **No raw data released:** Only adapter weights published

---

## 3. Evaluation Results

### Automated Evaluation (GPT-5.2 as Judge)

We evaluated on 7 held-out test participants using GPT-5.2 to score reports on a 1-5 scale:

| Criterion | Fine-tuned | Base MedGemma | Improvement |
|-----------|------------|---------------|-------------|
| Probability communication | **4.3** | 3.3 | +1.0 |
| Actionability | **3.7** | 2.6 | +1.1 |
| Clinical accuracy | 3.0 | 3.0 | 0 |
| Readability | 4.1 | 4.1 | 0 |
| Completeness | **5.0** | 4.9 | +0.1 |
| **Overall mean** | **3.9/5** | **3.5/5** | **+0.4** |

**Key Finding:** Fine-tuning significantly improved probability communication (+1.0) and actionability (+1.1) while maintaining readability and accuracy.

### DR Detection Performance

Using the community DR LoRA adapter:

| Metric | Value |
|--------|-------|
| Sensitivity (at threshold 0.05) | 100% (1/1 DR+ detected) |
| Specificity | 83% (5/6 DR- correct) |
| Calibration | Good for routine/urgent cases |

### Report Quality Highlights

From GPT-5.2 judge feedback:

> **Strengths:** "Clear natural-frequency risk explanation, repeatedly clarifies screening vs diagnosis, includes all required sections, ties glucose patterns to eye risk"

> **Weaknesses:** "Minor numerical rounding issues in natural frequency conversion for very low probabilities"

---

## 4. Differentiated Capabilities

### Why This Matters for Healthcare

1. **Configurable Sensitivity Thresholds:** Healthcare organizations can adjust detection sensitivity for their clinical context (community screening vs. specialist triage)

2. **Multimodal Context Integration:** Reports incorporate CGM glucose patterns, HbA1c, and clinical risk factors—not just image findings

3. **Edge AI Ready:** Full pipeline runs on Apple M4 Mac (18GB model) without cloud dependencies, enabling offline clinic deployment

4. **Compositional AI:** Demonstrates adapter chaining—detection, communication, and Q&A on one lightweight base model

### Bonus Prize Alignment

| Category | Alignment |
|----------|-----------|
| **Novel Fine-Tuned Adaptations** | First model specifically trained for natural frequency DR communication |
| **Edge AI Deployment** | Runs entirely on Apple Silicon; designed for clinic workstations |
| **Agentic Workflows** | 3-stage pipeline with interactive Q&A agent |

---

## 5. Resources

### Public Artifacts

- **Model:** [huggingface.co/dliebovi/medgemma-stage2-report](https://huggingface.co/dliebovi/medgemma-stage2-report)
- **Demo:** [huggingface.co/spaces/dliebovi/dr-screening-assistant](https://huggingface.co/spaces/dliebovi/dr-screening-assistant)
- **Code:** [github.com/drdavidl/multimodal-health-risk-communicator](https://github.com/drdavidl/multimodal-health-risk-communicator)
- **Video:** [3-minute demo video] *(to be linked)*

### Dataset

[AI-READI](https://aireadi.org/) (Bridge2AI Program): 2,280 participants with retinal images, CGM data, and clinical measurements. Used under Data Use Agreement.

### Acknowledgments

- **Google Health AI:** MedGemma base model
- **Community:** [qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy) DR detection adapter
- **AI-READI/Bridge2AI:** Dataset access
- **NEI/NIH:** Public domain fundus images for demo

---

## 6. Limitations & Future Work

### Current Limitations

1. **Small test set:** 7 held-out participants limits statistical power
2. **Single-site data:** AI-READI participants may not represent all populations
3. **Natural frequency rounding:** Very low probabilities (1-5%) sometimes rounded imprecisely

### Future Directions

1. **Stage 1 end-to-end training:** Train DR detection directly on image-to-probability, bypassing discrete grades
2. **Multilingual reports:** Extend to Spanish, Chinese for underserved communities
3. **Mobile deployment:** Optimize for smartphone-attached fundus cameras
4. **Clinical validation:** Prospective study comparing patient understanding with vs. without tool

---

*This submission demonstrates that fine-tuned MedGemma can bridge the health literacy gap in diabetic retinopathy screening through probabilistic communication, while running efficiently on edge hardware.*
