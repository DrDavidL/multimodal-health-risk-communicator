# Stage 3 Evaluation Plan

## Competition Criteria Alignment

Per the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge), submissions are evaluated on:

| Criterion | Weight | Our Focus |
|-----------|--------|-----------|
| **Effective use of HAI-DEF models** | High | MedGemma 4B with LoRA fine-tuning for both vision and text |
| **Importance of problem** | High | DR screening + health literacy gap = significant public health issue |
| **Potential real-world impact** | High | Smartphone-based screening → underserved populations |
| **Technical feasibility** | Medium | Running on M4 Mac locally, no cloud GPU required |
| **Execution & communication** | High | 3-min video, 3-page writeup, working demo |

## Bonus Prize Categories

The competition offers **additional prizes** beyond the main $75K pool. We should target:

| Bonus Category | Our Alignment | Strategy |
|----------------|---------------|----------|
| **Novel fine-tuned model adaptations** | **STRONG** | 2-stage LoRA: vision understanding + probabilistic report generation |
| **Effective edge AI deployment** | **STRONG** | Runs entirely on M4 Mac (no cloud GPU), designed for mobile/clinic deployment |
| **Agent-based workflows** | Partial | Could add: multi-step reasoning agent for complex cases |

### How We Qualify for Bonus Categories

#### 1. Novel Fine-Tuned Model Adaptations
- **Stage 1**: Leverage community-trained DR LoRA ([qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy))
  - Evaluated against our own fine-tuning; community model outperformed
  - Demonstrates good engineering judgment: use best available, contribute where novel
- **Stage 2**: Novel LoRA fine-tuning for probabilistic health communication
  - Trained on GPT-5.2 generated targets with natural frequency framing
  - First model specifically trained for health literacy-aware DR communication
- **Innovation**: Natural frequency communication ("7 out of 10") trained into model
- **Innovation**: Configurable sensitivity thresholds for different clinical contexts

#### 2. Edge AI Deployment
- **Local execution**: Full pipeline runs on Apple Silicon (M4 Mac, 18GB model)
- **No cloud dependency**: Works offline after initial model download
- **Target hardware**: Designed for clinic workstations, potentially mobile
- **Demo**: Show inference running locally in video

#### 3. Agent-Based Workflows (Strong Enhancement)
**Interactive Diabetes Lifestyle Q&A Agent:**

After receiving their report, patients can ask follow-up questions:
- "What foods should I avoid to protect my eyes?"
- "How does exercise affect my blood sugar and eye health?"
- "What symptoms should I watch for?"

**Implementation:**
- MedGemma grounded on authoritative sources (ADA guidelines, NIH, AAO)
- Context-aware: knows patient's P(DR), urgency, CGM patterns
- Safe: deflects to "ask your doctor" for clinical decisions

**Agent Workflow:**
```
1. DR Detection (Stage 1 LoRA)
        ↓
2. Report Generation (Stage 2 LoRA)
        ↓
3. Interactive Q&A (grounded MedGemma)
   └── Patient asks: "What can I do?"
   └── Agent retrieves relevant lifestyle guidance
   └── Personalizes based on their risk level
```

This demonstrates:
- Multi-step compositional AI
- Grounded generation (not hallucinating medical advice)
- Patient engagement beyond static reports

## Evaluation Methodology

### Primary Question

> Does fine-tuned MedGemma produce clinically appropriate, patient-friendly reports when inferring retinal findings from images, compared to GPT-5.2 given explicit findings?

### Comparison Setup

| Model | Retinal Input | DR Info | Clinical/CGM |
|-------|---------------|---------|--------------|
| **MedGemma (ours)** | Raw fundus image | Must infer P(DR) from image | Text summary |
| **GPT-5.2 (baseline)** | None | Provided as metadata | Text summary |

This tests whether MedGemma **learned to see** DR, not just explain pre-labeled findings.

## Evaluation Metrics

### 1. Clinical Accuracy (40%)

#### 1a. DR Detection Performance
Compare MedGemma's inferred P(DR) against ground truth:

```
Metrics:
- Sensitivity at default threshold (0.05)
- Specificity at default threshold
- AUC-ROC curve
- Calibration (predicted P(DR) vs actual DR rate)
```

#### 1b. Report Factual Accuracy
Human review of 10 randomly selected reports:

| Dimension | Scoring |
|-----------|---------|
| DR finding accuracy | 0-2 (wrong, partial, correct) |
| Urgency appropriateness | 0-2 (dangerous, acceptable, appropriate) |
| Clinical context integration | 0-2 (ignored, mentioned, integrated) |
| No hallucinated findings | 0-2 (fabrications, minor errors, accurate) |

### 2. Patient Accessibility (30%)

#### 2a. Readability Metrics (Automated)
```python
# Compute for each report:
- Flesch-Kincaid Grade Level (target: ≤ 8th grade)
- Flesch Reading Ease (target: ≥ 60)
- SMOG Index
- Average sentence length
- Medical jargon count
```

#### 2b. Probability Communication Quality
Check for best practices:

| Practice | Check |
|----------|-------|
| Natural frequencies used | "X out of 10" present? |
| Screening vs diagnosis distinction | Explicit caveat present? |
| Actionable recommendations | Specific next steps given? |
| Uncertainty acknowledged | Not overconfident in findings? |

#### 2c. Human Evaluation (if time permits)
Show 5 reports to 2-3 non-medical volunteers:
- "How confident do you feel you understand this?"
- "What would you do next after reading this?"
- "Rate clarity on 1-5 scale"

### 3. Technical Comparison (20%)

#### 3a. MedGemma vs GPT-5.2 Report Quality
For each test participant, compare:

| Metric | Measurement |
|--------|-------------|
| Information completeness | Coverage of relevant findings |
| Consistency with ground truth | Agreement on DR status |
| Actionability | Clarity of recommended actions |
| Tone appropriateness | Supportive without alarming |

#### 3b. Ablation: Image Contribution
Compare reports WITH vs WITHOUT retinal image to quantify visual understanding contribution.

### 4. Robustness (10%)

- Test on DR+ and DR- cases separately
- Test on borderline P(DR) cases (0.2-0.4)
- Verify no memorization (similarity to training targets < 5%)

## Test Set

### Participants
Use held-out test split from Stage 1/2 data preparation:
- ~7 test participants
- Known ground truth for DR, AMD, RVO
- Diversity of diabetes status and findings

### Data Requirements
For each test participant:
1. Retinal fundus image (mosaic preferred)
2. CGM metrics summary
3. Clinical measurements summary
4. Ground truth retinal findings (for evaluation only)

## Evaluation Pipeline

```bash
# Step 1: Generate MedGemma reports (no findings provided)
python scripts/evaluate_stage3.py --model medgemma --output outputs/evaluation/medgemma/

# Step 2: Generate GPT-5.2 baseline reports (findings provided)
python scripts/evaluate_stage3.py --model gpt5 --output outputs/evaluation/gpt5/

# Step 3: Compute metrics
python scripts/compute_metrics.py --medgemma outputs/evaluation/medgemma/ \
                                   --gpt5 outputs/evaluation/gpt5/ \
                                   --output outputs/evaluation/results.json

# Step 4: Generate evaluation report
python scripts/generate_eval_report.py --results outputs/evaluation/results.json
```

## Output Artifacts

### For Submission
1. `evaluation_results.json` - All metrics
2. `example_reports/` - 3-5 representative reports
3. `comparison_table.md` - MedGemma vs GPT-5.2 side-by-side

### For Video Demo
- Show one complete pipeline run
- Highlight probabilistic communication
- Show sensitivity slider in action
- Compare MedGemma vs GPT-5.2 output

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| DR Sensitivity | ≥ 80% | 100% |
| DR Specificity | ≥ 50% | ≥ 65% |
| Readability (FK Grade) | ≤ 10 | ≤ 8 |
| Natural freq. usage | 100% | 100% |
| Screening caveat | 100% | 100% |
| Human clarity rating | ≥ 3.5/5 | ≥ 4/5 |

## Timeline

| Task | Duration | Depends On |
|------|----------|------------|
| Stage 2 training | ~2.5 hrs | In progress |
| Implement evaluate_stage3.py | 1-2 hrs | Stage 2 done |
| Run evaluation | 1 hr | Script ready |
| Compute metrics | 30 min | Evaluation done |
| Generate report | 30 min | Metrics done |
| Create video | 2-3 hrs | Demo app ready |
