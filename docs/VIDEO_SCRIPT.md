# Video Demo Script (3 minutes)

**Format:** Screen recording with picture-in-picture (you at bottom), voiceover narration

---

## Timing Overview

| Segment | Duration | Cumulative |
|---------|----------|------------|
| Hook + Problem | 25s | 0:25 |
| Solution Overview | 20s | 0:45 |
| Architecture | 20s | 1:05 |
| Live Demo | 70s | 2:15 |
| Evaluation Results | 25s | 2:40 |
| Edge AI + Closing | 20s | 3:00 |

---

## Script

### [0:00-0:25] Hook + Problem Statement
**Visual:** Title slide → Statistics graphic

> "37 million Americans have diabetes, and diabetic retinopathy is the leading cause of blindness in working-age adults. But here's the problem: only 60% of diabetic patients get their recommended annual eye screening.
>
> And even when they do get screened, the results often come back in clinical jargon that patients can't understand or act on.
>
> What if AI could not only screen for diabetic retinopathy, but communicate the results in language patients actually understand?"

---

### [0:25-0:45] Solution Overview
**Visual:** Project title slide → High-level diagram

> "I built a Multimodal Health Risk Communicator using MedGemma 4B.
>
> It takes a retinal fundus image, combines it with CGM data and clinical labs, and generates a patient-friendly report that communicates risk using natural frequencies — like 'about 4 out of 10 people' — instead of percentages that are hard to interpret."

---

### [0:45-1:05] Architecture — Dual Adapter Innovation
**Visual:** Architecture diagram showing adapter swapping

> "The key innovation is a dual-adapter architecture on a single MedGemma base.
>
> Model 1 is a community-trained LoRA that detects diabetic retinopathy from the retinal image, outputting grade probabilities.
>
> Model 2 is my novel LoRA, fine-tuned on GPT-5.2 generated training targets, that takes those probabilities plus clinical context and generates the patient report.
>
> Both adapters share the same 8-gigabyte base model — we just swap adapters at inference time. This means the whole pipeline fits on a single T4 GPU or even runs locally on a Mac."

---

### [1:05-2:15] Live Demo (70 seconds)
**Visual:** Screen recording of Gradio app

> "Let me show you how it works."

**[1:05-1:20] Upload Image**
> "I'll upload a retinal fundus image. This one shows some diabetic changes. The model processes the image..."

**[1:20-1:40] Show DR Detection Results**
> "Model 1 detected diabetic retinopathy with a probability of about 58%. Based on this, the urgency level is 'moderate' — meaning the patient should schedule an eye exam within one to two months, not urgent but not routine either."

**[1:40-2:00] Show Generated Report**
> "Now Model 2 generates the patient report. Notice how it uses natural frequencies: 'If we screened 10 people with similar results, about 6 would have diabetic retinopathy and about 4 would not.'
>
> It clearly states this is a screening, not a diagnosis. It connects eye health to the patient's glucose control. And it gives specific, actionable next steps."

**[2:00-2:15] Interactive Q&A Demo**
> "Patients can also ask follow-up questions. If I ask 'What foods should I avoid to protect my eyes?', the system provides grounded guidance from ADA and NEI clinical guidelines — without hallucinating medical advice."

---

### [2:15-2:40] Evaluation Results
**Visual:** Results table on screen

> "I evaluated the fine-tuned model against base MedGemma on 7 held-out test participants, using GPT-5.2 as an automated judge.
>
> The fine-tuned model scored 3.9 out of 5 overall, compared to 3.5 for the base model.
>
> The biggest improvements were in probability communication — plus 1 point — and actionability — plus 1.1 points. These are exactly the skills the training targeted."

---

### [2:40-3:00] Edge AI + Closing
**Visual:** Show terminal running locally → Final slide

> "One more thing: this entire pipeline runs locally on my M4 Mac with 18 gigabytes of memory. No cloud required. That's critical for clinical environments where patient data can't leave the premises.
>
> The Multimodal Health Risk Communicator shows how MedGemma can not just detect disease, but communicate results in a way patients can understand and act on.
>
> Thank you."

---

## Production Notes

### Visuals to Prepare

1. **Title slide:** "Multimodal Health Risk Communicator — MedGemma Impact Challenge"
2. **Statistics graphic:** "37M Americans with diabetes, 60% screening rate"
3. **Architecture diagram:** Dual-adapter flowchart (from README)
4. **Results table:** Fine-tuned vs Base comparison
5. **Final slide:** Links to GitHub, HF Space, contact

### Screen Recording Tips

- Use Gradio app at `http://localhost:7860`
- Pre-load the "Diabetic with Mild Changes" example
- Have clinical values already filled in
- Keep terminal visible for edge AI segment

### Tone

- Conversational but professional
- Enthusiastic but not overselling
- Emphasize patient benefit, not just technical achievement

---

## Backup: If Demo Fails

If the live demo has issues, use pre-recorded clips:
- Screenshot of DR detection results
- Screenshot of generated report
- Screenshot of Q&A response

Say: "Due to time constraints, I'm showing a pre-recorded example..."
