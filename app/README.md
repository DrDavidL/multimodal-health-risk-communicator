---
title: DR Screening Assistant
emoji: '👁️'
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: demo.py
pinned: false
license: mit
---

# Diabetic Retinopathy Screening Assistant

A patient-friendly AI tool that analyzes retinal images and generates probabilistic health reports using MedGemma.

## Features

- **High-sensitivity DR detection** using community fine-tuned MedGemma LoRA
- **Patient-friendly reports** with natural frequency explanations ("7 out of 10 people")
- **Configurable sensitivity** for different clinical contexts
- **Interactive Q&A agent** for diabetes lifestyle questions
- **Edge AI ready** - runs on local hardware without cloud dependencies

## How It Works

1. **Upload** a retinal fundus image
2. **Add context** (optional): diabetes type, HbA1c, CGM data
3. **Get results**: probabilistic DR assessment with urgency level
4. **Read report**: patient-friendly explanation with next steps
5. **Ask questions**: interactive Q&A about diabetes lifestyle

## Technical Details

| Component | Model | Purpose |
|-----------|-------|---------|
| DR Detection | MedGemma 4B + LoRA | Identify diabetic retinopathy |
| Report Generation | MedGemma 4B + Novel LoRA | Generate health-literate reports |
| Lifestyle Q&A | MedGemma (grounded) | Answer patient questions |

## Credits

- **DR LoRA**: [qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)
- **Base Model**: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- **Training Data**: AI-READI (Bridge2AI program)

## Disclaimer

This tool is for educational and demonstration purposes only. It is not intended to diagnose, treat, or prevent any disease. Always consult qualified healthcare providers for medical advice.

---

**MedGemma Impact Challenge Submission** | David Liebovitz, MD
