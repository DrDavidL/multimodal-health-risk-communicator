---
title: DR Screening Assistant
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: demo.py
pinned: false
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
tags:
  - medgemma
  - health
  - diabetes
  - diabetic-retinopathy
  - patient-communication
  - medgemma-impact-challenge
---

# Diabetic Retinopathy Screening Assistant

AI-powered screening tool that analyzes retinal fundus images and generates patient-friendly probabilistic health reports using a dual-adapter MedGemma pipeline.

## Features

1. **DR Detection** — Community-trained LoRA identifies diabetic retinopathy from fundus images
2. **Report Generation** — Novel LoRA generates health-literate probabilistic reports
3. **Lifestyle Q&A** — Grounded agent answers diabetes management questions

All inference runs on a single MedGemma 4B model with efficient adapter swapping.

## Demo Examples

Sample retinal images and synthetic clinical data are provided for testing. The images are from the [National Eye Institute (NEI)](https://www.nei.nih.gov/) and are in the public domain.

## Architecture

| Stage | Adapter | Task |
|-------|---------|------|
| DR Detection | [qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy) | Image → P(DR) |
| Report Generation | [dliebovi/medgemma-stage2-report](https://huggingface.co/dliebovi/medgemma-stage2-report) | P(DR) + context → Report |

## Disclaimer

This tool is for educational and demonstration purposes only. It is not intended to diagnose, treat, or prevent any disease. Always consult qualified healthcare providers for medical advice.

---

**MedGemma Impact Challenge Submission** | David Liebovitz, MD

Part of [Multimodal Health Risk Communicator](https://github.com/dliebovi/multimodal-health-risk-communicator)
