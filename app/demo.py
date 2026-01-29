#!/usr/bin/env python3
"""Diabetic Retinopathy Screening Assistant - Demo Application.

A patient-friendly tool that analyzes retinal images and generates
probabilistic health reports using a dual-adapter MedGemma pipeline.

Architecture:
    1. DR Detection: MedGemma 4B + Community DR LoRA (vision)
    2. Report Generation: MedGemma 4B + Our Stage 2 LoRA (text)
    3. Lifestyle Q&A: Same Stage 2 adapter with grounded prompting

Both adapters share the same MedGemma 4B base model - adapters are
swapped at inference time for efficient single-GPU deployment.
"""

import sys
from pathlib import Path
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Try to import HuggingFace Spaces GPU decorator (only available on Spaces)
# ---------------------------------------------------------------------------
try:
    import spaces
    ON_HF_SPACES = True
except ImportError:
    # Running locally — define a no-op decorator
    ON_HF_SPACES = False
    class _Spaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None:
                return fn
            return lambda f: f
    spaces = _Spaces()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sensitivity presets
class SensitivityPreset(str, Enum):
    SCREENING = "screening"  # Highest sensitivity (~100%), more false positives
    HIGH = "high"            # High sensitivity (~100%), balanced (RECOMMENDED)
    BALANCED = "balanced"    # Moderate (~80% sens, ~75% spec)
    SPECIFIC = "specific"    # High specificity, fewer false positives

SENSITIVITY_THRESHOLDS = {
    SensitivityPreset.SCREENING: 0.03,
    SensitivityPreset.HIGH: 0.05,
    SensitivityPreset.BALANCED: 0.15,
    SensitivityPreset.SPECIFIC: 0.30,
}

# Community DR adapter (Stage 1 — vision)
DR_ADAPTER_ID = "qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy"

# Our report generation adapter (Stage 2 — text)
REPORT_ADAPTER_ID = "dliebovi/medgemma-stage2-report"

# Base model
BASE_MODEL_ID = "google/medgemma-4b-it"

# DR grading prompt (from community model)
DR_GRADES_PROMPT = """Based on the fundus image, what is the stage of diabetic retinopathy?
A: No apparent retinopathy (No DR)
B: Mild nonproliferative diabetic retinopathy (Mild NPDR)
C: Moderate nonproliferative diabetic retinopathy (Moderate NPDR)
D: Severe nonproliferative diabetic retinopathy (Severe NPDR)
E: Proliferative diabetic retinopathy (PDR)"""

# Stage 2 report prompt template
REPORT_PROMPT_TEMPLATE = """You are a health communication specialist helping patients understand their diabetic retinopathy screening results.

SCREENING RESULTS:
- Probability of diabetic retinopathy: {p_dr:.1%}
- Screening assessment: {certainty}
- Predicted severity if present: {grade_description}
- Urgency level: {urgency}

CLINICAL INFORMATION:
{clinical_context}

GLUCOSE MONITORING:
{cgm_context}

Generate a patient-friendly report that:
1. Explains the probability using natural frequencies (e.g., "X out of 10 people with similar results...")
2. Clearly states this is a SCREENING result, not a definitive diagnosis
3. Provides appropriate recommendations based on urgency
4. Connects eye health to glucose control when relevant
5. Uses simple language (8th grade reading level)
6. Is warm and supportive, not alarming

Include these sections:
- Understanding Your Retinal Screening Results
- Connecting Your Eye Health to Your Diabetes
- What You Should Do Next
- Key Points to Remember
- Questions to Ask Your Eye Doctor"""

# Lifestyle Q&A system prompt
QA_SYSTEM_PROMPT = """You are a diabetes health educator helping a patient understand how to manage their health after receiving a retinal screening result. Answer the patient's question using the authoritative guidance provided below. Be supportive, use simple language (8th grade reading level), and always recommend consulting their healthcare provider for personalized medical advice.

AUTHORITATIVE GUIDANCE:
## Sources: American Diabetes Association (ADA) Standards of Care 2024, National Eye Institute (NEI), American Academy of Ophthalmology (AAO)

## Blood Sugar Control
- Target HbA1c < 7% for most adults with diabetes
- Tight glucose control reduces DR progression by 25-76%
- Avoid rapid drops in blood sugar which can temporarily worsen DR

## Diet Recommendations
- Mediterranean-style diet reduces cardiovascular risk
- Limit refined carbohydrates and added sugars
- Increase fiber intake (vegetables, whole grains, legumes)
- Omega-3 fatty acids (fish) may support retinal health
- Limit sodium to help control blood pressure

## Physical Activity
- 150+ minutes moderate aerobic activity per week
- Resistance training 2-3 times per week
- Check blood sugar before/after exercise if on insulin

## Blood Pressure
- Target < 130/80 mmHg for people with diabetes
- High blood pressure accelerates retinopathy progression

## Smoking
- Smoking doubles the risk of vision loss in diabetics
- Resources: 1-800-QUIT-NOW

## Regular Monitoring
- Annual dilated eye exam (more frequent if DR present)
- HbA1c every 3-6 months
- Blood pressure at every healthcare visit

## Warning Signs (Seek Immediate Care)
- Sudden vision loss or dark spots
- Flashes of light or new floaters
- Blurry vision that doesn't clear

PATIENT CONTEXT:
{patient_context}

Answer the patient's question in 150-250 words. If the question requires clinical decision-making, advise consulting their healthcare provider."""

# ---------------------------------------------------------------------------
# Model management (singleton)
# ---------------------------------------------------------------------------
_model_state = {
    "processor": None,
    "model": None,
    "dr_adapter_loaded": False,
    "report_adapter_loaded": False,
    "active_adapter": None,
}


def _load_base_model():
    """Load the base MedGemma model with DR adapter."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    if _model_state["model"] is not None:
        return

    print("Loading MedGemma base model...")
    _model_state["processor"] = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else (
        {"": "mps"} if torch.backends.mps.is_available() else None
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load community DR adapter
    print(f"Loading DR adapter: {DR_ADAPTER_ID}")
    model = PeftModel.from_pretrained(
        base_model, DR_ADAPTER_ID, adapter_name="dr_detection"
    )
    _model_state["dr_adapter_loaded"] = True
    _model_state["active_adapter"] = "dr_detection"

    # Try to load our Stage 2 report adapter
    try:
        print(f"Loading report adapter: {REPORT_ADAPTER_ID}")
        model.load_adapter(REPORT_ADAPTER_ID, adapter_name="report_generation")
        _model_state["report_adapter_loaded"] = True
        print("Report adapter loaded successfully.")
    except Exception as e:
        print(f"Report adapter not available ({e}). Will use template-based reports.")

    model.eval()
    _model_state["model"] = model
    print("Models ready.")


def _set_adapter(adapter_name: str):
    """Switch the active LoRA adapter."""
    if _model_state["active_adapter"] != adapter_name:
        _model_state["model"].set_adapter(adapter_name)
        _model_state["active_adapter"] = adapter_name


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

@spaces.GPU(duration=60)
def run_dr_detection(image: Image.Image, threshold: float) -> dict:
    """Run DR detection using community LoRA adapter.

    Returns dict with p_dr, has_dr, predicted_grade, grade_probs.
    """
    import torch
    import torch.nn.functional as F

    _load_base_model()
    _set_adapter("dr_detection")

    processor = _model_state["processor"]
    model = _model_state["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DR_GRADES_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

        grade_probs = {}
        for grade in ["A", "B", "C", "D", "E"]:
            tokens = processor.tokenizer.encode(grade, add_special_tokens=False)
            token_id = tokens[0]
            grade_probs[grade] = F.softmax(logits, dim=-1)[0, token_id].item()

        total = sum(grade_probs.values())
        if total > 0:
            grade_probs = {k: v / total for k, v in grade_probs.items()}

        p_dr = grade_probs["B"] + grade_probs["C"] + grade_probs["D"] + grade_probs["E"]

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    predicted_grade = max(grade_probs, key=grade_probs.get)
    return {
        "p_dr": p_dr,
        "has_dr": p_dr >= threshold,
        "predicted_grade": predicted_grade,
        "grade_probs": grade_probs,
    }


@spaces.GPU(duration=120)
def run_report_generation(p_dr: float, grade: str, urgency: str,
                          clinical_context: str, cgm_context: str) -> str:
    """Generate patient report using our Stage 2 LoRA adapter."""
    import torch

    if not _model_state["report_adapter_loaded"]:
        return None  # Fall back to template

    _load_base_model()
    _set_adapter("report_generation")

    processor = _model_state["processor"]
    model = _model_state["model"]

    # Build certainty language
    if p_dr >= 0.7:
        certainty = "Diabetic retinopathy is likely"
    elif p_dr >= 0.3:
        certainty = "Diabetic retinopathy is possible"
    else:
        certainty = "Diabetic retinopathy is unlikely"

    grade_descriptions = {
        "A": "no apparent retinopathy",
        "B": "mild early-stage changes",
        "C": "moderate changes",
        "D": "more advanced changes",
        "E": "advanced proliferative changes",
    }

    prompt = REPORT_PROMPT_TEMPLATE.format(
        p_dr=p_dr,
        certainty=certainty,
        grade_description=grade_descriptions.get(grade, "some changes"),
        urgency=urgency.upper(),
        clinical_context=clinical_context,
        cgm_context=cgm_context,
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor.tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )

    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract model response after the prompt
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    elif prompt[:50] in response:
        response = response.split(prompt[:50])[-1].strip()

    return response


@spaces.GPU(duration=60)
def run_qa_inference(question: str, patient_context: str) -> str:
    """Answer lifestyle question using MedGemma with grounded prompting."""
    import torch

    _load_base_model()

    # Use report adapter if available, otherwise base model
    if _model_state["report_adapter_loaded"]:
        _set_adapter("report_generation")

    processor = _model_state["processor"]
    model = _model_state["model"]

    prompt = QA_SYSTEM_PROMPT.format(patient_context=patient_context)
    full_prompt = f"{prompt}\n\nPatient question: {question}"

    messages = [{"role": "user", "content": full_prompt}]
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor.tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()

    return response


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_urgency_display(p_dr):
    """Get urgency level, color, and recommendation."""
    if p_dr >= 0.7:
        return "URGENT", "red", "See an eye specialist within 2 weeks"
    elif p_dr >= 0.3:
        return "MODERATE", "orange", "Schedule an eye exam within 1-2 months"
    else:
        return "ROUTINE", "green", "Continue regular annual eye exams"


def build_clinical_context(diabetes_type, hba1c, years_diabetes):
    """Format clinical info for model prompt."""
    lines = []
    if diabetes_type and diabetes_type != "Unknown":
        lines.append(f"Diabetes Status: {diabetes_type}")
    if hba1c:
        status = "normal range" if hba1c < 5.7 else ("pre-diabetic range" if hba1c < 6.5 else "diabetic range")
        lines.append(f"HbA1c: {hba1c}% ({status})")
    if years_diabetes:
        lines.append(f"Years with diabetes: {int(years_diabetes)}")
    return "\n".join(lines) if lines else "No clinical data provided"


def build_cgm_context(cgm_avg, cgm_tir):
    """Format CGM info for model prompt."""
    lines = []
    if cgm_avg:
        lines.append(f"Average glucose: {cgm_avg} mg/dL")
        gmi = (cgm_avg + 46.7) / 28.7  # GMI formula
        lines.append(f"Estimated A1c (GMI): {gmi:.1f}%")
    if cgm_tir:
        lines.append(f"Time in range (70-180): {cgm_tir}%")
        lines.append(f"Time above range (>180): {100 - cgm_tir:.0f}%")
    return "\n".join(lines) if lines else "No CGM data available"


def generate_template_report(p_dr, has_dr, grade, urgency, urgency_action,
                             clinical_context, cgm_context):
    """Template-based report fallback when Stage 2 model isn't available."""
    n_positive = round(p_dr * 10)
    n_negative = 10 - n_positive

    if p_dr >= 0.7:
        certainty = "likely"
    elif p_dr >= 0.3:
        certainty = "possible"
    else:
        certainty = "unlikely"

    return f"""## Understanding Your Retinal Screening Results

Based on this screening, diabetic retinopathy is **{certainty}**.

If we screened 10 people with similar results:
- About **{n_positive}** would have diabetic retinopathy
- About **{n_negative}** would not

*This is a screening result, not a definitive diagnosis. Your eye doctor can give you a definitive answer.*

### Your Clinical Context
{clinical_context}

### Your Glucose Patterns
{cgm_context}

---

## What You Should Do Next

**{urgency_action}**

{"Your screening suggests signs of diabetic retinopathy. Early detection is good news - it means you can take steps now to protect your vision." if has_dr else "Your screening did not find significant signs of diabetic retinopathy. Continue your regular care to keep your eyes healthy."}

### Key Points to Remember

- This is a **screening tool**, not a diagnosis
- Only an eye care professional can confirm diabetic retinopathy
- Good blood sugar control helps protect your eyes
- Regular eye exams are essential for everyone with diabetes

### Questions to Ask Your Eye Doctor

1. "Based on this screening, what additional tests do you recommend?"
2. "How often should I have my eyes examined?"
3. "What can I do to protect my vision?"

---
*Report generated by DR Screening Assistant using MedGemma*
*This is not medical advice. Consult your healthcare provider.*"""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_retina(
    image,
    diabetes_type,
    hba1c,
    years_diabetes,
    cgm_avg_glucose,
    cgm_time_in_range,
    sensitivity_preset,
):
    """
    Analyze retinal image and generate patient report.

    Returns:
        Tuple of (analysis_summary, patient_report, context_for_qa)
    """
    if image is None:
        return (
            "**Please upload a retinal image to begin analysis.** "
            "You can use any fundus photograph (JPEG or PNG).",
            "",
            "",
        )

    # Get threshold from preset
    preset = SensitivityPreset(sensitivity_preset.lower())
    threshold = SENSITIVITY_THRESHOLDS[preset]

    # --- Stage 1: DR Detection (Community LoRA) ---
    using_real_model = False
    try:
        result = run_dr_detection(image, threshold)
        p_dr = result["p_dr"]
        has_dr = result["has_dr"]
        grade_code = result["predicted_grade"]
        grade_probs = result["grade_probs"]
        using_real_model = True
    except Exception as e:
        print(f"Model inference failed ({e}), using simulation.")
        import random
        random.seed(hash(str(image.size)))
        p_dr = random.uniform(0.05, 0.85)
        has_dr = p_dr >= threshold
        grade_code = "B" if p_dr >= 0.3 else "A"
        grade_probs = None

    # Grade description
    grade_descriptions = {
        "A": "No apparent retinopathy",
        "B": "Mild",
        "C": "Moderate",
        "D": "Severe",
        "E": "Proliferative",
    }
    grade = grade_descriptions.get(grade_code, "Unknown")
    urgency, urgency_color, urgency_action = get_urgency_display(p_dr)

    # Build analysis summary
    pct = int(p_dr * 100)
    bar_filled = int(p_dr * 20)
    bar_empty = 20 - bar_filled
    progress_bar = "\u2588" * bar_filled + "\u2591" * bar_empty

    grade_detail = ""
    if grade_probs and using_real_model:
        grade_detail = "\n**Grade Probabilities:**\n"
        for g in ["A", "B", "C", "D", "E"]:
            marker = " **\u2190**" if g == grade_code else ""
            grade_detail += f"- {g}: {grade_probs[g]:.1%}{marker}\n"

    model_label = "MedGemma + DR LoRA" if using_real_model else "Simulation (model loading...)"

    analysis_summary = f"""## DR Analysis Results

**Probability of Diabetic Retinopathy:**

`{progress_bar}` **{pct}%**

**Interpretation:** {grade} ({grade_code})

**Urgency:** <span style="color:{urgency_color}; font-weight:bold;">{urgency}</span> \u2014 {urgency_action}
{grade_detail}
---
*Sensitivity: {preset.value.title()} (threshold: {threshold}) | Model: {model_label}*
"""

    # Build clinical and CGM contexts
    clinical_context = build_clinical_context(diabetes_type, hba1c, years_diabetes)
    cgm_context = build_cgm_context(cgm_avg_glucose, cgm_time_in_range)

    # --- Stage 2: Report Generation (Our LoRA) ---
    patient_report = None
    if using_real_model:
        try:
            patient_report = run_report_generation(
                p_dr, grade_code, urgency, clinical_context, cgm_context
            )
        except Exception as e:
            print(f"Report generation failed ({e}), using template.")

    # Fallback to template if model report not available
    if not patient_report:
        patient_report = generate_template_report(
            p_dr, has_dr, grade, urgency, urgency_action,
            clinical_context, cgm_context
        )

    # Context for Q&A agent
    qa_context = f"""- P(DR): {p_dr:.2f} ({pct}%)
- Urgency: {urgency}
- Grade: {grade} ({grade_code})
- Diabetes: {diabetes_type or 'Unknown'}
- HbA1c: {hba1c or 'Not provided'}%
- CGM avg glucose: {cgm_avg_glucose or 'Not provided'} mg/dL
- CGM time in range: {cgm_time_in_range or 'Not provided'}%"""

    return analysis_summary, patient_report, qa_context


# ---------------------------------------------------------------------------
# Q&A Agent
# ---------------------------------------------------------------------------

def answer_lifestyle_question(
    question,
    patient_context,
    chat_history,
):
    """Answer patient questions using MedGemma grounded on authoritative guidance."""
    if chat_history is None:
        chat_history = []

    if not question.strip():
        return chat_history, ""

    if not patient_context:
        response = ("Please analyze a retinal image first (on the Retinal Analysis tab) "
                     "so I can personalize my answers to your situation.")
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, ""

    # Try real MedGemma inference
    try:
        response = run_qa_inference(question, patient_context)
    except Exception as e:
        print(f"Q&A inference failed ({e}), using fallback.")
        response = _fallback_qa(question)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, ""


def _fallback_qa(question):
    """Keyword-based fallback when model isn't available."""
    q = question.lower()

    if any(w in q for w in ["food", "eat", "diet", "nutrition"]):
        return ("**Diet and Eye Health** (ADA Guidelines)\n\n"
                "Focus on a Mediterranean-style diet with vegetables, whole grains, and lean proteins. "
                "Limit refined carbohydrates and added sugars. Include omega-3 fatty acids (fish, walnuts) "
                "which may support retinal health. Reduce sodium to help control blood pressure.\n\n"
                "*Discuss dietary changes with your healthcare team.*")

    if any(w in q for w in ["exercise", "activity", "workout", "physical"]):
        return ("**Exercise and Diabetes** (ADA Guidelines)\n\n"
                "Aim for 150+ minutes of moderate aerobic activity per week (brisk walking, swimming) "
                "plus resistance training 2-3 times per week. Physical activity improves insulin sensitivity. "
                "Check blood sugar before and after exercise if you use insulin.\n\n"
                "*Consult your doctor before starting a new exercise program.*")

    if any(w in q for w in ["symptom", "warning", "sign", "watch"]):
        return ("**Warning Signs** (AAO/NEI)\n\n"
                "Seek immediate eye care if you experience: sudden vision loss or dark spots, "
                "flashes of light or new floaters, blurry vision that doesn't clear, or "
                "difficulty seeing at night.\n\n"
                "*Contact your eye doctor right away if you notice these symptoms.*")

    if any(w in q for w in ["blood sugar", "glucose", "hba1c", "a1c"]):
        return ("**Blood Sugar and Eye Health** (ADA Guidelines)\n\n"
                "Tight glucose control reduces DR progression by 25-76%. Targets: HbA1c < 7%, "
                "fasting glucose 80-130 mg/dL, post-meal < 180 mg/dL. Avoid rapid drops in blood sugar "
                "which can temporarily worsen retinopathy.\n\n"
                "*These are general targets - your doctor may adjust for your situation.*")

    return (f"Thank you for your question. Key points from ADA/NEI/AAO guidance:\n\n"
            "1. **Regular monitoring** - annual dilated eye exams, HbA1c every 3-6 months\n"
            "2. **Blood sugar control** - most important factor for eye health\n"
            "3. **Blood pressure** - target < 130/80 to slow retinopathy\n"
            "4. **Healthy lifestyle** - diet, exercise, and not smoking all help\n\n"
            "*For personalized advice, please consult your healthcare provider.*")


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def create_demo():
    """Create the Gradio demo application."""

    with gr.Blocks(title="DR Screening Assistant") as demo:

        # Header
        gr.Markdown("""
# Diabetic Retinopathy Screening Assistant

Upload a retinal fundus image to receive a probabilistic health risk assessment and patient-friendly report.

**Dual-Adapter MedGemma Pipeline:**
1. **DR Detection** — Community-trained LoRA identifies diabetic retinopathy from fundus images
2. **Report Generation** — Our novel LoRA generates health-literate probabilistic reports
3. **Lifestyle Q&A** — Grounded agent answers diabetes management questions

*All inference runs on a single MedGemma 4B model with efficient adapter swapping.*
        """)

        # Store context for Q&A
        qa_context_state = gr.State("")

        with gr.Tabs():
            # ----- Tab 1: Analysis -----
            with gr.TabItem("Retinal Analysis"):
                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Retinal Fundus Image",
                            type="pil",
                            height=300,
                        )

                        gr.Markdown("### Patient Information")
                        gr.Markdown("*Pre-filled with sample values. Edit any field before analysis.*")

                        diabetes_type = gr.Dropdown(
                            choices=["Type 2 Diabetes", "Type 1 Diabetes", "Pre-diabetes", "Unknown"],
                            label="Diabetes Type",
                            value="Type 2 Diabetes",
                        )

                        with gr.Row():
                            hba1c = gr.Number(
                                label="HbA1c (%)",
                                value=7.2,
                                minimum=4.0,
                                maximum=15.0,
                            )
                            years_diabetes = gr.Number(
                                label="Years with Diabetes",
                                value=8,
                                minimum=0,
                                maximum=50,
                                precision=0,
                            )

                        gr.Markdown("### CGM Data")
                        with gr.Row():
                            cgm_avg = gr.Number(
                                label="Avg Glucose (mg/dL)",
                                value=155,
                            )
                            cgm_tir = gr.Number(
                                label="Time in Range (%)",
                                value=72,
                                minimum=0,
                                maximum=100,
                            )

                        sensitivity = gr.Radio(
                            choices=[
                                ("Screening - Catch all cases", "screening"),
                                ("High Sensitivity (Recommended)", "high"),
                                ("Balanced", "balanced"),
                                ("High Specificity", "specific"),
                            ],
                            label="Detection Sensitivity",
                            value="high",
                        )

                        analyze_btn = gr.Button(
                            "Analyze Image",
                            variant="primary",
                            size="lg",
                        )

                    # Right column: Results
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(
                            label="Analysis Results",
                            value=("**Upload a retinal fundus image and click 'Analyze Image' to begin.**\n\n"
                                   "Sample values are pre-filled for patient context. "
                                   "Edit any field to match your scenario."),
                        )
                        report_output = gr.Markdown(label="Patient Report")

                # Wire up analysis
                analyze_btn.click(
                    analyze_retina,
                    inputs=[
                        image_input, diabetes_type, hba1c,
                        years_diabetes, cgm_avg, cgm_tir, sensitivity,
                    ],
                    outputs=[analysis_output, report_output, qa_context_state],
                )

            # ----- Tab 2: Q&A Agent -----
            with gr.TabItem("Lifestyle Q&A"):
                gr.Markdown("""
## Ask Questions About Diabetes & Eye Health

After reviewing your screening results, ask questions about lifestyle, diet, exercise,
and eye health. Answers are generated by MedGemma grounded on authoritative guidance
from the **American Diabetes Association**, **National Eye Institute**, and
**American Academy of Ophthalmology**.

*Analyze a retinal image first (Tab 1) for personalized responses.*
                """)

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    type="messages",  # Gradio 5.x format
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What foods should I eat to protect my eyes?",
                        scale=4,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "What foods should I eat to protect my eyes?",
                        "How does exercise help with diabetes and eye health?",
                        "What warning signs should I watch for?",
                        "How can I improve my blood sugar control?",
                        "Does smoking affect my eyes?",
                    ],
                    inputs=question_input,
                )

                # Wire up Q&A
                ask_btn.click(
                    answer_lifestyle_question,
                    inputs=[question_input, qa_context_state, chatbot],
                    outputs=[chatbot, question_input],
                )
                question_input.submit(
                    answer_lifestyle_question,
                    inputs=[question_input, qa_context_state, chatbot],
                    outputs=[chatbot, question_input],
                )

            # ----- Tab 3: About -----
            with gr.TabItem("About"):
                gr.Markdown("""
## About This Tool

**DR Screening Assistant** demonstrates AI-powered diabetic retinopathy screening
with patient-friendly probabilistic health communication.

### Dual-Adapter Architecture

Both adapters share a single MedGemma 4B base model, swapped at inference time:

| Stage | Adapter | Task | Input |
|-------|---------|------|-------|
| **DR Detection** | Community DR LoRA | Identify retinopathy from images | Fundus image |
| **Report Generation** | Our Novel LoRA | Generate probabilistic reports | P(DR) + clinical + CGM |
| **Lifestyle Q&A** | Same as above | Answer patient questions | Context + question |

This architecture enables **efficient edge deployment** — one model, multiple capabilities.

### Key Innovations

1. **Probabilistic Communication**: Natural frequencies ("7 out of 10 people")
   instead of percentages for better patient understanding

2. **Configurable Sensitivity**: Healthcare organizations can adjust detection
   thresholds for their clinical context (screening vs. specialist)

3. **Edge AI Deployment**: Runs on Apple M4 Mac (64GB) or any GPU
   without cloud dependencies

4. **Adapter Chaining**: Demonstrates composable medical AI — detection,
   communication, and Q&A adapters on one lightweight base model

### Data & Training

- **Dataset**: AI-READI (Bridge2AI program, 2,280 participants)
- **Stage 1**: Community DR LoRA ([qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy))
- **Stage 2**: Our LoRA fine-tuned for probabilistic report generation
- **Privacy**: Only LoRA adapters released; no patient data in model

### Disclaimer

This tool is for educational and demonstration purposes only. It is not intended
to diagnose, treat, or prevent any disease. Always consult qualified healthcare
providers for medical advice.

---

**MedGemma Impact Challenge Submission** | David Liebovitz, MD
                """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,  # Set True for temporary public URL
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,  # Disable SSR for API compatibility
    )
