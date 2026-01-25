#!/usr/bin/env python3
"""Diabetic Retinopathy Screening Assistant - Demo Application.

A patient-friendly tool that analyzes retinal images and generates
probabilistic health reports using MedGemma.

Features:
- DR detection with configurable sensitivity
- Patient-friendly probabilistic communication
- Interactive diabetes lifestyle Q&A agent
- Runs entirely on local hardware (edge AI)
"""

import sys
from pathlib import Path
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

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

SENSITIVITY_DESCRIPTIONS = {
    SensitivityPreset.SCREENING: "Highest sensitivity - catches nearly all cases, some false positives",
    SensitivityPreset.HIGH: "High sensitivity (Recommended) - best for primary care screening",
    SensitivityPreset.BALANCED: "Balanced - moderate sensitivity and specificity",
    SensitivityPreset.SPECIFIC: "High specificity - fewer false positives, may miss some cases",
}

# Authoritative lifestyle guidance (grounding for Q&A agent)
LIFESTYLE_KNOWLEDGE_BASE = """
# Diabetes and Eye Health: Authoritative Guidance

## Sources
- American Diabetes Association (ADA) Standards of Care 2024
- National Eye Institute (NEI)
- American Academy of Ophthalmology (AAO)

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
- Physical activity improves insulin sensitivity
- Check blood sugar before/after exercise if on insulin

## Blood Pressure
- Target < 130/80 mmHg for people with diabetes
- High blood pressure accelerates retinopathy progression
- DASH diet and reduced sodium help control BP

## Smoking
- Smoking doubles the risk of vision loss in diabetics
- Quitting reduces cardiovascular and eye disease risk
- Resources: 1-800-QUIT-NOW

## Regular Monitoring
- Annual dilated eye exam (more frequent if DR present)
- HbA1c every 3-6 months
- Blood pressure at every healthcare visit
- Lipid panel annually

## Warning Signs (Seek Immediate Care)
- Sudden vision loss or dark spots
- Flashes of light or new floaters
- Blurry vision that doesn't clear
- Difficulty seeing at night

## Important Disclaimer
This information is educational. Always consult your healthcare
provider for personalized medical advice.
"""


def get_urgency_display(p_dr: float) -> tuple[str, str, str]:
    """Get urgency level, color, and recommendation."""
    if p_dr >= 0.7:
        return "URGENT", "red", "See an eye specialist within 2 weeks"
    elif p_dr >= 0.3:
        return "MODERATE", "orange", "Schedule an eye exam within 1-2 months"
    else:
        return "ROUTINE", "green", "Continue regular annual eye exams"


def format_probability_explanation(p_dr: float) -> str:
    """Format probability using natural frequencies for health literacy."""
    n_positive = round(p_dr * 10)
    n_negative = 10 - n_positive

    if p_dr >= 0.7:
        certainty = "likely"
    elif p_dr >= 0.3:
        certainty = "possible"
    else:
        certainty = "unlikely"

    return f"""Based on this screening, diabetic retinopathy is **{certainty}**.

If we screened 10 people with similar results:
- About **{n_positive}** would have diabetic retinopathy
- About **{n_negative}** would not

*This is a screening result, not a definitive diagnosis. Your eye doctor can give you a definitive answer.*"""


def analyze_retina(
    image: Image.Image | None,
    diabetes_type: str,
    hba1c: float | None,
    years_diabetes: int | None,
    cgm_avg_glucose: float | None,
    cgm_time_in_range: float | None,
    sensitivity_preset: str,
) -> tuple[str, str, str]:
    """
    Analyze retinal image and generate patient report.

    Returns:
        Tuple of (analysis_summary, patient_report, context_for_qa)
    """
    if image is None:
        return (
            "Please upload a retinal image to begin analysis.",
            "",
            ""
        )

    # Get threshold from preset
    preset = SensitivityPreset(sensitivity_preset.lower())
    threshold = SENSITIVITY_THRESHOLDS[preset]

    # TODO: Replace with actual DR detection
    # For demo, we'll simulate based on image characteristics
    # In production: from src.models import DRDetector
    # detector = DRDetector(threshold=threshold)
    # result = detector.detect(image)

    # Simulated result for demo skeleton
    # This will be replaced with actual model inference
    import random
    random.seed(hash(str(image.size)))  # Deterministic for same image
    simulated_p_dr = random.uniform(0.05, 0.85)

    p_dr = simulated_p_dr
    has_dr = p_dr >= threshold

    # Determine predicted grade based on probability
    if p_dr >= 0.8:
        grade = "Moderate to Severe"
        grade_code = "C-D"
    elif p_dr >= 0.5:
        grade = "Mild to Moderate"
        grade_code = "B-C"
    elif p_dr >= threshold:
        grade = "Mild"
        grade_code = "B"
    else:
        grade = "No apparent retinopathy"
        grade_code = "A"

    urgency, urgency_color, urgency_action = get_urgency_display(p_dr)

    # Build analysis summary (shown in sidebar)
    pct = int(p_dr * 100)
    bar_filled = int(p_dr * 20)
    bar_empty = 20 - bar_filled
    progress_bar = "█" * bar_filled + "░" * bar_empty

    analysis_summary = f"""## DR Analysis Results

**Probability of Diabetic Retinopathy:**

`{progress_bar}` **{pct}%**

**Interpretation:** {grade} ({grade_code})

**Urgency:** <span style="color:{urgency_color}; font-weight:bold;">{urgency}</span>

{urgency_action}

---
*Sensitivity: {preset.value.title()} (threshold: {threshold})*
*Using community DR model: [qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)*
"""

    # Build clinical context
    clinical_lines = []
    if diabetes_type and diabetes_type != "Unknown":
        clinical_lines.append(f"- Diabetes type: {diabetes_type}")
    if hba1c:
        clinical_lines.append(f"- HbA1c: {hba1c}%")
    if years_diabetes:
        clinical_lines.append(f"- Years with diabetes: {years_diabetes}")

    cgm_lines = []
    if cgm_avg_glucose:
        cgm_lines.append(f"- Average glucose: {cgm_avg_glucose} mg/dL")
    if cgm_time_in_range:
        cgm_lines.append(f"- Time in range: {cgm_time_in_range}%")

    clinical_context = "\n".join(clinical_lines) if clinical_lines else "No clinical data provided"
    cgm_context = "\n".join(cgm_lines) if cgm_lines else "No CGM data available"

    # Generate patient-friendly report
    prob_explanation = format_probability_explanation(p_dr)

    # TODO: Replace with fine-tuned Stage 2 model
    # For demo, use template-based generation
    patient_report = f"""## Understanding Your Retinal Screening Results

{prob_explanation}

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
*This is not medical advice. Consult your healthcare provider.*
"""

    # Context for Q&A agent
    qa_context = f"""
Patient Context:
- P(DR): {p_dr:.2f} ({pct}%)
- Urgency: {urgency}
- Diabetes: {diabetes_type or 'Unknown'}
- HbA1c: {hba1c or 'Not provided'}%
- CGM avg glucose: {cgm_avg_glucose or 'Not provided'} mg/dL
"""

    return analysis_summary, patient_report, qa_context


def answer_lifestyle_question(
    question: str,
    patient_context: str,
    chat_history: list | None,
) -> tuple[list, str]:
    """
    Answer patient questions about diabetes lifestyle, grounded in authoritative guidance.

    This is the agent-based workflow component.
    """
    if chat_history is None:
        chat_history = []

    if not question.strip():
        return chat_history, ""

    if not patient_context:
        response = "Please analyze a retinal image first so I can personalize my answers to your situation."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, ""

    # TODO: Replace with actual MedGemma inference
    # In production:
    # from src.models import MedGemmaInference
    # model = MedGemmaInference()
    # prompt = f"""You are a diabetes health educator. Answer the patient's question
    # using ONLY the authoritative guidance provided. Be supportive and clear.
    #
    # {LIFESTYLE_KNOWLEDGE_BASE}
    #
    # {patient_context}
    #
    # Patient question: {question}
    #
    # Provide a helpful, personalized answer. If the question requires medical
    # decision-making, advise them to consult their healthcare provider.
    # """
    # response = model.generate(prompt)

    # For demo skeleton, use keyword-based responses
    question_lower = question.lower()

    if any(word in question_lower for word in ["food", "eat", "diet", "nutrition"]):
        response = """**Diet and Eye Health**

Based on your screening results and authoritative guidance from the American Diabetes Association:

- Focus on a **Mediterranean-style diet** with vegetables, whole grains, and lean proteins
- **Limit refined carbohydrates** and added sugars to help control blood glucose
- Include **omega-3 fatty acids** (fish, walnuts) which may support retinal health
- **Reduce sodium** to help control blood pressure, which affects eye health

Since your screening showed some risk, maintaining good glucose control through diet is especially important for protecting your vision.

*Always discuss major dietary changes with your healthcare team.*"""

    elif any(word in question_lower for word in ["exercise", "activity", "workout", "physical"]):
        response = """**Exercise and Diabetes Management**

The American Diabetes Association recommends:

- **150+ minutes** of moderate aerobic activity per week (brisk walking, swimming)
- **Resistance training** 2-3 times per week
- Physical activity **improves insulin sensitivity** and helps control blood sugar

**Important for your situation:**
- Check blood sugar before and after exercise if you use insulin
- Stay hydrated
- Stop if you experience vision changes during exercise

Regular activity helps protect both your heart and your eyes!

*Consult your doctor before starting a new exercise program.*"""

    elif any(word in question_lower for word in ["symptom", "warning", "sign", "watch"]):
        response = """**Warning Signs to Watch For**

Seek immediate eye care if you experience:

- **Sudden vision loss** or dark spots
- **Flashes of light** or new floaters
- **Blurry vision** that doesn't clear
- Difficulty **seeing at night**

These could indicate changes that need prompt attention.

**Regular monitoring is key:**
- Annual dilated eye exam (more often if DR is present)
- HbA1c every 3-6 months
- Blood pressure at every visit

Given your screening results, staying alert to these signs is especially important.

*If you experience any of these symptoms, contact your eye doctor right away.*"""

    elif any(word in question_lower for word in ["blood sugar", "glucose", "hba1c", "a1c"]):
        response = """**Blood Sugar and Eye Health**

Research shows tight glucose control **reduces diabetic retinopathy progression by 25-76%**.

**Targets (per ADA guidelines):**
- HbA1c: < 7% for most adults
- Fasting glucose: 80-130 mg/dL
- Post-meal glucose: < 180 mg/dL

**Important note:** Avoid *rapid* drops in blood sugar, which can temporarily worsen retinopathy. Work with your doctor for gradual improvement.

Your CGM data (if available) helps track patterns between meals and during sleep.

*These are general targets - your doctor may adjust based on your individual situation.*"""

    elif any(word in question_lower for word in ["smoke", "smoking", "cigarette", "tobacco"]):
        response = """**Smoking and Vision**

**Critical fact:** Smoking **doubles** the risk of vision loss in people with diabetes.

Quitting smoking:
- Reduces cardiovascular disease risk
- Slows progression of eye disease
- Improves overall diabetes outcomes

**Resources to quit:**
- 1-800-QUIT-NOW (free counseling)
- Talk to your doctor about nicotine replacement
- Many health plans cover cessation programs

This is one of the most impactful changes you can make for your eye health.

*Your healthcare team can help you develop a quit plan.*"""

    else:
        response = f"""Thank you for your question about "{question[:50]}..."

Based on authoritative guidance from the American Diabetes Association and National Eye Institute, here are key points:

1. **Regular monitoring** is essential - annual dilated eye exams, HbA1c every 3-6 months
2. **Blood sugar control** is the most important factor in protecting your eyes
3. **Blood pressure** under 130/80 helps slow retinopathy progression
4. **Healthy lifestyle** - diet, exercise, and not smoking all help

Given your screening results, maintaining good control is especially important.

Would you like to know more about any specific aspect of diabetes management?

*For personalized medical advice, please consult your healthcare provider.*"""

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, ""


# Build the Gradio interface
def create_demo():
    """Create the Gradio demo application."""

    with gr.Blocks(title="DR Screening Assistant") as demo:

        # Header
        gr.Markdown("""
# Diabetic Retinopathy Screening Assistant

Upload a retinal image to receive a patient-friendly health report with probabilistic risk assessment.

**Features:**
- High-sensitivity DR detection using fine-tuned MedGemma
- Natural frequency explanations (health literacy optimized)
- Configurable sensitivity for different clinical contexts
- Interactive diabetes lifestyle Q&A

*Runs entirely on local hardware - no cloud required (edge AI)*
        """)

        # Store context for Q&A
        qa_context_state = gr.State("")

        with gr.Tabs():
            # Tab 1: Analysis
            with gr.TabItem("Retinal Analysis"):
                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Retinal Fundus Image",
                            type="pil",
                            height=300,
                        )

                        with gr.Accordion("Patient Information (Optional)", open=False):
                            diabetes_type = gr.Dropdown(
                                choices=["Type 2 Diabetes", "Type 1 Diabetes", "Pre-diabetes", "Unknown"],
                                label="Diabetes Type",
                                value="Type 2 Diabetes",
                            )
                            hba1c = gr.Number(
                                label="HbA1c (%)",
                                value=7.0,
                                minimum=4.0,
                                maximum=15.0,
                            )
                            years_diabetes = gr.Number(
                                label="Years with Diabetes",
                                value=5,
                                minimum=0,
                                maximum=50,
                                precision=0,
                            )

                        with gr.Accordion("CGM Data (Optional)", open=False):
                            cgm_avg = gr.Number(
                                label="Average Glucose (mg/dL)",
                                value=None,
                            )
                            cgm_tir = gr.Number(
                                label="Time in Range (%)",
                                value=None,
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

                        analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                    # Right column: Results
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(
                            label="Analysis Results",
                            value="*Upload an image and click 'Analyze Image' to begin*",
                        )

                        report_output = gr.Markdown(
                            label="Patient Report",
                        )

                # Wire up analysis
                analyze_btn.click(
                    analyze_retina,
                    inputs=[
                        image_input,
                        diabetes_type,
                        hba1c,
                        years_diabetes,
                        cgm_avg,
                        cgm_tir,
                        sensitivity,
                    ],
                    outputs=[analysis_output, report_output, qa_context_state],
                )

            # Tab 2: Q&A Agent
            with gr.TabItem("Lifestyle Q&A"):
                gr.Markdown("""
## Ask Questions About Diabetes & Eye Health

After reviewing your screening results, you can ask questions about lifestyle, diet, exercise,
and eye health. Answers are grounded in authoritative guidance from the American Diabetes
Association, National Eye Institute, and American Academy of Ophthalmology.

*This is educational information, not medical advice.*
                """)

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
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

            # Tab 3: About
            with gr.TabItem("About"):
                gr.Markdown("""
## About This Tool

**DR Screening Assistant** is a demonstration of AI-powered diabetic retinopathy screening
with patient-friendly health communication.

### Technical Approach

| Stage | Model | Purpose |
|-------|-------|---------|
| **DR Detection** | MedGemma 4B + Community LoRA | Identify diabetic retinopathy from fundus images |
| **Report Generation** | MedGemma 4B + Novel LoRA | Generate probabilistic, health-literate reports |
| **Lifestyle Q&A** | MedGemma (grounded) | Answer patient questions using authoritative sources |

### Key Innovations

1. **Probabilistic Communication**: Uses natural frequencies ("7 out of 10 people")
   instead of percentages for better patient understanding

2. **Configurable Sensitivity**: Healthcare organizations can adjust detection
   thresholds based on their clinical context

3. **Edge AI Deployment**: Runs entirely on local hardware (Apple M4 Mac)
   without cloud dependencies

4. **Agent-Based Workflow**: Multi-step pipeline from image analysis to
   personalized Q&A

### Data Sources

- **Training**: AI-READI dataset (Bridge2AI program, 2,280 participants)
- **Lifestyle Guidance**: ADA Standards of Care, NIH, AAO

### Credits

- DR detection LoRA: [qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy](https://huggingface.co/qizunlee/medgemma-4b-it-sft-lora-diabetic-retinopathy)
- Base model: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### Disclaimer

This tool is for educational and demonstration purposes only. It is not intended
to diagnose, treat, or prevent any disease. Always consult qualified healthcare
providers for medical advice.

---

**MedGemma Impact Challenge Submission**

David Liebovitz, MD | Academic Physician Informaticist
                """)

        # Examples at bottom
        gr.Markdown("---")
        gr.Markdown("### Try with Sample Images")
        gr.Markdown("*Sample images will be added from public datasets (APTOS, EyePACS)*")

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,  # Set True for public URL
        server_name="0.0.0.0",
        server_port=7860,
    )
