"""Multimodal health report generator.

Orchestrates data loading and MedGemma inference to produce
patient-friendly health reports from AI-READI multimodal data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ..loaders import ParticipantData, ParticipantLoader
from ..models.medgemma import MedGemmaInference


@dataclass
class HealthReport:
    """Generated health report for a participant."""

    person_id: str
    report_text: str
    modalities_used: list[str]
    warnings: list[str]

    # Raw components (for debugging/analysis)
    clinical_summary: Optional[str] = None
    cgm_summary: Optional[str] = None
    fundus_analysis: Optional[str] = None

    def to_markdown(self) -> str:
        """Format report as markdown for display."""
        lines = [
            f"# Health Report for Participant {self.person_id}",
            "",
            f"**Data sources used:** {', '.join(self.modalities_used)}",
            "",
        ]

        if self.warnings:
            lines.append("**Notes:**")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(self.report_text)

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())


class ReportGenerator:
    """Generate patient-friendly health reports from multimodal data.

    This is the main entry point for the Multimodal Health Risk Communicator.
    It orchestrates:
    1. Loading participant data (retinal images, CGM, clinical)
    2. Running MedGemma inference on the multimodal inputs
    3. Generating patient-friendly explanations
    """

    # Prompt templates
    MULTIMODAL_PROMPT = """You are a compassionate health educator helping patients understand their health data. Your task is to explain medical findings to someone with no medical background. Use simple, clear language.

## Patient Clinical Information
{clinical_summary}

## Continuous Glucose Monitoring Data
{cgm_summary}

## Task
Look at the attached retinal fundus image(s) along with the clinical and glucose data above. Then provide a comprehensive but easy-to-understand health summary.

Structure your response as follows:

### What Your Eye Exam Shows
Explain what you observe in the retinal image in simple terms. If there are any findings that might relate to diabetes, explain the connection.

### Understanding Your Glucose Patterns
Summarize what the CGM data tells us about blood sugar control. Highlight both positive aspects and areas that might need attention.

### How It All Connects
Explain how the different pieces of information (eye health, glucose control, other clinical measures) relate to each other and the patient's overall health.

### Key Points to Remember
List 3-5 important takeaways in simple bullet points.

### Questions for Your Healthcare Team
Suggest 2-3 specific questions the patient might want to ask at their next appointment.

Remember: Be honest but supportive. Celebrate what's going well. For any concerns, recommend professional follow-up without causing alarm."""

    RETINAL_ONLY_PROMPT = """You are a compassionate health educator. Analyze this retinal fundus image and explain your findings in simple terms that a patient with no medical background can understand.

{context}

Please describe:
1. What you observe in the image
2. What this might mean for the patient's eye health
3. Any recommendations for follow-up

Use simple language and avoid medical jargon."""

    CGM_CLINICAL_ONLY_PROMPT = """You are a compassionate health educator helping a patient understand their health data.

## Patient Clinical Information
{clinical_summary}

## Continuous Glucose Monitoring Data
{cgm_summary}

Please provide:
1. A simple explanation of what these numbers mean
2. What's going well
3. Areas that might need attention
4. 2-3 questions to discuss with their healthcare team

Use simple language that someone without medical training can understand."""

    def __init__(
        self,
        cache_dir: str | Path = "./data",
        model_id: str = "google/medgemma-4b-it",
        auto_download: bool = True,
        preload_model: bool = False,
    ):
        """Initialize report generator.

        Args:
            cache_dir: Directory for data caching.
            model_id: MedGemma model identifier.
            auto_download: Whether to auto-download from Azure.
            preload_model: Whether to load model immediately.
        """
        self.cache_dir = Path(cache_dir)

        # Initialize components
        self.loader = ParticipantLoader(
            cache_dir=cache_dir,
            auto_download=auto_download,
        )
        self.model = MedGemmaInference(model_id=model_id)

        if preload_model:
            self.model.load()

    def generate_report(
        self,
        person_id: str,
        prefer_eye: str = "left",
        include_both_eyes: bool = False,
    ) -> HealthReport:
        """Generate health report for a participant.

        Args:
            person_id: Participant ID (e.g., "1001").
            prefer_eye: Which eye to use if only one ("left" or "right").
            include_both_eyes: Whether to include both eyes if available.

        Returns:
            HealthReport with generated content.
        """
        # Load participant data
        data = self.loader.load(person_id)

        modalities = []
        warnings = []

        # Collect available data
        clinical_summary = None
        cgm_summary = None
        images = []

        if data.has_clinical():
            clinical_summary = data.clinical.to_summary()
            modalities.append("clinical")
        else:
            warnings.append("Clinical data not available")

        if data.has_cgm():
            cgm_summary = data.cgm_metrics.to_summary()
            modalities.append("CGM")
        else:
            warnings.append("CGM data not available")

        if data.has_retinal():
            modalities.append("retinal imaging")
            if include_both_eyes:
                if data.fundus_left:
                    images.append(data.fundus_left)
                if data.fundus_right:
                    images.append(data.fundus_right)
            else:
                # Use preferred eye, or whichever is available
                if prefer_eye == "left" and data.fundus_left:
                    images.append(data.fundus_left)
                elif prefer_eye == "right" and data.fundus_right:
                    images.append(data.fundus_right)
                elif data.fundus_left:
                    images.append(data.fundus_left)
                elif data.fundus_right:
                    images.append(data.fundus_right)
        else:
            warnings.append("Retinal images not available")

        # Generate report based on available modalities
        if images and (clinical_summary or cgm_summary):
            # Full multimodal report
            report_text = self._generate_multimodal_report(
                images, clinical_summary, cgm_summary
            )
        elif images:
            # Retinal only
            report_text = self._generate_retinal_report(images, data)
        elif clinical_summary or cgm_summary:
            # CGM/clinical only (no image)
            report_text = self._generate_text_only_report(
                clinical_summary, cgm_summary
            )
        else:
            report_text = "Unable to generate report: no data available for this participant."
            warnings.append("No data available")

        return HealthReport(
            person_id=person_id,
            report_text=report_text,
            modalities_used=modalities,
            warnings=warnings,
            clinical_summary=clinical_summary,
            cgm_summary=cgm_summary,
        )

    def _generate_multimodal_report(
        self,
        images: list[Image.Image],
        clinical_summary: Optional[str],
        cgm_summary: Optional[str],
    ) -> str:
        """Generate report using all three modalities."""
        prompt = self.MULTIMODAL_PROMPT.format(
            clinical_summary=clinical_summary or "Not available",
            cgm_summary=cgm_summary or "Not available",
        )

        return self.model.generate(
            images=images,
            prompt=prompt,
            max_new_tokens=2000,
            temperature=0.7,
        )

    def _generate_retinal_report(
        self,
        images: list[Image.Image],
        data: ParticipantData,
    ) -> str:
        """Generate report from retinal images only."""
        context = ""
        if data.clinical and data.clinical.study_group:
            context = f"Patient context: {data.clinical.study_group.replace('_', ' ')}"

        prompt = self.RETINAL_ONLY_PROMPT.format(context=context)

        return self.model.generate(
            images=images,
            prompt=prompt,
            max_new_tokens=1500,
        )

    def _generate_text_only_report(
        self,
        clinical_summary: Optional[str],
        cgm_summary: Optional[str],
    ) -> str:
        """Generate report from CGM and clinical data only (no image)."""
        prompt = self.CGM_CLINICAL_ONLY_PROMPT.format(
            clinical_summary=clinical_summary or "Not available",
            cgm_summary=cgm_summary or "Not available",
        )

        # Text-only generation - pass empty image list
        # Note: MedGemma can do text-only, but we'll need to handle this
        # For now, return a formatted version of the summaries
        lines = [
            "## Your Health Summary",
            "",
        ]

        if clinical_summary:
            lines.append(clinical_summary)
            lines.append("")

        if cgm_summary:
            lines.append(cgm_summary)
            lines.append("")

        lines.extend([
            "---",
            "",
            "*Note: Retinal imaging was not available for this report. "
            "Please discuss these findings with your healthcare provider.*",
        ])

        return "\n".join(lines)

    def generate_batch_reports(
        self,
        person_ids: list[str],
        output_dir: str | Path = "./reports",
    ) -> list[HealthReport]:
        """Generate reports for multiple participants.

        Args:
            person_ids: List of participant IDs.
            output_dir: Directory to save reports.

        Returns:
            List of generated HealthReports.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reports = []
        for i, person_id in enumerate(person_ids):
            print(f"Generating report {i+1}/{len(person_ids)}: {person_id}")
            try:
                report = self.generate_report(person_id)
                report.save(output_dir / f"{person_id}_report.md")
                reports.append(report)
            except Exception as e:
                print(f"  Error: {e}")

        return reports


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.report_generator <person_id>")
        print("Example: python -m src.pipeline.report_generator 1001")
        sys.exit(1)

    person_id = sys.argv[1]

    generator = ReportGenerator(cache_dir="./data")
    report = generator.generate_report(person_id)

    print(report.to_markdown())
