"""Retinal findings extraction from OMOP CDM clinical data.

This module provides functions to extract diabetic retinopathy, AMD, and RVO
diagnoses from the AI-READI condition_occurrence.csv file.

## Data Location

Retinal findings are stored in `clinical_data/condition_occurrence.csv` using
OMOP CDM format. The relevant columns are:

- `person_id`: Participant identifier (string)
- `condition_source_value`: Contains diagnosis codes like:
  - `mhoccur_pdr`: Diabetic retinopathy (proliferative or non-proliferative)
  - `mhoccur_amd`: Age-related macular degeneration
  - `mhoccur_rvo`: Retinal vascular occlusion

## Usage

```python
from src.training.retinal_findings import load_retinal_findings, format_retinal_findings

# Load all retinal findings
findings = load_retinal_findings()

# Get findings for a specific participant
participant_findings = findings.get("1001", {})
# Returns: {"diabetic_retinopathy": True, "amd": False, "rvo": False}

# Format for prompt context
context = format_retinal_findings(participant_findings)
# Returns: "Retinal Eye Exam Findings:\\n- Diabetic retinopathy..."
```

## Ground Truth

The presence of a row in condition_occurrence.csv with these codes indicates
a positive diagnosis. These diagnoses come from clinical evaluation, not
automated image analysis.
"""

import pandas as pd
from pathlib import Path


def load_retinal_findings(
    condition_csv: str | Path = "./data/clinical_data/condition_occurrence.csv",
) -> dict[str, dict]:
    """Load retinal findings from condition_occurrence.csv.

    Extracts diabetic retinopathy, AMD, and RVO diagnoses from OMOP CDM
    condition_occurrence table.

    Args:
        condition_csv: Path to condition_occurrence.csv file.

    Returns:
        Dict mapping person_id to findings dict:
        {
            "1001": {
                "diabetic_retinopathy": True,
                "amd": False,
                "rvo": False
            },
            ...
        }

    Note:
        Only participants WITH positive findings are included in the dict.
        Use .get(person_id, {}) to safely handle participants without findings.
    """
    df = pd.read_csv(condition_csv, dtype={"person_id": str})

    # Filter to retinal findings rows
    retinal_patterns = ["mhoccur_pdr", "mhoccur_amd", "mhoccur_rvo"]
    retinal_df = df[
        df["condition_source_value"].str.contains("|".join(retinal_patterns), na=False)
    ]

    findings = {}
    for _, row in retinal_df.iterrows():
        pid = row["person_id"]
        if pid not in findings:
            findings[pid] = {
                "diabetic_retinopathy": False,
                "amd": False,
                "rvo": False,
            }

        source = str(row["condition_source_value"])
        if "mhoccur_pdr" in source:
            findings[pid]["diabetic_retinopathy"] = True
        elif "mhoccur_amd" in source:
            findings[pid]["amd"] = True
        elif "mhoccur_rvo" in source:
            findings[pid]["rvo"] = True

    return findings


def format_retinal_findings(findings: dict | None) -> str:
    """Format retinal findings for inclusion in prompts.

    Creates a human-readable description of retinal findings for use
    as context in prompts to GPT-5.2 or MedGemma.

    Args:
        findings: Dict with keys "diabetic_retinopathy", "amd", "rvo"
                  and boolean values. None or empty dict treated as
                  no findings.

    Returns:
        Formatted string describing the findings.

    Example:
        >>> format_retinal_findings({"diabetic_retinopathy": True, "amd": False, "rvo": False})
        "Retinal Eye Exam Findings:\\n- Diabetic retinopathy (diabetes-related changes in the retina)"
    """
    if not findings:
        return "Retinal Eye Exam: No significant findings documented."

    conditions = []
    if findings.get("diabetic_retinopathy"):
        conditions.append(
            "Diabetic retinopathy (diabetes-related changes in the retina)"
        )
    if findings.get("amd"):
        conditions.append("Age-related macular degeneration (AMD)")
    if findings.get("rvo"):
        conditions.append(
            "Retinal vascular occlusion (blockage in retinal blood vessels)"
        )

    if not conditions:
        return "Retinal Eye Exam: No significant findings documented."

    return "Retinal Eye Exam Findings:\n- " + "\n- ".join(conditions)


def format_retinal_findings_for_target(findings: dict | None) -> str:
    """Format retinal findings as a target response for Stage 1 training.

    Creates a structured description that MedGemma should learn to produce
    when analyzing a retinal image.

    Args:
        findings: Dict with keys "diabetic_retinopathy", "amd", "rvo".

    Returns:
        Target response for training.
    """
    if not findings or not any(findings.values()):
        return """Based on the retinal image examination:

**Findings:** No significant abnormalities detected.

The retinal photograph shows:
- Normal optic disc appearance
- Healthy blood vessel patterns
- No signs of diabetic retinopathy, macular degeneration, or vascular occlusion

**Recommendation:** Continue regular eye exams as recommended by your healthcare provider."""

    conditions = []
    details = []

    if findings.get("diabetic_retinopathy"):
        conditions.append("Diabetic retinopathy")
        details.append(
            "- **Diabetic retinopathy** detected: This condition involves changes to the blood "
            "vessels in the retina caused by diabetes. Early detection allows for treatment "
            "to prevent vision loss."
        )

    if findings.get("amd"):
        conditions.append("Age-related macular degeneration (AMD)")
        details.append(
            "- **Age-related macular degeneration (AMD)** detected: This condition affects "
            "the central part of the retina (macula) and can impact central vision over time."
        )

    if findings.get("rvo"):
        conditions.append("Retinal vascular occlusion")
        details.append(
            "- **Retinal vascular occlusion** detected: This involves a blockage in the "
            "blood vessels of the retina, which requires monitoring and may need treatment."
        )

    return f"""Based on the retinal image examination:

**Findings:** {", ".join(conditions)}

{chr(10).join(details)}

**Recommendation:** Discuss these findings with your healthcare provider. They may recommend:
- More frequent eye exams
- Additional testing
- Referral to a retinal specialist
- Optimizing blood sugar control (if diabetic retinopathy is present)"""


def get_retinal_findings_summary(
    findings: dict[str, dict],
    person_ids: list[str] | None = None,
) -> dict:
    """Get summary statistics for retinal findings.

    Args:
        findings: Dict from load_retinal_findings().
        person_ids: Optional list to filter to specific participants.

    Returns:
        Summary dict with counts.
    """
    if person_ids:
        subset = {pid: findings.get(pid, {}) for pid in person_ids}
    else:
        subset = findings

    return {
        "total_participants": len(subset),
        "with_diabetic_retinopathy": sum(
            1 for f in subset.values() if f.get("diabetic_retinopathy")
        ),
        "with_amd": sum(1 for f in subset.values() if f.get("amd")),
        "with_rvo": sum(1 for f in subset.values() if f.get("rvo")),
        "with_any_finding": sum(1 for f in subset.values() if any(f.values())),
        "no_findings": sum(1 for f in subset.values() if not any(f.values())),
    }
