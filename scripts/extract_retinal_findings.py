#!/usr/bin/env python3
"""Extract retinal findings from condition_occurrence.csv using Azure GPT-5.2.

This script maintains DUA compliance by:
1. Using Azure GPT-5.2 (BAA-compliant) to analyze the clinical data
2. Never displaying or interpreting patient data outside the BAA-compliant environment

The condition_occurrence.csv contains diabetic retinopathy diagnoses:
- mhoccur_pdr: Diabetic retinopathy (proliferative/non-proliferative)
- mhoccur_amd: Age-related macular degeneration
- mhoccur_rvo: Retinal vascular occlusion
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.azure_query import query, load_env


def extract_retinal_findings_for_participants(person_ids: list[str]) -> dict[str, dict]:
    """Use Azure GPT-5.2 to extract retinal findings for training participants.

    Args:
        person_ids: List of participant IDs to extract findings for.

    Returns:
        Dict mapping person_id to retinal findings.
    """
    load_env()

    # Build the query for Azure GPT-5.2
    prompt = f"""Analyze the condition_occurrence.csv file at ./data/clinical_data/condition_occurrence.csv.

For each of these participant IDs: {person_ids[:20]}  (showing first 20)

Extract the following retinal condition columns:
- mhoccur_pdr (Diabetic retinopathy - 1=Yes, 0=No)
- mhoccur_amd (Age-related macular degeneration - 1=Yes, 0=No)
- mhoccur_rvo (Retinal vascular occlusion - 1=Yes, 0=No)

Return as JSON with format:
{{
    "person_id": {{
        "diabetic_retinopathy": true/false,
        "amd": true/false,
        "rvo": true/false
    }}
}}

Only include participants who have at least one positive finding.
If a participant has no retinal conditions, still include them with all false values.
"""

    result = query(prompt)
    return result


def main():
    # Load the participant IDs we used for training
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"Extracting retinal findings for {len(person_ids)} training participants")
    print("Using Azure GPT-5.2 (DUA/BAA compliant)")
    print("=" * 60)

    # Query Azure GPT-5.2 to extract findings
    findings_json = extract_retinal_findings_for_participants(person_ids)

    print("\nResponse from Azure GPT-5.2:")
    print(findings_json)

    # Save for later use in training
    output_path = Path('./data/training/retinal_findings.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(findings_json if isinstance(findings_json, str) else json.dumps(findings_json, indent=2))

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
