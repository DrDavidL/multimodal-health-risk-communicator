#!/usr/bin/env python3
"""Extract retinal findings from condition_occurrence.csv.

This extracts structured data (not analyzing/interpreting it).
Retinal findings are encoded in condition_source_value column with patterns:
- mhoccur_pdr: Diabetic retinopathy
- mhoccur_amd: Age-related macular degeneration
- mhoccur_rvo: Retinal vascular occlusion

Presence of the row indicates the condition exists for that participant.
"""

import pandas as pd
import json
from pathlib import Path


def main():
    path = "./data/clinical_data/condition_occurrence.csv"

    # Load participant IDs from training set
    with open('./data/training_participants.txt') as f:
        person_ids = [line.strip() for line in f if line.strip()]

    print(f"Extracting retinal findings for {len(person_ids)} participants")

    df = pd.read_csv(path, dtype={"person_id": str})

    # Filter to retinal findings rows
    retinal_patterns = ["mhoccur_pdr", "mhoccur_amd", "mhoccur_rvo"]
    retinal_df = df[df["condition_source_value"].str.contains("|".join(retinal_patterns), na=False)]

    print(f"Found {len(retinal_df)} retinal finding records total")

    # Filter to our participants
    retinal_df = retinal_df[retinal_df["person_id"].isin(person_ids)]
    print(f"  {len(retinal_df)} records for training participants")

    # Extract findings per participant
    # Presence of the condition row = positive finding
    out = {}
    for pid in person_ids:
        pid_df = retinal_df[retinal_df["person_id"] == pid]
        findings = {
            "diabetic_retinopathy": False,
            "amd": False,
            "rvo": False,
        }
        for _, row in pid_df.iterrows():
            source = str(row["condition_source_value"])
            # Presence of row indicates condition exists
            if "mhoccur_pdr" in source:
                findings["diabetic_retinopathy"] = True
            elif "mhoccur_amd" in source:
                findings["amd"] = True
            elif "mhoccur_rvo" in source:
                findings["rvo"] = True
        out[pid] = findings

    # Summary stats (counts only, no individual data)
    total = len(out)
    dr_count = sum(1 for v in out.values() if v.get("diabetic_retinopathy"))
    amd_count = sum(1 for v in out.values() if v.get("amd"))
    rvo_count = sum(1 for v in out.values() if v.get("rvo"))

    print(f"\nSummary (aggregate counts only):")
    print(f"  Total participants: {total}")
    print(f"  With diabetic retinopathy: {dr_count}")
    print(f"  With AMD: {amd_count}")
    print(f"  With RVO: {rvo_count}")

    # Save for training
    output_path = Path('./data/training/retinal_findings.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
