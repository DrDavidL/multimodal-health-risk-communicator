"""Clinical data loader for AI-READI participant information.

AI-READI clinical data uses OMOP CDM format with CSV files shared
across all participants. Data is filtered by person_id.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ClinicalData:
    """Clinical data for a participant."""

    person_id: str
    age: Optional[int] = None
    year_of_birth: Optional[int] = None
    study_group: Optional[str] = None  # diabetes status category
    clinical_site: Optional[str] = None

    # Metabolic markers
    hba1c: Optional[float] = None
    fasting_glucose: Optional[float] = None
    insulin: Optional[float] = None
    c_peptide: Optional[float] = None

    # Anthropometrics
    bmi: Optional[float] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None

    # Vitals
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None

    # Lipids
    total_cholesterol: Optional[float] = None
    hdl: Optional[float] = None
    ldl: Optional[float] = None
    triglycerides: Optional[float] = None

    # Renal
    creatinine: Optional[float] = None
    egfr: Optional[float] = None
    urine_albumin: Optional[float] = None

    # Vision
    visual_acuity_od: Optional[float] = None  # Right eye LogMAR
    visual_acuity_os: Optional[float] = None  # Left eye LogMAR

    # Cognitive
    moca_score: Optional[int] = None

    # All measurements as raw dict
    all_measurements: dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate text summary for prompt construction."""
        lines = [f"Clinical Summary for Participant {self.person_id}:"]

        if self.age:
            lines.append(f"  Age: {self.age} years")
        if self.study_group:
            # Make study group more readable
            group_display = self.study_group.replace("_", " ").title()
            lines.append(f"  Diabetes Status: {group_display}")

        lines.append("")
        lines.append("Metabolic Markers:")
        if self.hba1c:
            status = "normal" if self.hba1c < 5.7 else "pre-diabetic" if self.hba1c < 6.5 else "diabetic"
            lines.append(f"  HbA1c: {self.hba1c}% ({status} range)")
        if self.fasting_glucose:
            lines.append(f"  Fasting Glucose: {self.fasting_glucose} mg/dL")

        if self.bmi or self.systolic_bp:
            lines.append("")
            lines.append("Vitals:")
            if self.bmi:
                category = "normal" if self.bmi < 25 else "overweight" if self.bmi < 30 else "obese"
                lines.append(f"  BMI: {self.bmi:.1f} ({category})")
            if self.systolic_bp and self.diastolic_bp:
                lines.append(f"  Blood Pressure: {self.systolic_bp:.0f}/{self.diastolic_bp:.0f} mmHg")

        if self.total_cholesterol or self.ldl:
            lines.append("")
            lines.append("Lipid Panel:")
            if self.total_cholesterol:
                lines.append(f"  Total Cholesterol: {self.total_cholesterol:.0f} mg/dL")
            if self.ldl:
                lines.append(f"  LDL: {self.ldl:.0f} mg/dL")
            if self.hdl:
                lines.append(f"  HDL: {self.hdl:.0f} mg/dL")
            if self.triglycerides:
                lines.append(f"  Triglycerides: {self.triglycerides:.0f} mg/dL")

        if self.creatinine:
            lines.append("")
            lines.append("Kidney Function:")
            lines.append(f"  Creatinine: {self.creatinine:.2f} mg/dL")

        if self.visual_acuity_od is not None or self.visual_acuity_os is not None:
            lines.append("")
            lines.append("Vision:")
            if self.visual_acuity_od is not None:
                lines.append(f"  Right Eye (OD) LogMAR: {self.visual_acuity_od:.2f}")
            if self.visual_acuity_os is not None:
                lines.append(f"  Left Eye (OS) LogMAR: {self.visual_acuity_os:.2f}")

        if self.moca_score is not None:
            lines.append("")
            status = "normal" if self.moca_score >= 26 else "mild impairment" if self.moca_score >= 18 else "moderate impairment"
            lines.append(f"Cognitive Function (MoCA): {self.moca_score}/30 ({status})")

        return "\n".join(lines)


class ClinicalDataLoader:
    """Load clinical data from OMOP CDM CSV files.

    Clinical data is stored in shared CSV files for all participants.
    This loader filters by person_id and extracts relevant variables.
    """

    # Mapping from measurement_source_value patterns to our fields
    MEASUREMENT_MAP = {
        "Hemoglobin A1c": "hba1c",
        "Glucose [Mass/volume] in Serum": "fasting_glucose",
        "Insulin [Units/volume]": "insulin",
        "C peptide [Mass/volume]": "c_peptide",
        "BMI": "bmi",
        "Weight (kilograms)": "weight_kg",
        "Height (cm)": "height_cm",
        "Waist Circumference": "waist_cm",
        "Hip Circumference": "hip_cm",
        "Systolic (mmHg)": "systolic_bp",
        "Diastolic (mmHg)": "diastolic_bp",
        "Heart Rate (bpm)": "heart_rate",
        "Cholesterol [Mass/volum": "total_cholesterol",
        "Cholesterol in HDL": "hdl",
        "Cholesterol in LDL": "ldl",
        "Triglyceride [Mass/volume]": "triglycerides",
        "Creatinine [Mass/volume] in Se": "creatinine",
        "Photopic LogMAR OD Score": "visual_acuity_od",
        "Photopic LogMAR OS Score": "visual_acuity_os",
        "moca_total_score": "moca_score",
    }

    def __init__(self, data_dir: str | Path = "./data"):
        """Initialize loader.

        Args:
            data_dir: Directory containing clinical CSV files and participants.tsv.
        """
        self.data_dir = Path(data_dir)
        self._measurement_cache: Optional[dict] = None
        self._participants_cache: Optional[dict] = None

    def _load_participants(self) -> dict[str, dict]:
        """Load and cache participants.tsv data."""
        if self._participants_cache is not None:
            return self._participants_cache

        participants_path = self.data_dir / "participants.tsv"
        if not participants_path.exists():
            return {}

        self._participants_cache = {}
        with open(participants_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pid = row.get("person_id", "")
                if pid:
                    self._participants_cache[pid] = row

        return self._participants_cache

    def _load_measurements_for_person(self, person_id: str) -> dict[str, float]:
        """Load measurements from measurement.csv for a specific person.

        Args:
            person_id: Participant ID.

        Returns:
            Dict mapping measurement names to values.
        """
        measurement_path = self.data_dir / "clinical_data" / "measurement.csv"
        if not measurement_path.exists():
            # Try alternate location
            measurement_path = self.data_dir / "measurement.csv"
            if not measurement_path.exists():
                return {}

        measurements = {}
        with open(measurement_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("person_id") != person_id:
                    continue

                source = row.get("measurement_source_value", "")
                value_str = row.get("value_as_number", "")

                if not source or not value_str:
                    continue

                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Extract variable name (after comma in source)
                if ", " in source:
                    name = source.split(", ")[1]
                else:
                    name = source

                measurements[name] = value

        return measurements

    def load(self, person_id: str) -> ClinicalData:
        """Load clinical data for a participant.

        Args:
            person_id: Participant ID (e.g., "1001").

        Returns:
            ClinicalData object with available fields populated.
        """
        # Get participant info from participants.tsv
        participants = self._load_participants()
        participant_info = participants.get(person_id, {})

        # Get measurements from measurement.csv
        measurements = self._load_measurements_for_person(person_id)

        # Create clinical data object
        clinical = ClinicalData(
            person_id=person_id,
            age=self._safe_int(participant_info.get("age")),
            study_group=participant_info.get("study_group"),
            clinical_site=participant_info.get("clinical_site"),
            all_measurements=measurements,
        )

        # Map measurements to fields
        for pattern, field_name in self.MEASUREMENT_MAP.items():
            for name, value in measurements.items():
                if pattern.lower() in name.lower():
                    if field_name == "moca_score":
                        setattr(clinical, field_name, int(value))
                    else:
                        setattr(clinical, field_name, value)
                    break

        return clinical

    def load_from_cache(
        self,
        person_id: str,
        clinical_summary_path: str | Path,
    ) -> ClinicalData:
        """Load clinical data from pre-extracted JSON cache.

        Args:
            person_id: Participant ID.
            clinical_summary_path: Path to clinical_summary.json file.

        Returns:
            ClinicalData object.
        """
        import json

        with open(clinical_summary_path) as f:
            data = json.load(f)

        participant_info = data.get("participant_info", {})
        measurements = {
            m["source"].split(", ")[-1]: float(m["value"])
            for m in data.get("measurements", [])
            if m.get("value")
        }

        clinical = ClinicalData(
            person_id=person_id,
            age=self._safe_int(participant_info.get("age")),
            study_group=participant_info.get("study_group"),
            clinical_site=participant_info.get("clinical_site"),
            all_measurements=measurements,
        )

        # Map measurements to fields
        for pattern, field_name in self.MEASUREMENT_MAP.items():
            for name, value in measurements.items():
                if pattern.lower() in name.lower():
                    if field_name == "moca_score":
                        setattr(clinical, field_name, int(value))
                    else:
                        setattr(clinical, field_name, value)
                    break

        return clinical

    def get_available_participants(self) -> list[str]:
        """Get list of all participant IDs with clinical data.

        Returns:
            List of person_id strings.
        """
        participants = self._load_participants()
        return [
            pid for pid, info in participants.items()
            if info.get("clinical_data") == "TRUE"
        ]

    def get_complete_participants(self) -> list[str]:
        """Get participants with all three modalities (retinal, CGM, clinical).

        Returns:
            List of person_id strings with complete data.
        """
        participants = self._load_participants()
        return [
            pid for pid, info in participants.items()
            if (info.get("clinical_data") == "TRUE" and
                info.get("retinal_photography") == "TRUE" and
                info.get("wearable_blood_glucose") == "TRUE")
        ]

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
