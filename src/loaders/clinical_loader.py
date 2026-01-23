"""Clinical data loader for AI-READI participant information."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ClinicalData:
    """Clinical data for a participant."""
    
    participant_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    diabetes_status: Optional[str] = None  # "Type 2", "Control", etc.
    diabetes_duration_years: Optional[float] = None
    hba1c: Optional[float] = None
    bmi: Optional[float] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    medications: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    
    # Additional fields as discovered in data
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> str:
        """Generate text summary for prompt construction."""
        lines = [f"Participant: {self.participant_id}"]
        
        if self.age:
            lines.append(f"Age: {self.age} years")
        if self.sex:
            lines.append(f"Sex: {self.sex}")
        if self.diabetes_status:
            lines.append(f"Diabetes status: {self.diabetes_status}")
        if self.diabetes_duration_years:
            lines.append(f"Diabetes duration: {self.diabetes_duration_years} years")
        if self.hba1c:
            lines.append(f"HbA1c: {self.hba1c}%")
        if self.bmi:
            lines.append(f"BMI: {self.bmi}")
        if self.systolic_bp and self.diastolic_bp:
            lines.append(f"Blood pressure: {self.systolic_bp}/{self.diastolic_bp} mmHg")
        if self.medications:
            lines.append(f"Medications: {', '.join(self.medications)}")
        if self.comorbidities:
            lines.append(f"Comorbidities: {', '.join(self.comorbidities)}")
        
        return "\n".join(lines)


class ClinicalLoader:
    """Load clinical data for AI-READI participants.
    
    TODO: Verify actual file format and variable names from AI-READI dataset.
    """
    
    # Mapping from AI-READI variable names to our schema
    # TODO: Update based on actual dataset structure
    VARIABLE_MAP = {
        "participant_id": ["participant_id", "subject_id", "id"],
        "age": ["age", "age_years", "Age"],
        "sex": ["sex", "gender", "Sex"],
        "hba1c": ["hba1c", "HbA1c", "a1c", "glycated_hemoglobin"],
        "bmi": ["bmi", "BMI", "body_mass_index"],
        "systolic_bp": ["systolic_bp", "sbp", "systolic"],
        "diastolic_bp": ["diastolic_bp", "dbp", "diastolic"],
    }
    
    def __init__(self, clinical_data_dir: Optional[str | Path] = None):
        """Initialize loader.
        
        Args:
            clinical_data_dir: Base directory for clinical data files.
        """
        self.clinical_data_dir = Path(clinical_data_dir) if clinical_data_dir else None
    
    def load(self, participant_id: str) -> ClinicalData:
        """Load clinical data for a participant.
        
        Args:
            participant_id: Participant identifier (e.g., "sub-001").
            
        Returns:
            ClinicalData object with available fields populated.
        """
        if not self.clinical_data_dir:
            raise ValueError("clinical_data_dir not set")
        
        # Try to find participant's clinical data
        # TODO: Adjust based on actual AI-READI structure
        possible_paths = [
            self.clinical_data_dir / f"{participant_id}.json",
            self.clinical_data_dir / participant_id / "clinical.json",
            self.clinical_data_dir / f"{participant_id}_clinical.json",
        ]
        
        raw_data = {}
        for path in possible_paths:
            if path.exists():
                with open(path) as f:
                    raw_data = json.load(f)
                break
        
        return self._parse_clinical_data(participant_id, raw_data)
    
    def load_from_participants_file(
        self,
        participants_file: str | Path,
        participant_id: str,
    ) -> ClinicalData:
        """Load clinical data from master participants file.
        
        Args:
            participants_file: Path to participants.json or participants.tsv.
            participant_id: Participant identifier.
            
        Returns:
            ClinicalData object.
        """
        participants_file = Path(participants_file)
        
        if participants_file.suffix == ".json":
            with open(participants_file) as f:
                data = json.load(f)
            # Find participant in list
            for entry in data:
                if entry.get("participant_id") == participant_id:
                    return self._parse_clinical_data(participant_id, entry)
        
        elif participants_file.suffix == ".tsv":
            import pandas as pd
            df = pd.read_csv(participants_file, sep="\t")
            row = df[df["participant_id"] == participant_id]
            if not row.empty:
                return self._parse_clinical_data(
                    participant_id, 
                    row.iloc[0].to_dict()
                )
        
        # Return empty if not found
        return ClinicalData(participant_id=participant_id)
    
    def _parse_clinical_data(
        self,
        participant_id: str,
        raw_data: dict[str, Any],
    ) -> ClinicalData:
        """Parse raw data into ClinicalData object.
        
        Args:
            participant_id: Participant identifier.
            raw_data: Dictionary of clinical variables.
            
        Returns:
            ClinicalData with fields populated from raw_data.
        """
        def find_value(field_name: str) -> Any:
            """Find value using variable name mapping."""
            possible_names = self.VARIABLE_MAP.get(field_name, [field_name])
            for name in possible_names:
                if name in raw_data:
                    return raw_data[name]
            return None
        
        return ClinicalData(
            participant_id=participant_id,
            age=find_value("age"),
            sex=find_value("sex"),
            diabetes_status=raw_data.get("diabetes_status") or raw_data.get("group"),
            diabetes_duration_years=raw_data.get("diabetes_duration"),
            hba1c=find_value("hba1c"),
            bmi=find_value("bmi"),
            systolic_bp=find_value("systolic_bp"),
            diastolic_bp=find_value("diastolic_bp"),
            medications=raw_data.get("medications", []),
            comorbidities=raw_data.get("comorbidities", []),
            raw_data=raw_data,
        )
