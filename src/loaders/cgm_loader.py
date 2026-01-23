"""CGM data loader for AI-READI wearable blood glucose data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class CGMMetrics:
    """Glycemic variability metrics from CGM data."""
    
    mean_glucose: float
    std_glucose: float
    time_in_range: float  # 70-180 mg/dL, as percentage
    time_below_range: float  # <70 mg/dL, as percentage
    time_above_range: float  # >180 mg/dL, as percentage
    gmi: float  # Glucose Management Indicator
    cv: float  # Coefficient of variation
    readings_count: int
    duration_days: float


class CGMLoader:
    """Load and analyze continuous glucose monitoring data.
    
    Handles Dexcom CGM exports and calculates standard
    glycemic variability metrics.
    
    TODO: Verify actual file format from AI-READI dataset.
    """
    
    # Standard glucose targets (mg/dL)
    LOW_THRESHOLD = 70
    HIGH_THRESHOLD = 180
    VERY_LOW_THRESHOLD = 54
    VERY_HIGH_THRESHOLD = 250
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        glucose_col: str = "glucose",
    ):
        """Initialize loader.
        
        Args:
            timestamp_col: Name of timestamp column in CSV.
            glucose_col: Name of glucose value column in CSV.
        """
        self.timestamp_col = timestamp_col
        self.glucose_col = glucose_col
    
    def load(self, path: str | Path) -> pd.DataFrame:
        """Load raw CGM data from file.
        
        Args:
            path: Path to CGM data file (CSV expected).
            
        Returns:
            DataFrame with timestamp and glucose columns.
        """
        path = Path(path)
        
        # TODO: Adjust based on actual AI-READI format
        df = pd.read_csv(path)
        
        # Standardize column names
        # (May need adjustment based on actual format)
        if self.timestamp_col in df.columns:
            df["timestamp"] = pd.to_datetime(df[self.timestamp_col])
        
        if self.glucose_col in df.columns:
            df["glucose"] = pd.to_numeric(df[self.glucose_col], errors="coerce")
        
        return df.dropna(subset=["glucose"])
    
    def calculate_metrics(self, df: pd.DataFrame) -> CGMMetrics:
        """Calculate glycemic variability metrics.
        
        Args:
            df: DataFrame with glucose readings.
            
        Returns:
            CGMMetrics with calculated values.
        """
        glucose = df["glucose"].values
        
        # Basic statistics
        mean_glucose = np.mean(glucose)
        std_glucose = np.std(glucose)
        cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
        
        # Time in range calculations
        n = len(glucose)
        time_in_range = np.sum((glucose >= self.LOW_THRESHOLD) & (glucose <= self.HIGH_THRESHOLD)) / n * 100
        time_below_range = np.sum(glucose < self.LOW_THRESHOLD) / n * 100
        time_above_range = np.sum(glucose > self.HIGH_THRESHOLD) / n * 100
        
        # GMI (Glucose Management Indicator)
        # Formula: GMI (%) = 3.31 + 0.02392 × mean glucose (mg/dL)
        gmi = 3.31 + 0.02392 * mean_glucose
        
        # Duration
        if "timestamp" in df.columns:
            duration_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
        else:
            # Estimate based on typical 5-minute intervals
            duration_days = n * 5 / (60 * 24)
        
        return CGMMetrics(
            mean_glucose=round(mean_glucose, 1),
            std_glucose=round(std_glucose, 1),
            time_in_range=round(time_in_range, 1),
            time_below_range=round(time_below_range, 1),
            time_above_range=round(time_above_range, 1),
            gmi=round(gmi, 2),
            cv=round(cv, 1),
            readings_count=n,
            duration_days=round(duration_days, 1),
        )
    
    def load_participant(
        self,
        participant_dir: str | Path,
    ) -> Optional[CGMMetrics]:
        """Load and process CGM data for a participant.
        
        Args:
            participant_dir: Directory containing participant's CGM data.
            
        Returns:
            CGMMetrics if data found, None otherwise.
        """
        participant_dir = Path(participant_dir)
        
        # Search for CGM files
        # TODO: Adjust pattern based on actual AI-READI structure
        cgm_files = list(participant_dir.rglob("*.csv"))
        
        if not cgm_files:
            return None
        
        # Concatenate all CGM files for participant
        dfs = []
        for f in cgm_files:
            try:
                dfs.append(self.load(f))
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        if not dfs:
            return None
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates()
        
        return self.calculate_metrics(combined)
