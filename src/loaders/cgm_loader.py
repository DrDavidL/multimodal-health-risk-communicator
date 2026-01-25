"""CGM data loader for AI-READI wearable blood glucose data.

AI-READI CGM data uses Open mHealth JSON schema, not CSV.
Data is from Dexcom G6 continuous glucose monitors.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CGMReading:
    """Single CGM glucose reading."""

    timestamp: datetime
    glucose_mg_dl: float
    event_type: str  # "EGV" for estimated glucose value


@dataclass
class CGMMetrics:
    """Glycemic variability metrics from CGM data."""

    mean_glucose: float
    std_glucose: float
    min_glucose: float
    max_glucose: float
    time_in_range: float  # 70-180 mg/dL, as percentage
    time_below_range: float  # <70 mg/dL, as percentage
    time_above_range: float  # >180 mg/dL, as percentage
    time_very_low: float  # <54 mg/dL, as percentage
    time_very_high: float  # >250 mg/dL, as percentage
    gmi: float  # Glucose Management Indicator (estimated A1c)
    cv: float  # Coefficient of variation (%)
    readings_count: int
    duration_days: float

    def to_summary(self) -> str:
        """Generate text summary for prompt construction."""
        lines = [
            "CGM Summary:",
            f"  Recording period: {self.duration_days:.1f} days ({self.readings_count} readings)",
            f"  Mean glucose: {self.mean_glucose:.0f} mg/dL",
            f"  Glucose range: {self.min_glucose:.0f} - {self.max_glucose:.0f} mg/dL",
            f"  Coefficient of variation: {self.cv:.1f}%",
            f"  Estimated A1c (GMI): {self.gmi:.1f}%",
            "",
            "Time in Range:",
            f"  In range (70-180): {self.time_in_range:.1f}%",
            f"  Below range (<70): {self.time_below_range:.1f}%",
            f"  Above range (>180): {self.time_above_range:.1f}%",
        ]

        # Add warnings for concerning values
        if self.time_very_low > 1:
            lines.append(f"  ⚠️ Very low (<54): {self.time_very_low:.1f}%")
        if self.time_very_high > 5:
            lines.append(f"  ⚠️ Very high (>250): {self.time_very_high:.1f}%")
        if self.cv > 36:
            lines.append(f"  ⚠️ High variability (CV > 36%)")

        return "\n".join(lines)


# Standard glucose thresholds (mg/dL)
LOW_THRESHOLD = 70
HIGH_THRESHOLD = 180
VERY_LOW_THRESHOLD = 54
VERY_HIGH_THRESHOLD = 250


def load_cgm_data(path: str | Path) -> list[CGMReading]:
    """Load CGM data from Open mHealth JSON file.

    Args:
        path: Path to CGM JSON file (e.g., "1001_DEX.json").

    Returns:
        List of CGMReading objects sorted by timestamp.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CGM file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Validate structure
    if "body" not in data or "cgm" not in data["body"]:
        raise ValueError(
            f"Invalid CGM format: expected body.cgm array in {path}"
        )

    readings = []
    for entry in data["body"]["cgm"]:
        try:
            # Parse timestamp
            time_frame = entry["effective_time_frame"]["time_interval"]
            timestamp_str = time_frame["start_date_time"]
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # Parse glucose value
            glucose = entry["blood_glucose"]["value"]

            # Event type (EGV = estimated glucose value)
            event_type = entry.get("event_type", "EGV")

            readings.append(CGMReading(
                timestamp=timestamp,
                glucose_mg_dl=float(glucose),
                event_type=event_type,
            ))
        except (KeyError, ValueError) as e:
            # Skip malformed entries
            continue

    # Sort by timestamp
    readings.sort(key=lambda r: r.timestamp)

    return readings


def compute_cgm_metrics(readings: list[CGMReading]) -> CGMMetrics:
    """Calculate glycemic variability metrics from CGM readings.

    Args:
        readings: List of CGMReading objects.

    Returns:
        CGMMetrics with calculated values.

    Raises:
        ValueError: If readings list is empty.
    """
    if not readings:
        raise ValueError("Cannot compute metrics from empty readings list")

    # Extract glucose values
    glucose_values = np.array([r.glucose_mg_dl for r in readings])
    n = len(glucose_values)

    # Basic statistics
    mean_glucose = np.mean(glucose_values)
    std_glucose = np.std(glucose_values)
    min_glucose = np.min(glucose_values)
    max_glucose = np.max(glucose_values)
    cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0

    # Time in range calculations
    time_in_range = np.sum(
        (glucose_values >= LOW_THRESHOLD) & (glucose_values <= HIGH_THRESHOLD)
    ) / n * 100
    time_below_range = np.sum(glucose_values < LOW_THRESHOLD) / n * 100
    time_above_range = np.sum(glucose_values > HIGH_THRESHOLD) / n * 100
    time_very_low = np.sum(glucose_values < VERY_LOW_THRESHOLD) / n * 100
    time_very_high = np.sum(glucose_values > VERY_HIGH_THRESHOLD) / n * 100

    # GMI (Glucose Management Indicator) - estimates A1c from mean glucose
    # Formula: GMI (%) = 3.31 + 0.02392 × mean glucose (mg/dL)
    gmi = 3.31 + 0.02392 * mean_glucose

    # Duration calculation
    if len(readings) >= 2:
        duration = readings[-1].timestamp - readings[0].timestamp
        duration_days = duration.total_seconds() / 86400
    else:
        # Estimate based on typical 5-minute intervals
        duration_days = n * 5 / (60 * 24)

    return CGMMetrics(
        mean_glucose=round(mean_glucose, 1),
        std_glucose=round(std_glucose, 1),
        min_glucose=round(min_glucose, 0),
        max_glucose=round(max_glucose, 0),
        time_in_range=round(time_in_range, 1),
        time_below_range=round(time_below_range, 1),
        time_above_range=round(time_above_range, 1),
        time_very_low=round(time_very_low, 1),
        time_very_high=round(time_very_high, 1),
        gmi=round(gmi, 1),
        cv=round(cv, 1),
        readings_count=n,
        duration_days=round(duration_days, 1),
    )


def get_cgm_header(path: str | Path) -> dict:
    """Extract header metadata from CGM file.

    Args:
        path: Path to CGM JSON file.

    Returns:
        Header dictionary with patient_id, timezone, etc.
    """
    with open(path) as f:
        data = json.load(f)

    return data.get("header", {})
