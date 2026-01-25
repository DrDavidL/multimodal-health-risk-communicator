"""Data loaders for AI-READI multimodal health data.

This module provides loaders for the three data modalities:
- Retinal fundus images (DICOM with YBR_FULL_422 color space)
- Continuous glucose monitoring (JSON, Open mHealth schema)
- Clinical measurements (CSV, OMOP CDM format)

Uses lazy-download-and-cache pattern to minimize Azure egress costs.

Example:
    from src.loaders import ParticipantLoader

    loader = ParticipantLoader(cache_dir="./data")
    data = loader.load("1001")

    # Access individual modalities
    print(data.clinical.to_summary())
    print(data.cgm_metrics.to_summary())
    data.fundus_left.show()  # Display retinal image
"""

from .azure_storage import AzureBlobDownloader, get_downloader
from .cgm_loader import (
    CGMMetrics,
    CGMReading,
    compute_cgm_metrics,
    get_cgm_header,
    load_cgm_data,
)
from .clinical_loader import ClinicalData, ClinicalDataLoader
from .dicom_loader import DICOMLoader
from .participant import ParticipantData, ParticipantLoader

__all__ = [
    # Main loader
    "ParticipantLoader",
    "ParticipantData",
    # Azure utilities
    "AzureBlobDownloader",
    "get_downloader",
    # DICOM loader
    "DICOMLoader",
    # CGM loader
    "load_cgm_data",
    "compute_cgm_metrics",
    "get_cgm_header",
    "CGMReading",
    "CGMMetrics",
    # Clinical loader
    "ClinicalDataLoader",
    "ClinicalData",
]
