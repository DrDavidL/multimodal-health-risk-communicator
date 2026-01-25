"""Multimodal fusion and report generation pipeline.

This module provides the main entry point for generating
patient-friendly health reports from AI-READI multimodal data.

Example:
    from src.pipeline import ReportGenerator

    generator = ReportGenerator(cache_dir="./data")
    report = generator.generate_report("1001")

    print(report.to_markdown())
    report.save("./reports/1001_report.md")
"""

from .report_generator import HealthReport, ReportGenerator

__all__ = [
    "ReportGenerator",
    "HealthReport",
]
