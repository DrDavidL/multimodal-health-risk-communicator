"""Unified participant data loader with lazy Azure download.

Orchestrates loading of all three modalities (retinal, CGM, clinical)
for a single participant, with automatic caching to minimize Azure costs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from .azure_storage import AzureBlobDownloader
from .cgm_loader import CGMMetrics, load_cgm_data, compute_cgm_metrics
from .clinical_loader import ClinicalData, ClinicalDataLoader
from .dicom_loader import DICOMLoader


@dataclass
class ParticipantData:
    """Complete multimodal data for a participant."""

    person_id: str

    # Retinal images (PIL Images, RGB)
    fundus_left: Optional[Image.Image] = None
    fundus_right: Optional[Image.Image] = None

    # CGM metrics
    cgm_metrics: Optional[CGMMetrics] = None

    # Clinical data
    clinical: Optional[ClinicalData] = None

    def has_retinal(self) -> bool:
        """Check if retinal images are available."""
        return self.fundus_left is not None or self.fundus_right is not None

    def has_cgm(self) -> bool:
        """Check if CGM data is available."""
        return self.cgm_metrics is not None

    def has_clinical(self) -> bool:
        """Check if clinical data is available."""
        return self.clinical is not None

    def is_complete(self) -> bool:
        """Check if all three modalities are available."""
        return self.has_retinal() and self.has_cgm() and self.has_clinical()

    def to_prompt_context(self) -> str:
        """Generate combined context string for MedGemma prompts.

        Returns:
            Formatted string with all available clinical and CGM data.
        """
        sections = []

        if self.clinical:
            sections.append(self.clinical.to_summary())

        if self.cgm_metrics:
            sections.append("")
            sections.append(self.cgm_metrics.to_summary())

        return "\n".join(sections)


class ParticipantLoader:
    """Load multimodal data for AI-READI participants.

    Implements lazy-download-and-cache pattern:
    - Clinical CSVs: Downloaded once, shared across participants
    - CGM JSON: Downloaded on-demand per participant
    - Retinal DICOM: Downloaded on-demand per participant

    All downloads are cached locally to minimize Azure egress costs.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./data",
        auto_download: bool = True,
    ):
        """Initialize loader.

        Args:
            cache_dir: Local directory for caching downloaded files.
            auto_download: If True, automatically download from Azure
                          when files are missing. If False, only use
                          local cache (raises error if missing).
        """
        self.cache_dir = Path(cache_dir)
        self.auto_download = auto_download

        # Initialize sub-loaders
        self.azure = AzureBlobDownloader(cache_dir=cache_dir)
        self.dicom_loader = DICOMLoader()
        self.clinical_loader = ClinicalDataLoader(data_dir=cache_dir)

        # Ensure base directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_metadata(self) -> None:
        """Download metadata files if not present.

        Downloads participants.tsv and dataset structure files.
        """
        if self.auto_download:
            self.azure.download_metadata()

    def ensure_clinical_data(self) -> None:
        """Download clinical CSV files if not present.

        Clinical data is shared across all participants (~150MB total).
        """
        if self.auto_download:
            self.azure.download_clinical_data()

    def load(
        self,
        person_id: str,
        load_retinal: bool = True,
        load_cgm: bool = True,
        load_clinical: bool = True,
        retinal_image_type: str = "mosaic",
    ) -> ParticipantData:
        """Load all available data for a participant.

        Args:
            person_id: Participant ID (e.g., "1001").
            load_retinal: Whether to load retinal images.
            load_cgm: Whether to load CGM data.
            load_clinical: Whether to load clinical data.
            retinal_image_type: Type of retinal image to load
                               ("mosaic", "uwf_central", etc.)

        Returns:
            ParticipantData with available modalities populated.
        """
        data = ParticipantData(person_id=person_id)

        # Load clinical data (requires shared CSVs)
        if load_clinical:
            self.ensure_metadata()
            self.ensure_clinical_data()
            try:
                data.clinical = self.clinical_loader.load(person_id)
            except Exception as e:
                print(f"Warning: Could not load clinical data for {person_id}: {e}")

        # Load CGM data
        if load_cgm:
            try:
                cgm_path = self._get_cgm_path(person_id)
                if cgm_path and cgm_path.exists():
                    readings = load_cgm_data(cgm_path)
                    if readings:
                        data.cgm_metrics = compute_cgm_metrics(readings)
            except Exception as e:
                print(f"Warning: Could not load CGM data for {person_id}: {e}")

        # Load retinal images
        if load_retinal:
            try:
                retinal_paths = self._get_retinal_paths(person_id, retinal_image_type)
                if "left" in retinal_paths:
                    data.fundus_left = self.dicom_loader.load(retinal_paths["left"])
                if "right" in retinal_paths:
                    data.fundus_right = self.dicom_loader.load(retinal_paths["right"])
            except Exception as e:
                print(f"Warning: Could not load retinal images for {person_id}: {e}")

        return data

    def load_retinal_only(
        self,
        person_id: str,
        eye: str = "left",
        image_type: str = "mosaic",
    ) -> Optional[Image.Image]:
        """Load only retinal image for quick inference.

        Args:
            person_id: Participant ID.
            eye: "left" or "right".
            image_type: Type of retinal image.

        Returns:
            PIL Image or None if not available.
        """
        retinal_paths = self._get_retinal_paths(person_id, image_type)
        if eye in retinal_paths:
            return self.dicom_loader.load(retinal_paths[eye])
        return None

    def _get_cgm_path(self, person_id: str) -> Optional[Path]:
        """Get path to CGM file, downloading if needed.

        Args:
            person_id: Participant ID.

        Returns:
            Path to local CGM file, or None if not available.
        """
        if self.auto_download:
            return self.azure.download_participant_cgm(person_id)

        # Check local cache only
        local_path = (
            self.cache_dir / "wearable_blood_glucose" /
            "continuous_glucose_monitoring" / "dexcom_g6" /
            person_id / f"{person_id}_DEX.json"
        )
        if local_path.exists():
            return local_path

        # Also check participant-specific cache
        alt_path = self.cache_dir / "participants" / person_id / "cgm" / f"{person_id}_DEX.json"
        if alt_path.exists():
            return alt_path

        return None

    def _get_retinal_paths(
        self,
        person_id: str,
        image_type: str = "mosaic",
    ) -> dict[str, Path]:
        """Get paths to retinal images, downloading if needed.

        Args:
            person_id: Participant ID.
            image_type: Type of retinal image.

        Returns:
            Dict mapping "left"/"right" to local file paths.
        """
        if self.auto_download:
            return self.azure.download_participant_retinal(person_id, image_type)

        # Check local cache
        paths = {}

        # Check standard Azure cache location
        retinal_dir = (
            self.cache_dir / "retinal_photography" / "cfp" /
            "icare_eidon" / person_id
        )
        if retinal_dir.exists():
            for dcm_file in retinal_dir.glob("*.dcm"):
                if image_type in dcm_file.name:
                    if "_l_" in dcm_file.name.lower() or "_cfp_l" in dcm_file.name.lower():
                        paths["left"] = dcm_file
                    elif "_r_" in dcm_file.name.lower() or "_cfp_r" in dcm_file.name.lower():
                        paths["right"] = dcm_file

        # Also check participant-specific cache
        alt_dir = self.cache_dir / "participants" / person_id / "retinal"
        if alt_dir.exists():
            for dcm_file in alt_dir.glob("*.dcm"):
                if "left" in dcm_file.name.lower() or "_l" in dcm_file.name.lower():
                    paths["left"] = dcm_file
                elif "right" in dcm_file.name.lower() or "_r" in dcm_file.name.lower():
                    paths["right"] = dcm_file

        return paths

    def get_complete_participants(self) -> list[str]:
        """Get list of participants with all three modalities.

        Returns:
            List of person_id strings.
        """
        self.ensure_metadata()
        return self.clinical_loader.get_complete_participants()

    def get_sample_participants(
        self,
        n_per_group: int = 2,
    ) -> dict[str, list[str]]:
        """Get sample participants from each study group.

        Useful for testing and demonstrations.

        Args:
            n_per_group: Number of participants per study group.

        Returns:
            Dict mapping study group to list of person_ids.
        """
        self.ensure_metadata()
        participants = self.clinical_loader._load_participants()

        groups = {}
        for pid, info in participants.items():
            # Only include complete participants
            if not (info.get("clinical_data") == "TRUE" and
                    info.get("retinal_photography") == "TRUE" and
                    info.get("wearable_blood_glucose") == "TRUE"):
                continue

            group = info.get("study_group", "unknown")
            if group not in groups:
                groups[group] = []
            if len(groups[group]) < n_per_group:
                groups[group].append(pid)

        return groups
