"""Azure Blob Storage utilities for AI-READI data access.

Implements lazy-download-and-cache pattern to minimize Azure egress costs.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


# Azure storage configuration - load from environment variables
# Set these in your environment or .env file (not committed to repo)
STORAGE_ACCOUNT = os.environ.get("AIREADI_STORAGE_ACCOUNT", "")
CONTAINER_NAME = os.environ.get("AIREADI_CONTAINER", "")
BLOB_PREFIX = os.environ.get("AIREADI_BLOB_PREFIX", "")


class AzureBlobDownloader:
    """Download blobs from Azure with local caching.

    Uses Azure CLI for authentication (requires `az login`).
    Downloads are cached locally to avoid repeated egress costs.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./data",
        storage_account: str = STORAGE_ACCOUNT,
        container_name: str = CONTAINER_NAME,
        blob_prefix: str = BLOB_PREFIX,
    ):
        """Initialize downloader.

        Args:
            cache_dir: Local directory for caching downloaded files.
            storage_account: Azure storage account name.
            container_name: Azure container name.
            blob_prefix: Prefix path within the container (UUID/dataset).
        """
        self.cache_dir = Path(cache_dir)
        self.storage_account = storage_account
        self.container_name = container_name
        self.blob_prefix = blob_prefix

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_blob_path(self, relative_path: str) -> str:
        """Construct full blob path from relative path.

        Args:
            relative_path: Path relative to dataset root
                          (e.g., "clinical_data/measurement.csv")

        Returns:
            Full blob path including prefix.
        """
        return f"{self.blob_prefix}/{relative_path}"

    def get_local_path(self, relative_path: str) -> Path:
        """Get local cache path for a blob.

        Args:
            relative_path: Path relative to dataset root.

        Returns:
            Local file path in cache directory.
        """
        return self.cache_dir / relative_path

    def download_if_missing(
        self,
        relative_path: str,
        force: bool = False,
    ) -> Path:
        """Download blob if not already cached locally.

        Args:
            relative_path: Path relative to dataset root.
            force: If True, download even if file exists locally.

        Returns:
            Local file path (downloaded or from cache).

        Raises:
            RuntimeError: If download fails.
        """
        local_path = self.get_local_path(relative_path)

        # Return cached file if exists
        if local_path.exists() and not force:
            return local_path

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from Azure
        blob_path = self.get_blob_path(relative_path)

        cmd = [
            "az", "storage", "blob", "download",
            "--account-name", self.storage_account,
            "--container-name", self.container_name,
            "--name", blob_path,
            "--file", str(local_path),
            "--auth-mode", "key",
            "--only-show-errors",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download {blob_path}: {result.stderr}"
            )

        return local_path

    def list_blobs(
        self,
        prefix: str,
        pattern: Optional[str] = None,
    ) -> list[str]:
        """List blobs under a prefix.

        Args:
            prefix: Path prefix relative to dataset root.
            pattern: Optional glob pattern to filter results.

        Returns:
            List of blob paths (relative to dataset root).
        """
        full_prefix = self.get_blob_path(prefix)

        cmd = [
            "az", "storage", "blob", "list",
            "--account-name", self.storage_account,
            "--container-name", self.container_name,
            "--prefix", full_prefix,
            "--query", "[].name",
            "--output", "tsv",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return []

        blobs = result.stdout.strip().split("\n")

        # Remove blob prefix to get relative paths
        prefix_len = len(self.blob_prefix) + 1  # +1 for trailing slash
        relative_paths = [b[prefix_len:] for b in blobs if b]

        # Apply pattern filter if specified
        if pattern:
            from fnmatch import fnmatch
            relative_paths = [p for p in relative_paths if fnmatch(p, pattern)]

        return relative_paths

    def download_participant_retinal(
        self,
        person_id: str,
        image_type: str = "mosaic",
    ) -> dict[str, Path]:
        """Download retinal images for a participant.

        Args:
            person_id: Participant ID (e.g., "1001").
            image_type: Type of image ("mosaic", "uwf_central", etc.)

        Returns:
            Dict mapping eye ("left", "right") to local file paths.
        """
        prefix = f"retinal_photography/cfp/icare_eidon/{person_id}"
        blobs = self.list_blobs(prefix)

        # Filter for requested image type
        blobs = [b for b in blobs if image_type in b]

        paths = {}
        for blob in blobs:
            local_path = self.download_if_missing(blob)

            # Determine eye from filename
            if "_l_" in blob.lower() or "_cfp_l" in blob.lower():
                paths["left"] = local_path
            elif "_r_" in blob.lower() or "_cfp_r" in blob.lower():
                paths["right"] = local_path

        return paths

    def download_participant_cgm(self, person_id: str) -> Optional[Path]:
        """Download CGM data for a participant.

        Args:
            person_id: Participant ID (e.g., "1001").

        Returns:
            Local file path, or None if not available.
        """
        relative_path = (
            f"wearable_blood_glucose/continuous_glucose_monitoring/"
            f"dexcom_g6/{person_id}/{person_id}_DEX.json"
        )

        try:
            return self.download_if_missing(relative_path)
        except RuntimeError:
            return None

    def download_clinical_data(self) -> dict[str, Path]:
        """Download all clinical data CSV files.

        Clinical data is shared across all participants,
        so we download all files once.

        Returns:
            Dict mapping file type to local path.
        """
        files = {
            "measurement": "clinical_data/measurement.csv",
            "person": "clinical_data/person.csv",
            "condition": "clinical_data/condition_occurrence.csv",
            "observation": "clinical_data/observation.csv",
            "visit": "clinical_data/visit_occurrence.csv",
        }

        paths = {}
        for name, relative_path in files.items():
            try:
                paths[name] = self.download_if_missing(relative_path)
            except RuntimeError as e:
                print(f"Warning: Could not download {name}: {e}")

        return paths

    def download_metadata(self) -> dict[str, Path]:
        """Download dataset metadata files.

        Returns:
            Dict mapping file type to local path.
        """
        files = {
            "participants_tsv": "participants.tsv",
            "participants_json": "participants.json",
            "structure": "dataset_structure_description.json",
        }

        paths = {}
        for name, relative_path in files.items():
            try:
                paths[name] = self.download_if_missing(relative_path)
            except RuntimeError as e:
                print(f"Warning: Could not download {name}: {e}")

        return paths


# Convenience function for quick access
def get_downloader(cache_dir: str | Path = "./data") -> AzureBlobDownloader:
    """Get configured Azure downloader instance.

    Args:
        cache_dir: Local cache directory.

    Returns:
        Configured AzureBlobDownloader.
    """
    return AzureBlobDownloader(cache_dir=cache_dir)
