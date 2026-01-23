"""DICOM loader for AI-READI retinal fundus images."""

from pathlib import Path
from typing import Optional

import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import convert_color_space


class DICOMLoader:
    """Load and preprocess retinal fundus DICOM images.
    
    Handles color space conversion from YBR_FULL_422 to RGB,
    which is common in AI-READI retinal photography.
    """
    
    SUPPORTED_COLOR_SPACES = {"YBR_FULL_422", "YBR_FULL", "RGB"}
    
    def __init__(self, target_size: Optional[tuple[int, int]] = None):
        """Initialize loader.
        
        Args:
            target_size: Optional (width, height) to resize images.
                         If None, returns original size.
        """
        self.target_size = target_size
    
    def load(self, path: str | Path) -> Image.Image:
        """Load a single DICOM file and return as PIL Image.
        
        Args:
            path: Path to DICOM file.
            
        Returns:
            PIL Image in RGB format.
            
        Raises:
            FileNotFoundError: If DICOM file doesn't exist.
            ValueError: If color space is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")
        
        dcm = pydicom.dcmread(path)
        pixels = dcm.pixel_array
        
        # Handle color space conversion
        photometric = getattr(dcm, "PhotometricInterpretation", "RGB")
        
        if photometric == "YBR_FULL_422":
            pixels = convert_color_space(pixels, "YBR_FULL_422", "RGB")
        elif photometric == "YBR_FULL":
            pixels = convert_color_space(pixels, "YBR_FULL", "RGB")
        elif photometric != "RGB":
            raise ValueError(f"Unsupported color space: {photometric}")
        
        # Convert to PIL Image
        image = Image.fromarray(pixels)
        
        # Resize if requested
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def load_participant(
        self,
        participant_dir: str | Path,
    ) -> dict[str, Image.Image]:
        """Load all fundus images for a participant.
        
        Searches for DICOM files organized by eye (OD/OS).
        
        Args:
            participant_dir: Directory containing participant's images.
            
        Returns:
            Dictionary mapping eye identifier to PIL Image.
            E.g., {"OD": Image, "OS": Image}
        """
        participant_dir = Path(participant_dir)
        images = {}
        
        # Search for DICOM files
        for dcm_file in participant_dir.rglob("*.dcm"):
            # Try to determine eye from path or DICOM metadata
            eye = self._determine_eye(dcm_file)
            
            try:
                images[eye] = self.load(dcm_file)
            except Exception as e:
                print(f"Warning: Failed to load {dcm_file}: {e}")
                continue
        
        return images
    
    def _determine_eye(self, path: Path) -> str:
        """Determine eye laterality from path or DICOM metadata.
        
        Args:
            path: Path to DICOM file.
            
        Returns:
            "OD" (right eye), "OS" (left eye), or path stem if unknown.
        """
        # Check path components
        path_str = str(path).upper()
        if "/OD/" in path_str or "_OD" in path_str:
            return "OD"
        if "/OS/" in path_str or "_OS" in path_str:
            return "OS"
        
        # Try DICOM metadata
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            laterality = getattr(dcm, "ImageLaterality", None)
            if laterality == "R":
                return "OD"
            if laterality == "L":
                return "OS"
        except Exception:
            pass
        
        # Fallback to filename
        return path.stem
