"""De-identification service for face anonymization.

Uses the defacer module to detect and anonymize faces in images using
CenterFace detection and configurable anonymization methods (blur, solid, mosaic, etc).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import imageio.v2 as iio

from src_video.services.deidentification_service.defacer import get_anonymized_image


def run_deidentification(
    input_dir: str,
    output_dir: str,
    enabled: bool = False,
    threshold: float = 0.2,
    replacewith: str = "blur",
    mask_scale: float = 1.3,
    ellipse: bool = True,
    draw_scores: bool = False,
) -> Dict[str, Any]:
    """Run face de-identification on images in input directory.

    Args:
        input_dir: Directory containing images to de-identify.
        output_dir: Directory where de-identified outputs will be written.
        enabled: If False, this function will no-op.
        threshold: Detection threshold for face detection (0.0-1.0).
        replacewith: Anonymization method ('blur', 'solid', 'none', 'mosaic').
        mask_scale: Scale factor for face masks (default 1.3).
        ellipse: Use ellipse masks instead of rectangular boxes.
        draw_scores: Draw detection confidence scores on output.

    Returns:
        A summary dict with processing results.
    """
    if not enabled:
        return {
            "enabled": False,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "note": "deidentification disabled",
        }

    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            return {
                "enabled": True,
                "success": False,
                "error": f"Input directory not found: {input_dir}",
            }
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all images in the input directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        processed_count = 0
        failed_count = 0
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue
            
            try:
                # Read the image
                frame = iio.imread(str(image_file))
                
                # Apply anonymization
                anonymized_frame = get_anonymized_image(
                    frame,
                    threshold=threshold,
                    replacewith=replacewith,
                    mask_scale=mask_scale,
                    ellipse=ellipse,
                    draw_scores=draw_scores,
                )
                
                # Save anonymized image
                output_file = output_path / f"{image_file.stem}_anonymized{image_file.suffix}"
                iio.imwrite(str(output_file), anonymized_frame)
                processed_count += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to process {image_file.name}: {e}")
                failed_count += 1
        
        return {
            "enabled": True,
            "success": True,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "anonymization_method": replacewith,
        }

    except Exception as e:
        return {
            "enabled": True,
            "success": False,
            "error": f"De-identification failed: {str(e)}",
        }
