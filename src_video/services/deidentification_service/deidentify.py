"""De-identification service (placeholder).

The legacy implementation lives under `VisualProcessing/DeIdentification/`.
As we transition into the new `src_video` pipeline, this module is the intended
home for a pipeline-owned de-identification step.

For now, this is a no-op placeholder so the video pipeline can be wired end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict


def run_deidentification(input_dir: str, output_dir: str, enabled: bool = False) -> Dict[str, Any]:
    """Placeholder de-identification step.

    Args:
        input_dir: Directory containing frames/crops to de-identify.
        output_dir: Directory where de-identified outputs would be written.
        enabled: If False, this function will no-op.

    Returns:
        A small summary dict.
    """
    if not enabled:
        return {
            "enabled": False,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "note": "deidentification placeholder (disabled)",
        }

    raise NotImplementedError(
        "De-identification is not implemented in src_video yet. "
        "Port it from VisualProcessing/DeIdentification when ready."
    )
