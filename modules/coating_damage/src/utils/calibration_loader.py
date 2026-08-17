from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple


def get_calibration_parameters(
    resolution: str, json_path: str | Path | None = None
) -> Tuple[float, float]:
    """Return (Kc, Kv) for the given resolution.

    Args:
        resolution: Input resolution string, e.g. "640*480" or "640x480".
        json_path: Optional path to calibration_data.json.

    Raises:
        FileNotFoundError: If the calibration json file is missing.
        ValueError: If the resolution is not found or json format is invalid.

    Returns:
        Tuple[float, float]: (Kc, Kv)
    """
    target_resolution = resolution.replace("x", "*").strip()

    if json_path is None:
        json_path = (
            Path(__file__).resolve().parents[1] / "data" / "calibration_data.json"
        )
    else:
        json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        calibration_items = json.load(f)

    for item in calibration_items:
        if item.get("resolution") == target_resolution:
            params = item.get("calibration_parameters", {})
            if "Kc" not in params or "Kv" not in params:
                raise ValueError(
                    f"Missing Kc/Kv in calibration_parameters for resolution: {target_resolution}"
                )
            return float(params["Kc"]), float(params["Kv"])

    raise ValueError(f"Resolution not found in calibration file: {target_resolution}")
