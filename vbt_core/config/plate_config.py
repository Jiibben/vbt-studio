"""
plate_config — persisted plate diameters used for pixel→metre calibration.

The user measures their actual plates once (diameters vary by brand) and saves
them from the webapp settings panel. Stored as JSON; defaults below are standard
gym sizes and are merged under whatever was saved.

The store lives at ``data/plate_config.json`` by default (gitignored runtime
state), overridable with the ``VBT_PLATE_CONFIG`` environment variable.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def config_path() -> Path:
    override = os.environ.get("VBT_PLATE_CONFIG")
    return Path(override) if override else ROOT / "data" / "plate_config.json"


# kg -> diameter in mm. 25/20/15 kg follow the IPF/IWF 450 mm standard.
DEFAULTS = {"25": 450.0, "20": 450.0, "15": 400.0,
            "10": 325.0, "5": 230.0, "2.5": 190.0}

MIN_MM, MAX_MM = 50.0, 600.0


def load_plate_config() -> dict[str, float]:
    cfg = dict(DEFAULTS)
    try:
        saved = json.loads(config_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return cfg
    for key, val in saved.items():
        if key in DEFAULTS:
            try:
                cfg[key] = min(MAX_MM, max(MIN_MM, float(val)))
            except (TypeError, ValueError):
                pass
    return cfg


def save_plate_config(new: dict) -> dict[str, float]:
    """Validate + persist the config atomically; returns the merged result."""
    if not isinstance(new, dict):
        raise ValueError("Expected a JSON object of {kg: diameter_mm}.")
    cfg = load_plate_config()
    for key, val in new.items():
        if key not in DEFAULTS:
            raise ValueError(f"Unknown plate weight: {key!r}")
        try:
            mm = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid diameter for {key} kg: {val!r}")
        if not (MIN_MM <= mm <= MAX_MM):
            raise ValueError(f"Diameter for {key} kg must be "
                             f"{MIN_MM:.0f}–{MAX_MM:.0f} mm, got {mm}")
        cfg[key] = mm

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return cfg
