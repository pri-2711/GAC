"""Project configuration for the Gujarati Accent Classifier.

All accent / speaker / folder assumptions live here so the rest of the code
stays generic. To add a new accent (or enable Surti once recordings exist),
edit ACCENTS only - nothing else needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths (always relative to the project root) -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
PROCESSED_WAV_DIR = PROCESSED_DIR / "wav"
REPORTS_DIR = PROCESSED_DIR / "reports"
METADATA_CSV = DATASET_DIR / "recordings_metadata.csv"

# --- Target audio specification ----------------------------------------------
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"  # soundfile name for signed 16-bit PCM
SUPPORTED_EXTENSIONS = (".m4a", ".ogg")  # add e.g. ".mp3" here later


@dataclass(frozen=True)
class AccentConfig:
    accent: str          # canonical accent name (also the processed sub-folder)
    region: str          # region of Gujarat
    speaker_id: str      # one consistent ID per speaker
    code: str            # short prefix used in recording IDs
    folder_names: tuple  # raw-data folder names (lower-case) that identify it


# One speaker per accent at the moment. If more speakers are added later, extend
# speaker handling here (e.g. a per-subfolder speaker map).
ACCENTS: tuple[AccentConfig, ...] = (
    AccentConfig("Ahmedabad",   "Central Gujarat", "AHM_S01", "AHM", ("ahmedabad",)),
    AccentConfig("Kutchi",      "Kutch",           "KUT_S01", "KUT", ("kutch", "kutchi")),
    AccentConfig("Mehsani",     "North Gujarat",   "MEH_S01", "MEH", ("mehsani",)),
    AccentConfig("Kathiyawadi", "Saurashtra",      "KAT_S01", "KAT", ("kathiyawadi",)),
    # Pending data collection - no recordings yet. Dropping files into
    # dataset/south-Gujarat/surti/ and rerunning the pipeline is enough.
    AccentConfig("Surti",       "South Gujarat",   "SUR_S01", "SUR", ("surti",)),
)
