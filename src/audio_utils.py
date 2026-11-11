"""Reusable audio standardisation utilities (Phase 1).

Pipeline: discover -> convert (FFmpeg) -> validate (SoundFile) -> metadata (pandas).
Raw recordings are never modified; output goes only to dataset/processed/wav/.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import config as cfg

METADATA_COLUMNS = [
    "recording_id", "speaker_id", "accent", "region", "recording_type",
    "original_file_path", "processed_file_path", "original_format",
    "duration_seconds", "sample_rate", "channels", "audio_subtype",
    "conversion_status",
]


# --------------------------------------------------------------------------- #
# FFmpeg
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str:
    """Return an FFmpeg executable: PATH first, then the imageio-ffmpeg binary."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg not found. Install it on PATH or run: pip install imageio-ffmpeg"
        ) from exc


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class Recording:
    source_path: Path
    accent: str
    region: str
    speaker_id: str
    recording_id: str
    original_format: str
    processed_path: Path


def _lookup_accent(relative_parts: tuple[str, ...]) -> cfg.AccentConfig | None:
    """Match the accent from the actual folder location (deepest folder first)."""
    for folder in reversed(relative_parts[:-1]):  # exclude the filename itself
        for accent_cfg in cfg.ACCENTS:
            if folder.lower() in accent_cfg.folder_names:
                return accent_cfg
    return None


def _safe_stem(stem: str) -> str:
    """Keep the original filename; only whitespace becomes '_' for portability."""
    return re.sub(r"\s+", "_", stem.strip())


def discover_recordings(dataset_dir: Path = cfg.DATASET_DIR,
                        processed_dir: Path = cfg.PROCESSED_DIR):
    """Recursively scan raw data.

    Returns (recordings, unmapped_files). Files under processed_dir are skipped;
    files in folders matching no configured accent are returned in unmapped_files
    (reported, never silently dropped or guessed).
    """
    dataset_dir, processed_dir = Path(dataset_dir).resolve(), Path(processed_dir).resolve()
    recordings: list[Recording] = []
    unmapped: list[Path] = []
    used_outputs: set[Path] = set()

    candidates = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in cfg.SUPPORTED_EXTENSIONS
        and processed_dir not in p.resolve().parents
    )

    per_accent_index: dict[str, int] = {}
    seen_ids: set[str] = set()
    for path in candidates:
        rel_parts = path.relative_to(dataset_dir).parts
        accent_cfg = _lookup_accent(rel_parts)
        if accent_cfg is None:
            unmapped.append(path)
            continue

        # Recording number: taken from the filename (e.g. "-03-") so IDs stay
        # stable when files are added later; falls back to a running index.
        per_accent_index[accent_cfg.accent] = per_accent_index.get(accent_cfg.accent, 0) + 1
        match = re.search(r"(\d+)", path.stem)
        number = int(match.group(1)) if match else per_accent_index[accent_cfg.accent]
        recording_id = f"{accent_cfg.speaker_id}_R{number:02d}"
        if recording_id in seen_ids:  # duplicate number -> guarantee uniqueness
            recording_id = f"{accent_cfg.speaker_id}_R{number:02d}_{per_accent_index[accent_cfg.accent]:02d}"
        seen_ids.add(recording_id)

        out_dir = cfg.PROCESSED_WAV_DIR / accent_cfg.accent
        out_path = out_dir / f"{_safe_stem(path.stem)}.wav"
        if out_path in used_outputs:  # same stem, different extension
            out_path = out_dir / f"{_safe_stem(path.stem)}_{path.suffix.lower().lstrip('.')}.wav"
        used_outputs.add(out_path)

        recordings.append(Recording(
            source_path=path, accent=accent_cfg.accent, region=accent_cfg.region,
            speaker_id=accent_cfg.speaker_id, recording_id=recording_id,
            original_format=path.suffix.lower().lstrip(".").upper(),
            processed_path=out_path,
        ))
    return recordings, unmapped


# --------------------------------------------------------------------------- #
# Inspection of ORIGINAL files
# --------------------------------------------------------------------------- #
def inspect_original(path: Path) -> dict:
    """Sample rate / channels / duration of an original file.

    SoundFile is tried first (works for OGG); FFmpeg's stream info is the
    fallback for M4A, which libsndfile cannot read.
    """
    try:
        info = sf.info(str(path))
        return {"sample_rate": info.samplerate, "channels": info.channels,
                "duration_seconds": round(info.duration, 3), "reader": "soundfile"}
    except Exception:
        pass
    result = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = result.stderr
    out = {"sample_rate": None, "channels": None, "duration_seconds": None, "reader": "ffmpeg"}
    if m := re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text):
        h, mnt, s = m.groups()
        out["duration_seconds"] = round(int(h) * 3600 + int(mnt) * 60 + float(s), 3)
    if m := re.search(r"Audio:.*?,\s*(\d+) Hz,\s*([^,]+)", text):
        out["sample_rate"] = int(m.group(1))
        layout = m.group(2).strip().lower()
        out["channels"] = {"mono": 1, "stereo": 2}.get(layout, layout)
    return out


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_to_wav(source: Path, destination: Path, ffmpeg: str | None = None) -> tuple[bool, str]:
    """Convert one file to 16 kHz / mono / PCM s16 WAV. Returns (ok, message).

    Only format conversion: no filters, no denoising, no trimming. Writes to a
    temporary file first so an interrupted run never leaves a half-written WAV.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".part.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(source), "-vn",
           "-ac", str(cfg.TARGET_CHANNELS), "-ar", str(cfg.TARGET_SAMPLE_RATE),
           "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            return False, f"ffmpeg exit {proc.returncode}: {proc.stderr.strip()[:300]}"
        os.replace(tmp, destination)  # only ever replaces our own processed output
        return True, "converted"
    except Exception as exc:  # keep the batch going
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Validation of PROCESSED files
# --------------------------------------------------------------------------- #
def validate_wav(path: Path) -> dict:
    """Independently inspect a processed WAV. 'problems' is empty when valid."""
    result = {"exists": path.is_file(), "readable": False, "sample_rate": None,
              "channels": None, "audio_subtype": None, "duration_seconds": None,
              "problems": []}
    if not result["exists"]:
        result["problems"].append("file does not exist")
        return result
    try:
        info = sf.info(str(path))
        result.update(sample_rate=info.samplerate, channels=info.channels,
                      audio_subtype=info.subtype, duration_seconds=round(info.duration, 3))
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)  # full decode
        result["readable"] = True
    except Exception as exc:
        result["problems"].append(f"unreadable/corrupted: {exc}")
        return result

    if info.format != "WAV":
        result["problems"].append(f"container is {info.format}, expected WAV")
    if info.samplerate != cfg.TARGET_SAMPLE_RATE:
        result["problems"].append(f"sample rate {info.samplerate} != {cfg.TARGET_SAMPLE_RATE}")
    if info.channels != cfg.TARGET_CHANNELS:
        result["problems"].append(f"channels {info.channels} != {cfg.TARGET_CHANNELS}")
    if info.subtype != cfg.TARGET_SUBTYPE:
        result["problems"].append(f"subtype {info.subtype} != {cfg.TARGET_SUBTYPE}")
    if not info.duration > 0 or data.shape[0] == 0:
        result["problems"].append("duration is zero")
    if data.shape[0] != info.frames:
        result["problems"].append(f"decoded {data.shape[0]} frames, header says {info.frames}")
    if not np.isfinite(data).all():
        result["problems"].append("contains non-finite samples")
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class RecordingResult:
    recording: Recording
    conversion_status: str       # converted | already_converted | failed
    message: str = ""
    validation: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.validation) and not self.validation["problems"]


def process_recordings(recordings: list[Recording], force: bool = False) -> list[RecordingResult]:
    """Convert (skipping up-to-date, valid outputs unless force) and validate each file."""
    ffmpeg = find_ffmpeg()
    results = []
    for rec in recordings:
        status, message = "failed", ""
        up_to_date = (not force and rec.processed_path.is_file()
                      and rec.processed_path.stat().st_mtime >= rec.source_path.stat().st_mtime
                      and not validate_wav(rec.processed_path)["problems"])
        if up_to_date:
            status, message = "already_converted", "valid output exists; skipped"
        else:
            ok, message = convert_to_wav(rec.source_path, rec.processed_path, ffmpeg)
            status = "converted" if ok else "failed"
        validation = validate_wav(rec.processed_path) if status != "failed" else {
            "problems": ["conversion failed"], "exists": rec.processed_path.is_file()}
        results.append(RecordingResult(rec, status, message, validation))
    return results


def _rel(path: Path) -> str:
    return path.resolve().relative_to(cfg.PROJECT_ROOT).as_posix()


def build_metadata(results: list[RecordingResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rec, v = r.recording, r.validation
        ok = r.conversion_status != "failed"
        status = ("failed" if not ok else
                  "validation_failed" if not r.valid else
                  "success")
        rows.append({
            "recording_id": rec.recording_id, "speaker_id": rec.speaker_id,
            "accent": rec.accent, "region": rec.region,
            "recording_type": "",  # not known - deliberately left empty
            "original_file_path": _rel(rec.source_path),
            "processed_file_path": _rel(rec.processed_path) if ok else "",
            "original_format": rec.original_format,
            "duration_seconds": v.get("duration_seconds") if ok else "",
            "sample_rate": v.get("sample_rate") if ok else "",
            "channels": v.get("channels") if ok else "",
            "audio_subtype": v.get("audio_subtype") if ok else "",
            "conversion_status": status,
        })
    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def save_metadata(df: pd.DataFrame, path: Path = cfg.METADATA_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def summarise(results: list[RecordingResult], discovered_unmapped: list[Path]) -> dict:
    """Counts for the validation report; every configured accent is listed (Surti -> 0)."""
    per_accent = {a.accent: 0 for a in cfg.ACCENTS}
    for r in results:
        if r.valid:
            per_accent[r.recording.accent] += 1
    return {
        "total_discovered": len(results) + len(discovered_unmapped),
        "mapped": len(results),
        "unmapped_files": [str(p) for p in discovered_unmapped],
        "converted_ok": sum(r.conversion_status != "failed" for r in results),
        "conversion_failed": [r.recording.source_path.name for r in results if r.conversion_status == "failed"],
        "validation_failed": [(r.recording.recording_id, r.validation["problems"])
                              for r in results if r.conversion_status != "failed" and not r.valid],
        "validated_ok": sum(r.valid for r in results),
        "per_accent_valid": per_accent,
    }


def format_report(summary: dict) -> str:
    lines = ["=== Phase 1 validation report ===",
             f"Total files discovered      : {summary['total_discovered']}",
             f"Successfully converted      : {summary['converted_ok']}",
             f"Failed conversions          : {len(summary['conversion_failed'])} {summary['conversion_failed'] or ''}",
             f"Validation failures         : {len(summary['validation_failed'])} {summary['validation_failed'] or ''}",
             f"Files in unmapped folders   : {len(summary['unmapped_files'])} {summary['unmapped_files'] or ''}",
             "", "Valid recordings per accent:"]
    for accent, n in summary["per_accent_valid"].items():
        note = " (pending future collection)" if n == 0 else ""
        lines.append(f"  {accent:<12}: {n}{note}")
    lines += ["", f"Total validated: {summary['validated_ok']}"]
    return "\n".join(lines)


def run_pipeline(force: bool = False):
    """Full Phase 1 run. Returns (results, metadata_df, summary, report_text)."""
    recordings, unmapped = discover_recordings()
    results = process_recordings(recordings, force=force)
    df = build_metadata(results)
    save_metadata(df)
    summary = summarise(results, unmapped)
    report = format_report(summary)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORTS_DIR / "validation_report.txt").write_text(report + "\n", encoding="utf-8")
    return results, df, summary, report
