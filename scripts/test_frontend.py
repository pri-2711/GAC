"""Tests for the Phase 7 frontend (no model training; inference only).

Usage:  python scripts/test_frontend.py
Writes results/frontend/frontend_test_results.txt. Exit code 0 only if every test passed.

What it covers (all executed here, against the real final model):
  - valid WAV / M4A / OGG / MP3 uploads through frontend_utils.prepare_upload + classify
  - WAV with another sample rate and stereo (standardised like Phase 1)
  - invalid inputs: wrong extension, empty file, text file named .wav, WAV header with zero samples, very short audio,
    digital silence, very long audio
  - region/accent mapping consistency and probability validity
  - consistency with scripts/predict_accent.py (run as a separate process)
  - every Streamlit page renders without an exception (streamlit.testing AppTest)
NOT covered here (needs a person and a browser): clicking through the upload widget, visual layout on different screens.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

import audio_utils as au  # noqa: E402
import config as cfg  # noqa: E402
import frontend_utils as fu  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""), flush=True)


def expect_problem(name: str, fname: str, data: bytes, predictor, must_contain: str) -> None:
    try:
        fu.prepare_upload(fname, data, predictor)
        check(name, False, "no error was raised")
    except fu.AudioProblem as exc:
        check(name, must_contain.lower() in str(exc).lower(), f"message: {str(exc)[:110]}")
    except Exception as exc:                                      # an uncaught crash is a failure
        check(name, False, f"unhandled {type(exc).__name__}: {exc}")


def main() -> int:
    predictor = fu.load_predictor()
    facts = fu.model_facts(predictor)
    meta = pd.read_csv(cfg.METADATA_CSV)
    meta = meta[meta.conversion_status.astype(str).str.startswith("success")]
    ffmpeg = au.find_ffmpeg()
    tmp = Path(tempfile.mkdtemp())

    # ---- sources: real project recordings (plumbing test; they are TRAINING recordings, so accuracy is not tested here)
    rows = {a: g.iloc[0] for a, g in meta.groupby("accent", sort=False)}
    wav_row = rows[facts["accents"][0]]
    wav_path = ROOT / wav_row.processed_file_path
    m4a_row = meta[meta.original_format == "M4A"].iloc[0]
    ogg_row = meta[meta.original_format == "OGG"].iloc[0]

    # 1. valid WAV
    data = wav_path.read_bytes()
    prep = fu.prepare_upload(wav_path.name, data, predictor)
    res = fu.classify(prep, predictor)
    check("Valid WAV: accepted and classified", res["accent"] in facts["accents"], f"{res['region']} / {res['accent']}, {prep.duration_s:.1f} s")
    check("Result region equals the region mapped to the predicted accent", facts["accent_to_region"][res["accent"]] == res["region"])
    pa, pr = np.array(list(res["accent_probabilities"].values())), np.array(list(res["region_probabilities"].values()))
    check("Both heads return valid probabilities (finite, in [0,1], sum to 1)", np.isfinite(pa).all() and np.isfinite(pr).all()
          and abs(pa.sum() - 1) < 1e-4 and abs(pr.sum() - 1) < 1e-4 and (pa >= 0).all() and (pr >= 0).all())
    check("Result lists exactly the model's accents and regions", list(res["accent_probabilities"]) == facts["accents"] and list(res["region_probabilities"]) == facts["regions"])

    # 2. consistency with the existing command-line script (separate process)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict_accent.py"), str(wav_path)], capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    try:
        cli = json.loads(out.stdout[out.stdout.index("{"):])
        same = (cli["accent"] == res["accent"] and cli["region"] == res["region"] and
                max(abs(cli["accent_probabilities"][a] - res["accent_probabilities"][a]) for a in facts["accents"]) < 1e-6)
        check("Frontend result == scripts/predict_accent.py result (same file, separate process)", same,
              f"max probability difference {max(abs(cli['accent_probabilities'][a] - res['accent_probabilities'][a]) for a in facts['accents']):.2e}")
    except Exception as exc:
        check("Frontend result == scripts/predict_accent.py result (same file, separate process)", False, f"CLI output not parsable: {exc}; stderr: {out.stderr[-200:]}")

    # 3. prediction equals the Phase 6 batch prediction pipeline (processed WAV of a recording the final model has seen)
    demo = pd.read_csv(ROOT / "results" / "model_development" / "hierarchical_inference" / "final_model_demo_predictions.csv").set_index("recording_id")
    rid = wav_row.recording_id
    check("Frontend result == stored Phase 6 final-model prediction for the same recording", demo.loc[rid, "predicted_accent"] == res["accent"]
          and abs(demo.loc[rid, "region_confidence"] - res["region_confidence"]) < 1e-5)

    # 4. other valid formats (original M4A / OGG files, MP3 re-encoded from a WAV)
    for label, row in (("M4A", m4a_row), ("OGG", ogg_row)):
        p = ROOT / row.original_file_path
        try:
            pr_ = fu.prepare_upload(p.name, p.read_bytes(), predictor)
            r_ = fu.classify(pr_, predictor)
            same_as_wav = r_["accent"] == row.accent
            check(f"Valid {label} upload: decoded, standardised and classified", r_["accent"] in facts["accents"], f"{r_['accent']} ({pr_.duration_s:.1f} s, {pr_.original_sample_rate} Hz source)")
            check(f"{label} upload gives the same decision as the project's own prediction for that recording", same_as_wav, f"true {row.accent}, predicted {r_['accent']}")
        except Exception as exc:
            check(f"Valid {label} upload", False, f"{type(exc).__name__}: {exc}")
    mp3 = tmp / "sample.mp3"
    mp3_run = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3)], capture_output=True, text=True)
    if mp3.is_file():
        try:
            r_ = fu.classify(fu.prepare_upload("sample.mp3", mp3.read_bytes(), predictor), predictor)
            check("Valid MP3 upload (re-encoded from a project WAV) classified", r_["accent"] in facts["accents"], f"{r_['accent']}")
        except Exception as exc:
            check("Valid MP3 upload classified", False, f"{type(exc).__name__}: {exc}")
    else:
        check("Valid MP3 upload classified", False, f"could not create an MP3 test file: {mp3_run.stderr[:150]}")

    # 5. WAV with a different sample rate / stereo (must be standardised, not rejected or mis-read)
    for label, args in (("44.1 kHz stereo WAV", ["-ar", "44100", "-ac", "2"]), ("8 kHz mono WAV", ["-ar", "8000", "-ac", "1"])):
        f = tmp / f"{label.replace(' ', '_')}.wav"
        subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(wav_path), *args, str(f)], capture_output=True)
        try:
            p_ = fu.prepare_upload(f.name, f.read_bytes(), predictor)
            check(f"{label} is standardised to 16 kHz mono and classified", p_.sample_rate == 16000 and fu.classify(p_, predictor)["accent"] in facts["accents"],
                  f"duration {p_.duration_s:.1f} s, original {p_.original_sample_rate} Hz")
        except Exception as exc:
            check(f"{label} standardised", False, f"{type(exc).__name__}: {exc}")

    # 6. invalid inputs
    expect_problem("Unsupported extension (.txt) is rejected with a clear message", "notes.txt", b"hello", predictor, "not a supported file type")
    expect_problem("Empty file (0 bytes) is rejected", "empty.wav", b"", predictor, "empty")
    expect_problem("Text file renamed to .wav is rejected gracefully (no crash)", "fake.wav", b"this is not audio at all " * 50, predictor, "could not be decoded")
    expect_problem("Corrupted M4A (random bytes) is rejected gracefully", "broken.m4a", os.urandom(4000), predictor, "could not be decoded")
    empty_wav = tmp / "zero_samples.wav"
    sf.write(str(empty_wav), np.zeros(0, dtype="float32"), 16000, subtype="PCM_16")
    expect_problem("Valid WAV header with zero samples ('empty audio') is rejected", "zero_samples.wav", empty_wav.read_bytes(), predictor, "")
    short = tmp / "short.wav"
    sf.write(str(short), (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(0.3 * 16000)) / 16000)).astype("float32"), 16000, subtype="PCM_16")
    expect_problem("Very short audio (0.3 s) is rejected", "short.wav", short.read_bytes(), predictor, "too short")
    silent = tmp / "silence.wav"
    sf.write(str(silent), np.zeros(10 * 16000, dtype="float32"), 16000, subtype="PCM_16")
    expect_problem("Digital silence (10 s) is rejected", "silence.wav", silent.read_bytes(), predictor, "silent")
    longf = tmp / "long.wav"
    sf.write(str(longf), (0.05 * np.random.default_rng(0).standard_normal(int(190 * 16000))).astype("float32"), 16000, subtype="PCM_16")
    expect_problem("Over-long audio (190 s) is rejected", "long.wav", longf.read_bytes(), predictor, "too long")
    quiet = tmp / "quiet_noise.wav"
    sf.write(str(quiet), (1e-4 * np.random.default_rng(1).standard_normal(8 * 16000)).astype("float32"), 16000, subtype="PCM_16")
    try:
        pq = fu.prepare_upload("quiet_noise.wav", quiet.read_bytes(), predictor)
        check("Near-silent noise is accepted with a 'very quiet' warning (not a crash)", any("very quiet" in w for w in pq.warnings), f"warnings: {pq.warnings}")
    except fu.AudioProblem as exc:
        check("Near-silent noise handled (rejected with a clear message)", True, str(exc)[:100])
    three = tmp / "three_s.wav"
    y, sr = sf.read(str(wav_path))
    sf.write(str(three), y[: 3 * sr].astype("float32"), sr, subtype="PCM_16")
    p3 = fu.prepare_upload("three_s.wav", three.read_bytes(), predictor)
    check("Short but valid audio (3 s) is accepted with a warning", any("Short recording" in w for w in p3.warnings) and fu.classify(p3, predictor)["accent"] in facts["accents"])

    # 7. every page renders; sidebar uses tab-style buttons (no radio); tabs switch pages (Streamlit AppTest; no upload interaction possible)
    from streamlit.testing.v1 import AppTest
    pages = {"Home": "Supported accents and their regions", "Classify audio": "Classify a recording", "About the model": "The 78 numerical acoustic features",
             "Dataset & limitations": "The key limitation: one speaker per accent"}

    def texts(at):
        return " ".join(str(m.value) for m in at.markdown)

    for page, marker in pages.items():
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        if page != "Home":
            at.sidebar.button(key=f"nav_{page}").click().run()
        check(f"App page renders without exception: {page}", not at.exception, "; ".join(str(e.value)[:120] for e in at.exception))
        check(f"Clicking the '{page}' tab shows that page", marker in texts(at), f"marker: {marker}")
        check(f"'{page}' tab is the only active (primary) tab", [b.key for b in at.sidebar.button if b.proto.type == "primary"] == [f"nav_{page}"])
    check("Sidebar navigation uses tab buttons and has no radio widget", len(at.sidebar.radio) == 0 and len(at.sidebar.button) == len(pages))
    check("Gender-related limitation is shown on the Dataset & limitations page, without claiming proven bias",
          "Gender-independent performance has not been established" in texts(at) and "not</b> an experimentally demonstrated bias" in texts(at))

    n_ok = sum(ok for _, ok, _ in results)
    out_dir = ROOT / "results" / "frontend"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 7 frontend tests (executed by scripts/test_frontend.py)", "=" * 60,
             *[f"[{'PASS' if ok else 'FAIL'}] {n}" + (f" -> {d}" if d else "") for n, ok, d in results], "", f"{n_ok}/{len(results)} tests passed"]
    (out_dir / "frontend_test_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(results)} tests passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
