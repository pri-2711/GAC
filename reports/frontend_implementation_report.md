# Phase 7 Report: Streamlit Frontend for the Gujarati Accent Classifier

*Implementation report for the demonstration application*

## 1. Summary

A Streamlit web application (`app.py`) lets a user upload a speech recording (WAV, MP3, M4A or OGG), listen to it, see its waveform, and obtain the **region and accent predicted by the real final model** (the Phase 6 MLP with hierarchical heads). The application is a thin interface: audio standardisation, feature extraction, normalisation and the neural network are all performed by the project's existing inference pipeline (`model_dev.AccentPredictor`), the same code that backs `scripts/predict_accent.py`. Nothing is hard-coded, re-implemented or retrained, and no earlier result or model file was changed.

> **Limitation shown prominently in the app:** the model was trained on 35 recordings from 5 speakers (one speaker per accent). It is an **experimental prototype that has not been validated on unseen speakers**; its output reflects similarity to those five voices and recording setups, not a verified accent recognition. The probability bars are raw softmax scores, **not calibrated confidence**.

**Launch (from the project root):**

```bash
pip install -r requirements.txt     # once
streamlit run app.py                # opens http://localhost:8501
```

## 2. What was inspected first (verified, not assumed)

| Item | Finding |
|---|---|
| Final model bundle | `results/model_development/final_model/bundle/` with `model.keras`, `preprocessing.npz` (saved training-set normalisation), `labels.json` (feature columns, region→accent mapping), `history.csv` |
| Command-line script | `scripts/predict_accent.py` (the real filename) – calls `model_dev.AccentPredictor` |
| Inference pipeline | `src/model_dev.py::AccentPredictor` → FFmpeg standardisation (`audio_utils.convert_to_wav`) → `mfcc_features.extract_frame_features` / `recording_level_stats` (78 Phase 3 features) → saved z-score → network → `hierarchical_decode` |
| Dependencies | Streamlit 1.45.1 was installed but not pinned (now added to `requirements.txt`); Altair ships with Streamlit; Plotly is **not** installed, so charts use Altair and matplotlib; FFmpeg is available through `imageio-ffmpeg` (decodes M4A, OGG and MP3) |
| Project structure | Source modules in `src/`, runners and tools in `scripts/`, reports in `reports/`, results in `results/` |

## 3. Architecture of the application

| Component | Role |
|---|---|
| `app.py` | The Streamlit user interface: styling, navigation, pages, charts. Contains no signal-processing or model code |
| `src/frontend_utils.py` | Upload validation and the call into the pipeline: `prepare_upload` (type/size/empty/decode/duration/silence checks, then standardisation through the predictor), `classify`, and read-only facts shown in the app (`model_facts`, `dataset_facts`) |
| `src/model_dev.py::AccentPredictor` | The existing inference pipeline (see below for the small additive change) |
| `.streamlit/config.toml` | Light theme and 50 MB upload limit |
| `scripts/test_frontend.py` | Reproducible tests (Section 7) |

**Data flow for one upload:** uploaded bytes → temporary file → `AccentPredictor.load_audio` (FFmpeg → 16 kHz mono) → validation (duration, silence, level) → *Classify* button → `AccentPredictor.predict_audio` (78 features → saved normalisation → MLP → hierarchical decoding) → result card and charts. The model is loaded once per server process (`st.cache_resource`).

### The one change to existing code (additive, backwards compatible)

`AccentPredictor` previously read any `.wav` file as it was and converted only non-WAV files. A WAV upload at another sample rate (e.g. 44.1 kHz) or in stereo would therefore not have been standardised, which would break the "exact same preprocessing" requirement for user uploads. `src/model_dev.py` now has:

* a public `load_audio(path)`: files that are already 16 kHz mono WAV are read unchanged (so every existing prediction is identical); everything else, including WAVs with another rate or channel count, goes through the Phase 1 FFmpeg step;
* a public `predict_audio(y, sr)` (the former body of `predict_file`, which now calls it).

Verified: the frontend result for a project WAV is bit-identical to `scripts/predict_accent.py` (max probability difference 0.00e+00) and to the stored Phase 6 final-model prediction. No saved model, result file or earlier report was modified, and no training was run.

## 4. Interface

* **Navigation:** sidebar with four pages – **Home**, **Classify audio**, **About the model**, **Dataset & limitations** – plus a permanent warning and model summary in the sidebar.
* **Home:** title, project introduction, three summary tiles (recordings, accents, speakers), five accent cards with their regions (colours are the project's accent palette), a "how it works" card, a "please read" card and the warning banner.
* **Classify audio:** drag-and-drop uploader (WAV, MP3, M4A, OGG); on upload the file is validated and standardised with a spinner; audio player (listen before predicting), information chips (name, format, original sample rate and channels, duration, level), any warnings, a waveform; a **Classify accent** button; a loading spinner during inference; the result.
* **Result:** large predicted accent and region, the model output score of that accent and region, the warning, two bar charts (accent head and region head scores), an expandable explanation of the hierarchical decision, and the raw model output. Labelling is explicit: *"softmax scores of a small network – not calibrated confidence"*.
* **About the model:** the 78 numerical features (13 MFCC, delta and delta-delta × mean and standard deviation), the MLP layer by layer, the two heads and the hierarchical decision, "trained entirely from scratch", the pipeline steps, and what the cross-validation score means (within-speaker).
* **Dataset & limitations:** 35 recordings, 5 accents, 5 speakers, 7 recordings per speaker (all read from the metadata, not typed in), a per-accent table, the one-speaker-per-accent limitation, and future scope.
* **Design:** a navy–teal palette with a light theme, card layout, consistent typography (system sans-serif stack, no web-font dependency so it works offline), rounded cards and result sections, responsive columns and a narrow-screen style rule.

Where the app displays numbers about the model or data (accents, regions, parameter count, number of recordings, cross-validation score) they are read from the model bundle, the metadata and the Phase 6 results at run time.

## 5. Input validation and error handling

| Situation | Behaviour |
|---|---|
| Unsupported extension | Rejected with the list of supported types |
| Empty file (0 bytes) | Rejected: "The uploaded file is empty" |
| File that is not really audio / corrupted | "The file could not be decoded as audio…" (decoder message shortened); no stack trace |
| Valid header, zero samples | Rejected: "No audio samples were found" |
| Shorter than 1 s | Rejected: "too short" |
| Digital silence | Rejected: "The recording is silent" |
| Longer than 180 s or larger than 50 MB | Rejected |
| Under 5 s, over 30 s, or very quiet (< −55 dBFS RMS) | Accepted with a visible warning |
| Unexpected exception | Caught and shown as a message; the app keeps running |

Decoding is done by FFmpeg (through the project's `convert_to_wav`), so MP3, M4A, OGG and WAV are all supported. Thresholds are constants at the top of `src/frontend_utils.py`.

## 6. Integration details

* **Model:** `results/model_development/final_model/bundle/model.keras` – the final MLP (78 features → Dense 64 → BatchNorm → Dropout 0.30 → Dense 32 → Dropout 0.20 → shared Dense 64 → region head and accent head), 10,026 trainable parameters, trained from scratch.
* **Normalisation:** the saved training-set statistics in `preprocessing.npz` (applied inside `AccentPredictor`).
* **Features:** `mfcc_features.extract_frame_features` and `recording_level_stats` (Phase 3 code), restricted to the schema's 78 columns in the stored order.
* **Region/accent mapping:** read from `labels.json`; hierarchical decoding picks the region, then the accent among that region's accents. With one accent per region this is trivial; the app says so.
* **Probabilities:** the real softmax outputs of both heads, shown as "output scores".

## 7. Testing

### Tests actually performed (`python scripts/test_frontend.py`, 27 of 27 passed; log in `results/frontend/frontend_test_results.txt`)

All tests ran against the real final model and real project recordings.

| Area | Tests (all passed) |
|---|---|
| Valid WAV | accepted and classified; region equals the region mapped to the predicted accent; both heads return valid probabilities (finite, in [0,1], sum to 1); the result lists exactly the model's accents and regions |
| Consistency | frontend result equals `scripts/predict_accent.py` run as a separate process (probability difference 0.00e+00); equals the stored Phase 6 final-model prediction for the same recording |
| Other formats | original M4A upload and original OGG upload decoded, standardised and classified; MP3 (re-encoded from a project WAV) classified; 44.1 kHz stereo WAV and 8 kHz mono WAV standardised to 16 kHz mono and classified |
| Invalid input | `.txt` file rejected; empty file rejected; text file named `.wav` rejected gracefully; random bytes named `.m4a` rejected gracefully; WAV with zero samples rejected; 0.3 s audio rejected; 10 s of digital silence rejected; 190 s audio rejected |
| Edge cases accepted with a warning | near-silent noise (−80 dBFS) → "very quiet" warning; 3 s of speech → "short recording" warning |
| Rendering | all four pages render without an exception (Streamlit `AppTest`) |

Additionally the server was started with `streamlit run app.py --server.headless true` and answered its health check (`ok`) and the main page (HTTP 200) before being stopped.

**Important:** the recordings used in these tests are project recordings that the final model was fitted on, so their predictions being "correct" tests the **integration** (decoding, standardisation, features, model, mapping), **not** accuracy or generalisation.

### Tests not performed

* **Manual interaction in a real browser:** clicking through the upload widget, the audio player, the Classify button and the charts. (`AppTest` cannot drive the file uploader, so the upload flow was tested through the same service functions the app calls, not through the widget.)
* **Visual inspection** of the layout, colours and the narrow-screen (mobile) style in real browsers; no screenshots were taken.
* **Load or concurrent-user testing**, uploads near the 50 MB limit through the browser, and deployment testing (hosted or Docker).
* **Recording from a microphone** (not implemented) and accessibility auditing.
* Testing with recordings from **speakers outside the dataset** (none are available), so no statement about real-world behaviour is made.

## 8. Remaining limitations

1. **One speaker per accent:** the model cannot separate accent from speaker; any new speaker is matched to the most similar of five known voices. Results are not validated on unseen speakers.
2. **Scores are not calibrated probabilities** and are usually close to 0% or 100%.
3. **Each region has one accent:** distinguishing several accents within a region is not demonstrated (the decoder supports it).
4. **Recording-condition shortcuts:** the five groups differ in format, sample rate and noise level; the model may use these.
5. Best used with 8–18 s of clear speech from one speaker; short or noisy clips give less meaningful output.
6. First start-up takes a few seconds while TensorFlow and the model load.

## 9. Files

**Created:** `app.py`, `src/frontend_utils.py`, `scripts/test_frontend.py`, `.streamlit/config.toml`, `results/frontend/frontend_test_results.txt`, `reports/frontend_implementation_report.md`.
**Modified:** `src/model_dev.py` (additive: `load_audio`, `predict_audio`, WAV standardisation), `requirements.txt` (adds `streamlit==1.45.1`), `README.md`, `reports/README.md`, `scripts/validate_project.py` (Phase 7 checks added; none removed).

## 10. Suggested demonstration flow

1. Open **Home**: the five accents and regions, then the warning.
2. **Classify audio** → upload a project recording (e.g. from `dataset/processed/wav/…` or an original M4A/OGG): play it, show the waveform, classify, explain region head, accent head and the hierarchical decision. Mention that project recordings were seen during training.
3. Upload an invalid or silent file to show the error handling.
4. **About the model** and **Dataset & limitations**: the 78 features, the MLP, and why the one-speaker-per-accent limitation prevents any generalisation claim.
